"""
CXR Analysis Server  v4.0
==========================
Two-stage pipeline:
  Stage 1 — TorchXRayVision DenseNet-121   fast 18-pathology classifier  (~2 s, CPU)
  Stage 2 — CheXOne (Qwen2.5-VL-3B, CXR fine-tuned, 4-bit NF4)          (~20 s, GPU)
           (alt: Qwen2-VL-7B-Instruct 4-bit, llamacpp GGUF)

Hardware target: Ryzen 7 / 32 GB RAM / RTX 5060 8 GB VRAM
"""

import os, io, re, uuid, time, logging, threading, base64
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request

# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════
CONFIG = {
    "SECRET_TOKEN":     os.getenv("CXR_SECRET_TOKEN", "change-me-to-a-strong-token"),
    "UPLOAD_DIR":       Path("uploads"),
    "MAX_FILE_SIZE_MB": 20,
    "HOST":             os.getenv("CXR_HOST", "0.0.0.0"),
    "PORT":             int(os.getenv("CXR_PORT", "8765")),

    # VLM backend ─────────────────────────────────────────────
    #  "chexone"  : CheXOne 4B CXR-fine-tuned 4-bit  (GPU ~20s)  ← default
    #  "qwen2vl"  : Qwen2-VL-7B-Instruct 4-bit        (GPU ~30s)
    #  "llamacpp" : GGUF hybrid GPU+CPU               (GPU+CPU ~5-15 min)
    "VLM_BACKEND": os.getenv("VLM_BACKEND", "chexone"),

    # llama.cpp settings (only used when VLM_BACKEND=llamacpp)
    "LLAMACPP_MODEL_PATH":   os.getenv(
        "LLAMACPP_MODEL_PATH",
        str(Path("models") / "Llama-3.2-11B-Vision-Instruct-Q4_K_M.gguf")
    ),
    "LLAMACPP_N_GPU_LAYERS": int(os.getenv("LLAMACPP_N_GPU_LAYERS", "35")),
    "LLAMACPP_N_CTX":        int(os.getenv("LLAMACPP_N_CTX", "8192")),
}

CONFIG["UPLOAD_DIR"].mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cxr-server")

# ═══════════════════════════════════════════════════════════════
# App
# ═══════════════════════════════════════════════════════════════
app = FastAPI(title="CXR Analysis Server", version="3.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ═══════════════════════════════════════════════════════════════
# Job store  (in-memory, daemon-thread safe)
# ═══════════════════════════════════════════════════════════════
jobs: dict = {}   # job_id → {"status": str, "result": dict|None}

# ═══════════════════════════════════════════════════════════════
# Model singletons  (lazy-loaded, cached for lifetime of process)
# ═══════════════════════════════════════════════════════════════
_xrv_model = None
_vlm_model  = None
_vlm_proc   = None


def get_xrv():
    global _xrv_model
    if _xrv_model is None:
        logger.info("Loading TorchXRayVision DenseNet-121 …")
        import torchxrayvision as xrv
        _xrv_model = xrv.models.DenseNet(weights="densenet121-res224-all")
        _xrv_model.eval()
        logger.info("XRV ready.")
    return _xrv_model


def get_vlm():
    """Load the configured VLM once and keep it in memory."""
    global _vlm_model, _vlm_proc
    if _vlm_model is not None:
        return _vlm_model, _vlm_proc

    backend = CONFIG["VLM_BACKEND"]

    # ── CheXOne  4B CXR-fine-tuned 4-bit NF4 (default) ───────
    if backend == "chexone":
        import torch
        from transformers import (
            Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
        )
        name = "StanfordAIMI/CheXOne"
        logger.info(f"Loading {name} (4-bit NF4, device_map=auto) …")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        _vlm_proc  = AutoProcessor.from_pretrained(
            name, min_pixels=256*28*28, max_pixels=512*512
        )
        _vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            name, quantization_config=bnb, device_map="auto",
        ).eval()
        logger.info("CheXOne ready.")

    # ── Qwen2-VL-7B  4-bit NF4 ────────────────────────────────
    elif backend == "qwen2vl":
        import torch
        from transformers import (
            Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
        )
        name = "Qwen/Qwen2-VL-7B-Instruct"
        logger.info(f"Loading {name} (4-bit NF4, device_map=auto) …")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        _vlm_proc  = AutoProcessor.from_pretrained(name)
        _vlm_model = Qwen2VLForConditionalGeneration.from_pretrained(
            name, quantization_config=bnb, device_map="auto",
        ).eval()
        logger.info("Qwen2-VL-7B ready.")

    # ── llama.cpp GGUF  GPU+CPU hybrid ────────────────────────
    elif backend == "llamacpp":
        from llama_cpp import Llama
        path = CONFIG["LLAMACPP_MODEL_PATH"]
        if not Path(path).exists():
            raise FileNotFoundError(
                f"GGUF model not found: {path}\n"
                "Download Llama-3.2-11B-Vision-Instruct-Q4_K_M.gguf "
                "from https://huggingface.co/bartowski/Llama-3.2-11B-Vision-Instruct-GGUF "
                "and place it in the models/ folder."
            )
        logger.info(
            f"Loading GGUF {Path(path).name} "
            f"(n_gpu_layers={CONFIG['LLAMACPP_N_GPU_LAYERS']}) …"
        )
        _vlm_model = Llama(
            model_path=path,
            n_gpu_layers=CONFIG["LLAMACPP_N_GPU_LAYERS"],
            n_ctx=CONFIG["LLAMACPP_N_CTX"],
            verbose=False,
        )
        _vlm_proc = None
        logger.info("llama.cpp model ready.")

    else:
        raise ValueError(f"Unknown VLM_BACKEND: '{backend}'. Use 'chexone', 'qwen2vl', or 'llamacpp'.")

    return _vlm_model, _vlm_proc


# ═══════════════════════════════════════════════════════════════
# Auth
# ═══════════════════════════════════════════════════════════════
def verify_token(token: Optional[str] = None):
    if CONFIG["SECRET_TOKEN"] == "change-me-to-a-strong-token":
        return True
    if token != CONFIG["SECRET_TOKEN"]:
        raise HTTPException(status_code=401, detail="Invalid token")
    return True


# ═══════════════════════════════════════════════════════════════
# Stage 1 — TorchXRayVision
# ═══════════════════════════════════════════════════════════════
def _preprocess_xrv(image: Image.Image):
    import torch, torchvision, torchxrayvision as xrv
    img = np.array(image.convert("L"))
    img = xrv.datasets.normalize(img, 255)
    if img.ndim == 2:
        img = img[None, ...]
    tf = torchvision.transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224),
    ])
    return torch.from_numpy(tf(img)).unsqueeze(0)


def run_xrv(image: Image.Image) -> dict:
    import torch
    model = get_xrv()
    tensor = _preprocess_xrv(image)
    with torch.no_grad():
        out = model(tensor)
    probs = dict(zip(model.pathologies, out[0].detach().numpy().tolist()))
    return _xrv_report(probs)


def _xrv_report(probs: dict) -> dict:
    HIGH   = 0.55
    MOD    = 0.30
    CRIT_LOW = 0.15
    CRITICAL = {"Pneumothorax", "Mass", "Nodule", "Lung Lesion"}

    DESCRIPTIONS = {
        "Atelectasis":              "Atelectatic changes noted",
        "Consolidation":            "Consolidation present, may indicate pneumonia or other infiltrative process",
        "Infiltration":             "Infiltrative opacity observed",
        "Pneumothorax":             "Pneumothorax identified",
        "Edema":                    "Pulmonary edema present",
        "Emphysema":                "Emphysematous changes noted",
        "Fibrosis":                 "Fibrotic changes present",
        "Effusion":                 "Pleural effusion noted",
        "Pneumonia":                "Findings suggestive of pneumonia",
        "Pleural_Thickening":       "Pleural thickening observed",
        "Cardiomegaly":             "Cardiac silhouette enlarged, suggesting cardiomegaly",
        "Nodule":                   "Pulmonary nodule identified — clinical correlation and follow-up recommended",
        "Mass":                     "Pulmonary mass identified — further evaluation with CT recommended",
        "Hernia":                   "Hiatal hernia noted",
        "Lung Lesion":              "Lung lesion identified",
        "Fracture":                 "Fracture identified",
        "Lung Opacity":             "Lung opacity present",
        "Enlarged Cardiomediastinum": "Enlarged cardiomediastinum noted",
    }

    findings, impressions, alerts = [], [], []
    high_conf, mod_conf = {}, {}

    for name, prob in probs.items():
        desc = DESCRIPTIONS.get(name, f"{name} identified")
        if prob >= HIGH:
            high_conf[name] = prob
            findings.append(f"- **{name}** ({prob:.1%}): {desc}.")
            impressions.append(name)
            if name in CRITICAL:
                alerts.append(f"⚠️ CRITICAL: {name} detected ({prob:.1%})")
        elif prob >= MOD:
            mod_conf[name] = prob
            findings.append(f"- {name} ({prob:.1%}): {desc}. Recommend clinical correlation.")
            if name in CRITICAL:
                alerts.append(f"🔍 SUSPICIOUS: {name} possible ({prob:.1%}) — further evaluation recommended")
        elif prob >= CRIT_LOW and name in CRITICAL:
            mod_conf[name] = prob
            findings.append(f"- {name} ({prob:.1%}): Low probability but cannot exclude. Clinical correlation advised.")
            alerts.append(f"🔍 LOW SIGNAL: {name} ({prob:.1%}) — flagged for review")

    if not findings:
        findings.append("- No acute cardiopulmonary abnormality identified.")
        impressions.append("Normal study")

    return {
        "findings":          "\n".join(findings),
        "impression":        ", ".join(impressions) if impressions else "Normal study",
        "alerts":            alerts,
        "high_confidence":   high_conf,
        "moderate_confidence": mod_conf,
        "all_probabilities": probs,
    }


# ═══════════════════════════════════════════════════════════════
# Stage 2 — VLM radiology report
# ═══════════════════════════════════════════════════════════════
def _build_prompt(xrv_report: dict | None) -> str:
    """
    Build a structured radiology prompt.
    XRV pathology findings are injected so the VLM knows which areas to focus on.
    """
    xrv_context = ""
    if xrv_report:
        high = list(xrv_report.get("high_confidence", {}).keys())
        mod  = list(xrv_report.get("moderate_confidence", {}).keys())
        if high:
            xrv_context += f"\nHigh-confidence preliminary findings: {', '.join(high)}."
        if mod:
            xrv_context += f"\nPossible (moderate confidence): {', '.join(mod)}."
        if xrv_context:
            xrv_context = (
                "\n\nPreliminary AI screening results (TorchXRayVision classifier):"
                + xrv_context
                + "\nPlease confirm or refute these findings in your report."
            )

    return (
        "You are a board-certified radiologist. "
        "Interpret the following PA chest radiograph and produce a formal radiology report "
        "using exactly this structure:\n\n"
        "CLINICAL INDICATION: Chest radiograph for AI-assisted interpretation.\n\n"
        "TECHNIQUE: PA chest radiograph.\n\n"
        "FINDINGS:\n"
        "Lungs: [Describe right and left lung separately. "
        "Note any opacities, consolidation, masses, nodules, effusion, pneumothorax, "
        "or other abnormalities. Specify laterality and zone (upper/mid/lower).]\n"
        "Heart: [Size, contour of cardiac silhouette.]\n"
        "Mediastinum: [Width, contours, tracheal deviation, hilar prominence.]\n"
        "Pleura: [Effusion, pneumothorax, thickening.]\n"
        "Bones: [Visible ribs, thoracic spine, clavicles — fractures, lesions.]\n\n"
        "IMPRESSION:\n"
        "[Numbered list of primary diagnoses, most to least significant. "
        "Include specific follow-up recommendations where appropriate.]\n"
        + xrv_context
    )


def run_vlm(image: Image.Image, prompt: str) -> str:
    """Run VLM inference and return the raw text response."""
    import torch
    from qwen_vl_utils import process_vision_info
    backend = CONFIG["VLM_BACKEND"]
    model, proc = get_vlm()

    # ── CheXOne / Qwen2-VL ────────────────────────────────────
    if backend in ("chexone", "qwen2vl"):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text":  prompt},
            ],
        }]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = proc(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output = proc.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output[0].strip()

    # ── llama.cpp GGUF ────────────────────────────────────────
    elif backend == "llamacpp":
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        resp = model.create_chat_completion(
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
            max_tokens=1024,
            temperature=0.1,
        )
        return resp["choices"][0]["message"]["content"].strip()

    raise ValueError(f"Unknown backend: {backend}")


# ═══════════════════════════════════════════════════════════════
# Background inference job
# ═══════════════════════════════════════════════════════════════
def _run_job(job_id: str, contents: bytes, filename: str, mode: str, save_path: Path):
    jobs[job_id]["status"] = "running"
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        result = {
            "file_id":   job_id,
            "filename":  filename,
            "timestamp": datetime.now().isoformat(),
            "mode":      mode,
        }

        # ── Stage 1: XRV ──────────────────────────────────────
        if mode in ("fast", "full"):
            try:
                t0 = time.time()
                report = run_xrv(image)
                result["xrv"] = {
                    "inference_time_sec": round(time.time() - t0, 2),
                    "report": report,
                }
                logger.info(f"[{job_id}] XRV done in {time.time()-t0:.1f}s")
            except Exception as e:
                logger.error(f"[{job_id}] XRV failed: {e}")
                result["xrv"] = {"error": str(e)}

        # ── Stage 2: VLM ──────────────────────────────────────
        if mode == "full":
            try:
                t0 = time.time()
                xrv_report = result.get("xrv", {}).get("report")
                prompt   = _build_prompt(xrv_report)
                response = run_vlm(image, prompt)
                elapsed  = round(time.time() - t0, 2)
                result["vlm"] = {
                    "backend":            CONFIG["VLM_BACKEND"],
                    "inference_time_sec": elapsed,
                    "report":             response,
                }
                logger.info(f"[{job_id}] VLM done in {elapsed}s")
            except Exception as e:
                logger.error(f"[{job_id}] VLM failed: {e}")
                result["vlm"] = {"error": str(e)}

        jobs[job_id]["status"] = "done"
        jobs[job_id]["result"] = result

    except Exception as e:
        logger.error(f"[{job_id}] Job failed: {e}")
        jobs[job_id]["status"] = "error"
        jobs[job_id]["result"] = {"error": str(e)}


# ═══════════════════════════════════════════════════════════════
# API routes
# ═══════════════════════════════════════════════════════════════
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "vlm_backend": CONFIG["VLM_BACKEND"],
    })


@app.get("/health")
async def health():
    return {
        "status":      "ok",
        "timestamp":   datetime.now().isoformat(),
        "vlm_backend": CONFIG["VLM_BACKEND"],
    }


@app.post("/api/analyze")
async def analyze(
    file:  UploadFile = File(...),
    mode:  str        = Form(default="fast"),
    token: Optional[str] = Form(default=None),
):
    """Submit image for analysis. Returns job_id; poll /api/result/{job_id}."""
    verify_token(token)

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Please upload an image file (JPEG/PNG)")

    contents = await file.read()
    if len(contents) > CONFIG["MAX_FILE_SIZE_MB"] * 1024 * 1024:
        raise HTTPException(400, f"File too large (max {CONFIG['MAX_FILE_SIZE_MB']} MB)")

    job_id    = str(uuid.uuid4())[:8]
    save_path = CONFIG["UPLOAD_DIR"] / f"{job_id}_{file.filename}"
    save_path.write_bytes(contents)

    jobs[job_id] = {"status": "pending", "result": None}
    threading.Thread(
        target=_run_job,
        args=(job_id, contents, file.filename, mode, save_path),
        daemon=True,
    ).start()

    return JSONResponse({"job_id": job_id})


@app.get("/api/result/{job_id}")
async def result(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    j = jobs[job_id]
    return JSONResponse({"status": j["status"], "result": j["result"]})


@app.get("/api/history")
async def history():
    files = sorted(CONFIG["UPLOAD_DIR"].iterdir(), key=os.path.getmtime, reverse=True)
    return [{"filename": f.name, "size_kb": f.stat().st_size // 1024} for f in files[:20]]


# ═══════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    logger.info(f"CXR Server starting on {CONFIG['HOST']}:{CONFIG['PORT']}")
    logger.info(f"VLM backend: {CONFIG['VLM_BACKEND']}")
    uvicorn.run(app, host=CONFIG["HOST"], port=CONFIG["PORT"])
