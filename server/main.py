#!/usr/bin/env python3
"""
AgriSense - Serveur de réception station laptop
Gestion du mode maintenance, réception des données de terrain, et
évaluation automatique du dataset de test (294 images).

Nouveau dans cette version : l'endpoint /test/submit/{node_id}/{image_id}
reçoit l'image originale + les artefacts compressés WZ-OSEG et ADRES
(profils Q et E), reconstruit chaque variante, calcule PSNR/SSIM/IoU/Dice
et ajoute une ligne par variante à data/results.csv.

Hypothèses explicites (à vérifier / discuter, pas des faits établis) :
  - WZ-OSEG est décodé en utilisant l'image originale elle-même comme
    "side information". C'est un choix assumé et documenté comme un biais
    dans les limites de l'article (les 294 images ne sont pas des rafales
    temporellement corrélées) -- le PSNR/SSIM WZ-OSEG mesuré ici est donc
    une BORNE SUPÉRIEURE, pas une mesure de déploiement réaliste.
  - Le "masque de référence" pour IoU/Dice est un Otsu recalculé côté
    serveur sur l'image brute uploadée -- PAS un masque annoté à la main.
    Ce chiffre mesure donc la FIDÉLITÉ DE TRANSMISSION du masque
    (subsampling/quantification), pas une exactitude de segmentation par
    rapport à une vérité terrain humaine.
"""

import csv
import json
import shutil
import sys
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel
from skimage.filters import threshold_otsu

# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------
BASE_DIR      = Path(__file__).parent
DATA_DIR      = BASE_DIR / "data"
MODE_FILE     = DATA_DIR / "mode.json"
NODES_DIR     = DATA_DIR / "nodes"
TEST_RUN_DIR  = DATA_DIR / "test_run"
RESULTS_CSV   = DATA_DIR / "results.csv"
DECODERS_DIR  = BASE_DIR / "decoders"

DATA_DIR.mkdir(exist_ok=True)
TEST_RUN_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(DECODERS_DIR))
import metrics as qmetrics          # noqa: E402  (metrics.py, same folder as main.py)
import wz_oseg_decode                # noqa: E402  (decoders/wz_oseg_decode.py)
import adres_decode                  # noqa: E402  (decoders/adres_decode.py)
import reference_jpeg_decode         # noqa: E402  (decoders/reference_jpeg_decode.py, branch reference/jpeg)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="AgriSense Server", version="1.1.0")

_csv_lock = threading.Lock()
_csv_fields = [
    "timestamp", "node_id", "image_id", "algorithm", "profile",
    "psnr_db", "ssim", "mask_iou", "mask_dice",
    "cpu_time_ms", "memory_kb", "compressed_bytes", "compression_ratio",
]


# ---------------------------------------------------------------------------
# Utilitaires existants (mode terrain, inchangés)
# ---------------------------------------------------------------------------
def get_node_dir(node_id: str) -> Path:
    node_dir = NODES_DIR / node_id
    if not node_dir.exists():
        raise HTTPException(status_code=404, detail=f"Noeud {node_id} inconnu")
    return node_dir


def read_mode() -> str:
    if not MODE_FILE.exists():
        write_mode("NORMAL")
    with open(MODE_FILE, "r") as f:
        return json.load(f)["mode"]


def write_mode(mode: str) -> None:
    with open(MODE_FILE, "w") as f:
        json.dump({"mode": mode}, f)


class ModeUpdate(BaseModel):
    mode: str  # "NORMAL" ou "MAINTENANCE"


@app.get("/node/{node_id}/mode")
def get_mode(node_id: str):
    get_node_dir(node_id)
    mode = read_mode()
    return {"node_id": node_id, "mode": mode}


@app.put("/node/mode")
def set_mode(update: ModeUpdate):
    mode = update.mode.upper()
    if mode not in ("NORMAL", "MAINTENANCE"):
        raise HTTPException(status_code=400, detail="Mode invalide. Valeurs acceptées : NORMAL, MAINTENANCE")
    write_mode(mode)
    return {"status": "ok", "mode": mode}


@app.post("/node/{node_id}/upload/image")
async def upload_image(node_id: str, file: UploadFile = File(...)):
    node_dir = get_node_dir(node_id)
    images_dir = node_dir / "images"
    images_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = Path(file.filename).suffix if file.filename else ".jpg"
    dest = images_dir / f"{timestamp}{suffix}"

    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return {"status": "ok", "saved_as": str(dest.relative_to(BASE_DIR))}


@app.post("/node/{node_id}/upload/metrics")
async def upload_metrics(node_id: str, file: UploadFile = File(...)):
    node_dir = get_node_dir(node_id)
    metrics_dir = node_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = metrics_dir / f"{timestamp}.json"

    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return {"status": "ok", "saved_as": str(dest.relative_to(BASE_DIR))}


@app.get("/status")
def status():
    mode = read_mode()
    nodes = [d.name for d in NODES_DIR.iterdir() if d.is_dir()] if NODES_DIR.exists() else []
    return {"server": "AgriSense", "mode": mode, "nodes": nodes}


# ---------------------------------------------------------------------------
# NEW: évaluation automatique du dataset de test
# ---------------------------------------------------------------------------
def parse_metrics_txt(path: Path) -> dict:
    """node_metrics.txt is a simple 'key,value' CSV written by the C encoders."""
    values = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or "," not in line:
                continue
            key, val = line.split(",", 1)
            try:
                values[key] = float(val)
            except ValueError:
                values[key] = val
    return values


def reference_otsu_mask(gray_array: np.ndarray) -> np.ndarray:
    """Same 'gray > threshold' convention as the C encoders' otsu_threshold()."""
    thresh = threshold_otsu(gray_array)
    return (gray_array > thresh).astype(np.uint8) * 255


def append_result_row(row: dict) -> None:
    with _csv_lock:
        file_exists = RESULTS_CSV.exists()
        with open(RESULTS_CSV, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_csv_fields)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)


async def save_upload(upload: UploadFile, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        shutil.copyfileobj(upload.file, f)


@app.post("/test/submit/wz-oseg/{node_id}/{image_id}")
async def test_submit_wzoseg(
    node_id: str,
    image_id: str,
    original: UploadFile = File(...),
    wz_compressed: UploadFile = File(...),
    wz_otsu_mask: UploadFile = File(...),
    wz_metrics: UploadFile = File(...),
):
    """Independent endpoint for the algo/wz-oseg branch's pipeline_test.py.
    No dependency on ADRES artifacts -- this branch's test run is self-contained."""
    run_dir = TEST_RUN_DIR / "wz-oseg" / node_id / image_id
    run_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "original": run_dir / "original.ppm",
        "wz_compressed": run_dir / "wz_compressed.p1",
        "wz_otsu_mask": run_dir / "wz_otsu_mask.ppm",
        "wz_metrics": run_dir / "wz_metrics.txt",
    }
    for key, upload in {
        "original": original, "wz_compressed": wz_compressed,
        "wz_otsu_mask": wz_otsu_mask, "wz_metrics": wz_metrics,
    }.items():
        await save_upload(upload, paths[key])

    try:
        original_img = Image.open(paths["original"]).convert("RGB")
        original_rgb = np.array(original_img)
        ref_mask = reference_otsu_mask(np.array(original_img.convert("L")))
        timestamp = datetime.now().isoformat()

        # NOTE: side info = original image itself (documented bias, see module docstring).
        try:
            otsu_data, hue_data, dct_coeffs, params = wz_oseg_decode.read_compressed(paths["wz_compressed"])
            width, height, bw, bh, coeff_count = params
            side_gray = (0.299 * original_rgb[:, :, 0] + 0.587 * original_rgb[:, :, 1] +
                         0.114 * original_rgb[:, :, 2]).astype(np.uint8)
            y_rec = wz_oseg_decode.reconstruct_from_dct_sparse(dct_coeffs, width, height, bw, bh, side_gray)
            wz_rgb = wz_oseg_decode.gray_to_rgb(y_rec)
            Image.fromarray(wz_rgb, "RGB").save(run_dir / "reconstructed_wz.png")

            wz_metrics_vals = parse_metrics_txt(paths["wz_metrics"])
            wz_mask = np.array(Image.open(paths["wz_otsu_mask"]).convert("L"))
            iou, dice = qmetrics.compute_iou_dice(ref_mask, wz_mask)

            row = {
                "timestamp": timestamp, "node_id": node_id, "image_id": image_id,
                "algorithm": "WZ-OSEG", "profile": "-",
                "psnr_db": round(qmetrics.compute_psnr(original_rgb, wz_rgb), 3),
                "ssim": round(qmetrics.compute_ssim(original_rgb, wz_rgb), 4),
                "mask_iou": round(iou, 4), "mask_dice": round(dice, 4),
                "cpu_time_ms": wz_metrics_vals.get("cpu_time_ms"),
                "memory_kb": wz_metrics_vals.get("memory_kb"),
                "compressed_bytes": wz_metrics_vals.get("compressed_bytes"),
                "compression_ratio": wz_metrics_vals.get("compression_ratio"),
            }
        except Exception as exc:
            row = {
                "timestamp": timestamp, "node_id": node_id, "image_id": image_id,
                "algorithm": "WZ-OSEG", "profile": "ERROR",
                "psnr_db": None, "ssim": None, "mask_iou": None, "mask_dice": None,
                "cpu_time_ms": None, "memory_kb": None, "compressed_bytes": None,
                "compression_ratio": None,
            }
            print(f"[WZ-OSEG] Erreur reconstruction {image_id}: {exc}")

        append_result_row(row)
        return JSONResponse({"status": "ok", "image_id": image_id, "rows_appended": 1})

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erreur traitement {image_id}: {exc}")


@app.post("/test/submit/adres/{node_id}/{image_id}")
async def test_submit_adres(
    node_id: str,
    image_id: str,
    original: UploadFile = File(...),
    adres_q_compressed: UploadFile = File(...),
    adres_q_roi_mask: UploadFile = File(...),
    adres_q_metrics: UploadFile = File(...),
    adres_e_compressed: UploadFile = File(...),
    adres_e_roi_mask: UploadFile = File(...),
    adres_e_metrics: UploadFile = File(...),
):
    """Independent endpoint for the algo/adres branch's pipeline_test.py.
    No dependency on WZ-OSEG artifacts -- this branch's test run is self-contained."""
    run_dir = TEST_RUN_DIR / "adres" / node_id / image_id
    run_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "original": run_dir / "original.ppm",
        "adres_q_compressed": run_dir / "adres_q_compressed.p2",
        "adres_q_roi_mask": run_dir / "adres_q_roi_mask.ppm",
        "adres_q_metrics": run_dir / "adres_q_metrics.txt",
        "adres_e_compressed": run_dir / "adres_e_compressed.p2",
        "adres_e_roi_mask": run_dir / "adres_e_roi_mask.ppm",
        "adres_e_metrics": run_dir / "adres_e_metrics.txt",
    }
    for key, upload in {
        "original": original,
        "adres_q_compressed": adres_q_compressed, "adres_q_roi_mask": adres_q_roi_mask,
        "adres_q_metrics": adres_q_metrics, "adres_e_compressed": adres_e_compressed,
        "adres_e_roi_mask": adres_e_roi_mask, "adres_e_metrics": adres_e_metrics,
    }.items():
        await save_upload(upload, paths[key])

    try:
        original_img = Image.open(paths["original"]).convert("RGB")
        original_rgb = np.array(original_img)
        ref_mask = reference_otsu_mask(np.array(original_img.convert("L")))
        timestamp = datetime.now().isoformat()

        results = []
        for profile, comp_key, mask_key, metrics_key in [
            ("Q", "adres_q_compressed", "adres_q_roi_mask", "adres_q_metrics"),
            ("E", "adres_e_compressed", "adres_e_roi_mask", "adres_e_metrics"),
        ]:
            try:
                roi_mask, roi_png_bytes, bg_png_bytes, params = adres_decode.read_compressed(paths[comp_key])
                adres_rgb = adres_decode.reconstruct_image(roi_mask, roi_png_bytes, bg_png_bytes, params)
                Image.fromarray(adres_rgb, "RGB").save(run_dir / f"reconstructed_adres_{profile}.png")

                adres_metrics_vals = parse_metrics_txt(paths[metrics_key])
                adres_mask = np.array(Image.open(paths[mask_key]).convert("L"))
                iou, dice = qmetrics.compute_iou_dice(ref_mask, adres_mask)

                results.append({
                    "timestamp": timestamp, "node_id": node_id, "image_id": image_id,
                    "algorithm": "ADRES", "profile": profile,
                    "psnr_db": round(qmetrics.compute_psnr(original_rgb, adres_rgb), 3),
                    "ssim": round(qmetrics.compute_ssim(original_rgb, adres_rgb), 4),
                    "mask_iou": round(iou, 4), "mask_dice": round(dice, 4),
                    "cpu_time_ms": adres_metrics_vals.get("cpu_time_ms"),
                    "memory_kb": adres_metrics_vals.get("memory_kb"),
                    "compressed_bytes": adres_metrics_vals.get("compressed_bytes"),
                    "compression_ratio": adres_metrics_vals.get("compression_ratio"),
                })
            except Exception as exc:
                results.append({
                    "timestamp": timestamp, "node_id": node_id, "image_id": image_id,
                    "algorithm": "ADRES", "profile": f"{profile}-ERROR",
                    "psnr_db": None, "ssim": None, "mask_iou": None, "mask_dice": None,
                    "cpu_time_ms": None, "memory_kb": None, "compressed_bytes": None,
                    "compression_ratio": None,
                })
                print(f"[ADRES-{profile}] Erreur reconstruction {image_id}: {exc}")

        for row in results:
            append_result_row(row)

        return JSONResponse({"status": "ok", "image_id": image_id, "rows_appended": len(results)})

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erreur traitement {image_id}: {exc}")


@app.post("/test/submit/reference/{node_id}/{image_id}")
async def test_submit_reference(
    node_id: str,
    image_id: str,
    original: UploadFile = File(...),
    jpeg_compressed: UploadFile = File(...),
    jpeg_metrics: UploadFile = File(...),
    jp2roi_compressed: UploadFile = File(...),
    jp2roi_roi_mask: UploadFile = File(...),
    jp2roi_metrics: UploadFile = File(...),
):
    """Independent endpoint for the branch reference/jpeg's pipeline_test.py.
    Two methods, both decoded here:
      - "JPEG": plain Pillow-encoded JPEG, decoded directly by Pillow.
      - "JPEG2000-ROI-2stream": OpenJPEG dual-stream, region-differentiated
        encoding (NOT the codestream-native Annex H ROI feature -- see
        reference_jpeg_decode.py / jpeg2000_roi_test.py docstrings for why).
    Naming is deliberately explicit ("2stream") so results.csv never implies
    a native-ROI JPEG2000 measurement that wasn't actually performed."""
    run_dir = TEST_RUN_DIR / "reference" / node_id / image_id
    run_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "original": run_dir / "original.ppm",
        "jpeg_compressed": run_dir / "jpeg_compressed.jpg",
        "jpeg_metrics": run_dir / "jpeg_metrics.txt",
        "jp2roi_compressed": run_dir / "jp2roi_compressed.jp2roi",
        "jp2roi_roi_mask": run_dir / "jp2roi_roi_mask.ppm",
        "jp2roi_metrics": run_dir / "jp2roi_metrics.txt",
    }
    for key, upload in {
        "original": original,
        "jpeg_compressed": jpeg_compressed, "jpeg_metrics": jpeg_metrics,
        "jp2roi_compressed": jp2roi_compressed, "jp2roi_roi_mask": jp2roi_roi_mask,
        "jp2roi_metrics": jp2roi_metrics,
    }.items():
        await save_upload(upload, paths[key])

    try:
        original_img = Image.open(paths["original"]).convert("RGB")
        original_rgb = np.array(original_img)
        ref_mask = reference_otsu_mask(np.array(original_img.convert("L")))
        timestamp = datetime.now().isoformat()

        results = []

        # --- JPEG (plain, Pillow) ---------------------------------------
        try:
            jpeg_rgb = reference_jpeg_decode.decode_jpeg(paths["jpeg_compressed"])
            jpeg_metrics_vals = parse_metrics_txt(paths["jpeg_metrics"])
            Image.fromarray(jpeg_rgb, "RGB").save(run_dir / "reconstructed_jpeg.png")

            results.append({
                "timestamp": timestamp, "node_id": node_id, "image_id": image_id,
                "algorithm": "JPEG", "profile": "-",
                "psnr_db": round(qmetrics.compute_psnr(original_rgb, jpeg_rgb), 3),
                "ssim": round(qmetrics.compute_ssim(original_rgb, jpeg_rgb), 4),
                "mask_iou": None, "mask_dice": None,  # JPEG has no ROI mask
                "cpu_time_ms": jpeg_metrics_vals.get("cpu_time_ms"),
                "memory_kb": jpeg_metrics_vals.get("memory_kb"),
                "compressed_bytes": jpeg_metrics_vals.get("compressed_bytes"),
                "compression_ratio": jpeg_metrics_vals.get("compression_ratio"),
            })
        except Exception as exc:
            results.append({
                "timestamp": timestamp, "node_id": node_id, "image_id": image_id,
                "algorithm": "JPEG", "profile": "ERROR",
                "psnr_db": None, "ssim": None, "mask_iou": None, "mask_dice": None,
                "cpu_time_ms": None, "memory_kb": None, "compressed_bytes": None,
                "compression_ratio": None,
            })
            print(f"[JPEG] Erreur reconstruction {image_id}: {exc}")

        # --- JPEG2000-ROI-2stream ----------------------------------------
        try:
            jp2roi_rgb, jp2roi_pixel_mask = reference_jpeg_decode.reconstruct_jpeg2000_roi(
                paths["jp2roi_compressed"]
            )
            Image.fromarray(jp2roi_rgb, "RGB").save(run_dir / "reconstructed_jp2roi.png")

            jp2roi_metrics_vals = parse_metrics_txt(paths["jp2roi_metrics"])
            iou, dice = qmetrics.compute_iou_dice(ref_mask, jp2roi_pixel_mask)

            results.append({
                "timestamp": timestamp, "node_id": node_id, "image_id": image_id,
                "algorithm": "JPEG2000-ROI-2stream", "profile": "-",
                "psnr_db": round(qmetrics.compute_psnr(original_rgb, jp2roi_rgb), 3),
                "ssim": round(qmetrics.compute_ssim(original_rgb, jp2roi_rgb), 4),
                "mask_iou": round(iou, 4), "mask_dice": round(dice, 4),
                "cpu_time_ms": jp2roi_metrics_vals.get("cpu_time_ms"),
                "memory_kb": jp2roi_metrics_vals.get("memory_kb"),
                "compressed_bytes": jp2roi_metrics_vals.get("compressed_bytes"),
                "compression_ratio": jp2roi_metrics_vals.get("compression_ratio"),
            })
        except Exception as exc:
            results.append({
                "timestamp": timestamp, "node_id": node_id, "image_id": image_id,
                "algorithm": "JPEG2000-ROI-2stream", "profile": "ERROR",
                "psnr_db": None, "ssim": None, "mask_iou": None, "mask_dice": None,
                "cpu_time_ms": None, "memory_kb": None, "compressed_bytes": None,
                "compression_ratio": None,
            })
            print(f"[JPEG2000-ROI-2stream] Erreur reconstruction {image_id}: {exc}")

        for row in results:
            append_result_row(row)

        return JSONResponse({"status": "ok", "image_id": image_id, "rows_appended": len(results)})

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erreur traitement {image_id}: {exc}")


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
