#!/usr/bin/env python3
"""
AgriSense - Serveur de réception station laptop  v1.3
======================================================
Nouveautés v1.3 — Samples visuels pour publication :
  - Sauvegarde de reconstructions côté serveur pour un sous-ensemble
    fixe d'images (samples), utilisables directement dans les articles.
  - Deux formats produits par image sample :
      1. PNG individuels  →  data/samples/<image_id>/
             original.png | wz_oseg.png | adres_q.png | adres_e.png
             sam_mask.png (ground truth) | otsu_mask.png (encodeur)
      2. Planche de comparaison  →  data/samples/<image_id>/comparison.png
             figure 7 colonnes côte à côte, prête à insérer dans LaTeX
             (long paper) ou rognée à 4 colonnes (short paper).
  - Sélection du sous-ensemble fixe :
      • Priorité 1 : data/sample_ids.txt  (un image_id par ligne,
        éditable manuellement avant le run).
      • Priorité 2 : sélection automatique de SAMPLE_AUTO_N images
        réparties uniformément sur l'ordre alphabétique des image_id
        vus au premier run — liste écrite dans data/sample_ids.txt
        pour reproductibilité.
  - Endpoint GET /samples/download  →  ZIP de tout data/samples/.
  - Endpoint GET /samples/list       →  liste JSON des samples produits.

Hypothèses inchangées de v1.2 :
  - WZ-OSEG décodé avec l'image originale comme side information
    (borne supérieure PSNR/SSIM, biais documenté).
  - Priorité masque de référence : sam_uploaded > sam_local > otsu_fallback.
  - Énergie NON rapportée (INA219 saturait à 3.2 A au boot du Pi).
"""

import csv
import io
import json
import shutil
import sys
import threading
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel
from skimage.filters import threshold_otsu

# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------
BASE_DIR         = Path(__file__).parent
DATA_DIR         = BASE_DIR / "data"
MODE_FILE        = DATA_DIR / "mode.json"
NODES_DIR        = DATA_DIR / "nodes"
TEST_RUN_DIR     = DATA_DIR / "test_run"
RESULTS_CSV      = DATA_DIR / "results.csv"
DECODERS_DIR     = BASE_DIR / "decoders"
SAMPLES_DIR      = DATA_DIR / "samples"
SAMPLE_IDS_FILE  = DATA_DIR / "sample_ids.txt"

SERVER_DATASET_DIR = Path.home() / "DATASET"
SERVER_MASK_DIR    = SERVER_DATASET_DIR / "mask"

# Nombre d'images choisies automatiquement si sample_ids.txt est absent
SAMPLE_AUTO_N = 15

DATA_DIR.mkdir(exist_ok=True)
TEST_RUN_DIR.mkdir(exist_ok=True)
SAMPLES_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(DECODERS_DIR))
import metrics as qmetrics      # noqa: E402
import wz_oseg_decode            # noqa: E402
import adres_decode              # noqa: E402

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="AgriSense Server", version="1.3.0")

_csv_lock    = threading.Lock()
_sample_lock = threading.Lock()

_csv_fields = [
    "timestamp", "node_id", "image_id", "algorithm", "profile",
    "psnr_db", "ssim", "mask_iou", "mask_dice",
    "mask_ref_source",
    "cpu_time_ms", "memory_kb", "compressed_bytes", "compression_ratio",
]

# Cache en mémoire des image_id retenus comme samples
# (chargé au premier accès, mis à jour si sélection auto)
_sample_ids: set | None = None
_seen_ids: list = []        # ordre d'apparition pour la sélection auto


# ---------------------------------------------------------------------------
# Gestion du sous-ensemble sample
# ---------------------------------------------------------------------------

def _load_sample_ids() -> set:
    """
    Charge sample_ids.txt si présent.
    Retourne un ensemble vide si le fichier n'existe pas encore
    (la sélection auto se fera plus tard dans _register_seen_id).
    """
    if SAMPLE_IDS_FILE.exists():
        ids = set()
        with open(SAMPLE_IDS_FILE) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    ids.add(line)
        return ids
    return set()


def _save_sample_ids(ids: set) -> None:
    with open(SAMPLE_IDS_FILE, "w") as f:
        f.write("# Sous-ensemble fixe pour figures de publication\n")
        f.write("# Généré automatiquement — éditable manuellement\n")
        for id_ in sorted(ids):
            f.write(f"{id_}\n")


def _register_seen_id(image_id: str) -> None:
    """
    Enregistre image_id dans _seen_ids.
    Si sample_ids.txt n'existe pas et qu'on a vu >= SAMPLE_AUTO_N images,
    déclenche la sélection automatique uniforme et écrit le fichier.
    """
    global _sample_ids, _seen_ids
    with _sample_lock:
        if image_id not in _seen_ids:
            _seen_ids.append(image_id)

        # Sélection auto uniquement si le fichier n'existait pas au démarrage
        if _sample_ids is not None and len(_sample_ids) > 0:
            return  # liste manuelle déjà chargée

        # Pas encore de liste : on attend d'avoir vu assez d'images
        # (ou on prend ce qu'on a si on dépasse SAMPLE_AUTO_N)
        if len(_seen_ids) >= SAMPLE_AUTO_N and not SAMPLE_IDS_FILE.exists():
            step = len(_seen_ids) / SAMPLE_AUTO_N
            selected = {
                _seen_ids[round(i * step)] for i in range(SAMPLE_AUTO_N)
            }
            _sample_ids = selected
            _save_sample_ids(selected)
            print(f"[Samples] Sélection automatique de {len(selected)} images "
                  f"→ {SAMPLE_IDS_FILE}")


def is_sample(image_id: str) -> bool:
    """Retourne True si image_id fait partie du sous-ensemble sample."""
    global _sample_ids
    if _sample_ids is None:
        _sample_ids = _load_sample_ids()
    return image_id in _sample_ids


# ---------------------------------------------------------------------------
# Construction de la planche de comparaison
# ---------------------------------------------------------------------------

# Colonnes de la planche (ordre = ordre d'affichage)
_PANEL_COLS = [
    ("original",  "Original"),
    ("sam_mask",  "SAM mask"),
    ("otsu_mask", "Otsu mask"),
    ("wz_oseg",   "WZ-OSEG"),
    ("adres_q",   "ADRES-Q"),
    ("adres_e",   "ADRES-E"),
]

_LABEL_H   = 22    # pixels réservés pour le label texte sous chaque vignette
_THUMB_W   = 160   # largeur d'une vignette (hauteur calculée proportionnellement)
_PAD       = 4     # espacement entre vignettes


def _build_comparison(images: dict[str, np.ndarray | None],
                      thumb_w: int = _THUMB_W) -> Image.Image:
    """
    Construit une planche de comparaison horizontale.
    images : dict col_key → ndarray RGB uint8 ou None (colonne absente).
    Retourne une image PIL.
    """
    # Colonnes présentes uniquement
    cols = [(k, lbl) for k, lbl in _PANEL_COLS if images.get(k) is not None]
    if not cols:
        raise ValueError("Aucune image à assembler")

    # Dimensions uniformes : toutes les vignettes ont la même largeur
    ref = images[cols[0][0]]
    h0, w0 = ref.shape[:2]
    thumb_h = round(thumb_w * h0 / w0)

    panel_w = len(cols) * (thumb_w + _PAD) - _PAD
    panel_h = thumb_h + _LABEL_H
    panel = Image.new("RGB", (panel_w, panel_h), (245, 245, 245))

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except Exception:
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(panel)

    for col_idx, (key, label) in enumerate(cols):
        arr = images[key]
        # Conversion niveaux de gris → RGB si nécessaire
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        thumb = Image.fromarray(arr.astype(np.uint8), "RGB").resize(
            (thumb_w, thumb_h), Image.LANCZOS)

        x0 = col_idx * (thumb_w + _PAD)
        panel.paste(thumb, (x0, 0))

        # Label centré sous la vignette
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        tx = x0 + (thumb_w - text_w) // 2
        ty = thumb_h + 2
        draw.text((tx, ty), label, fill=(30, 30, 30), font=font)

    return panel


def save_sample(
    image_id: str,
    original_rgb: np.ndarray,
    wz_rgb: np.ndarray | None,
    adres_q_rgb: np.ndarray | None,
    adres_e_rgb: np.ndarray | None,
    sam_mask: np.ndarray | None,
    otsu_mask: np.ndarray | None,
) -> None:
    """
    Sauvegarde les PNG individuels et la planche de comparaison dans
    data/samples/<image_id>/.
    Idempotent : écrase silencieusement si déjà présent.
    """
    out_dir = SAMPLES_DIR / image_id
    out_dir.mkdir(parents=True, exist_ok=True)

    def _save(arr: np.ndarray | None, name: str) -> None:
        if arr is None:
            return
        if arr.ndim == 2:
            img = Image.fromarray(arr.astype(np.uint8), "L")
        else:
            img = Image.fromarray(arr.astype(np.uint8), "RGB")
        img.save(out_dir / name, "PNG")

    _save(original_rgb,  "original.png")
    _save(wz_rgb,        "wz_oseg.png")
    _save(adres_q_rgb,   "adres_q.png")
    _save(adres_e_rgb,   "adres_e.png")
    _save(sam_mask,      "sam_mask.png")
    _save(otsu_mask,     "otsu_mask.png")

    # Planche de comparaison
    images_dict = {
        "original":  original_rgb,
        "sam_mask":  sam_mask,
        "otsu_mask": otsu_mask,
        "wz_oseg":   wz_rgb,
        "adres_q":   adres_q_rgb,
        "adres_e":   adres_e_rgb,
    }
    try:
        panel = _build_comparison(images_dict)
        panel.save(out_dir / "comparison.png", "PNG")
        print(f"[Samples] Planche sauvegardée : {out_dir / 'comparison.png'}")
    except Exception as exc:
        print(f"[Samples] Erreur planche {image_id}: {exc}")


# ---------------------------------------------------------------------------
# Utilitaires mode terrain (inchangés)
# ---------------------------------------------------------------------------

def get_node_dir(node_id: str) -> Path:
    node_dir = NODES_DIR / node_id
    if not node_dir.exists():
        raise HTTPException(status_code=404, detail=f"Noeud {node_id} inconnu")
    return node_dir


def read_mode() -> str:
    if not MODE_FILE.exists():
        write_mode("NORMAL")
    with open(MODE_FILE) as f:
        return json.load(f)["mode"]


def write_mode(mode: str) -> None:
    with open(MODE_FILE, "w") as f:
        json.dump({"mode": mode}, f)


class ModeUpdate(BaseModel):
    mode: str


@app.get("/node/{node_id}/mode")
def get_mode(node_id: str):
    get_node_dir(node_id)
    return {"node_id": node_id, "mode": read_mode()}


@app.put("/node/mode")
def set_mode(update: ModeUpdate):
    mode = update.mode.upper()
    if mode not in ("NORMAL", "MAINTENANCE"):
        raise HTTPException(status_code=400, detail="Mode invalide.")
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


@app.get("/test/done_ids")
def get_done_ids():
    done = set()
    if RESULTS_CSV.exists():
        with open(RESULTS_CSV, newline="") as f:
            for row in csv.DictReader(f):
                done.add(row["image_id"])
    return {"done_ids": list(done)}


# ---------------------------------------------------------------------------
# Endpoints samples
# ---------------------------------------------------------------------------

@app.get("/samples/list")
def samples_list():
    """Liste les image_id pour lesquels un dossier sample existe."""
    if not SAMPLES_DIR.exists():
        return {"samples": []}
    entries = []
    for d in sorted(SAMPLES_DIR.iterdir()):
        if d.is_dir():
            files = [p.name for p in sorted(d.iterdir()) if p.is_file()]
            entries.append({"image_id": d.name, "files": files})
    return {"count": len(entries), "samples": entries}


@app.get("/samples/download")
def samples_download():
    """
    Retourne un ZIP de data/samples/ en streaming.
    Pratique pour récupérer toutes les planches d'un coup :
        curl http://server:8000/samples/download -o samples.zip
    """
    if not SAMPLES_DIR.exists() or not any(SAMPLES_DIR.iterdir()):
        raise HTTPException(status_code=404, detail="Aucun sample disponible.")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(SAMPLES_DIR.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(SAMPLES_DIR.parent))
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=agrisense_samples.zip"},
    )


@app.get("/samples/ids")
def samples_ids():
    """Retourne le contenu de sample_ids.txt (liste manuelle ou auto-générée)."""
    ids = _load_sample_ids()
    return {
        "source": "manual" if SAMPLE_IDS_FILE.exists() else "not_generated_yet",
        "count": len(ids),
        "sample_ids": sorted(ids),
    }


# ---------------------------------------------------------------------------
# Utilitaires évaluation
# ---------------------------------------------------------------------------

def parse_metrics_txt(path: Path) -> dict:
    values = {}
    with open(path) as f:
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


def find_server_sam_mask(image_id: str) -> Path | None:
    if not SERVER_MASK_DIR.is_dir():
        return None
    for p in [SERVER_MASK_DIR / f"{image_id}_mask.png",
              SERVER_MASK_DIR / f"{image_id}.png"]:
        if p.exists():
            return p
    return None


def load_reference_mask(
    image_id: str,
    uploaded_sam_path: Path | None,
    original_gray: np.ndarray,
) -> tuple[np.ndarray, str]:
    """
    Priorité : sam_uploaded > sam_local > otsu_fallback.
    Retourne (masque uint8 0/255, source_string).
    """
    if uploaded_sam_path is not None and uploaded_sam_path.exists():
        m = np.array(Image.open(uploaded_sam_path).convert("L"))
        return (m > 127).astype(np.uint8) * 255, "sam_uploaded"

    local = find_server_sam_mask(image_id)
    if local is not None:
        m = np.array(Image.open(local).convert("L"))
        return (m > 127).astype(np.uint8) * 255, "sam_local"

    thresh = threshold_otsu(original_gray)
    print(f"  [Avertissement] {image_id} : aucun masque SAM, fallback Otsu.")
    return (original_gray > thresh).astype(np.uint8) * 255, "otsu_fallback"


def append_result_row(row: dict) -> None:
    with _csv_lock:
        exists = RESULTS_CSV.exists()
        with open(RESULTS_CSV, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_csv_fields)
            if not exists:
                writer.writeheader()
            writer.writerow(row)


async def save_upload(upload: UploadFile, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        shutil.copyfileobj(upload.file, f)


# ---------------------------------------------------------------------------
# Endpoint WZ-OSEG
# ---------------------------------------------------------------------------

@app.post("/test/submit/wz-oseg/{node_id}/{image_id}")
async def test_submit_wzoseg(
    node_id: str,
    image_id: str,
    original:      UploadFile = File(...),
    wz_compressed: UploadFile = File(...),
    wz_otsu_mask:  UploadFile = File(...),
    wz_metrics:    UploadFile = File(...),
    sam_mask:      UploadFile = File(None),
):
    run_dir = TEST_RUN_DIR / "wz-oseg" / node_id / image_id
    run_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "original":      run_dir / "original.png",
        "wz_compressed": run_dir / "wz_compressed.p1",
        "wz_otsu_mask":  run_dir / "wz_otsu_mask.png",
        "wz_metrics":    run_dir / "wz_metrics.txt",
        "sam_mask":      run_dir / "sam_mask.png",
    }
    for key, upload in [
        ("original", original), ("wz_compressed", wz_compressed),
        ("wz_otsu_mask", wz_otsu_mask), ("wz_metrics", wz_metrics),
    ]:
        await save_upload(upload, paths[key])

    uploaded_sam_path = None
    if sam_mask is not None:
        await save_upload(sam_mask, paths["sam_mask"])
        uploaded_sam_path = paths["sam_mask"]

    # Enregistrement pour sélection auto si nécessaire
    _register_seen_id(image_id)

    try:
        original_img  = Image.open(paths["original"]).convert("RGB")
        original_rgb  = np.array(original_img)
        original_gray = np.array(original_img.convert("L"))
        timestamp     = datetime.now().isoformat()

        ref_mask, mask_ref_source = load_reference_mask(
            image_id, uploaded_sam_path, original_gray)

        wz_rgb = None
        otsu_mask_arr = None
        try:
            otsu_data, hue_data, dct_coeffs, params = wz_oseg_decode.read_compressed(
                paths["wz_compressed"])
            width, height, bw, bh, coeff_count = params
            side_gray = (0.299 * original_rgb[:, :, 0] +
                         0.587 * original_rgb[:, :, 1] +
                         0.114 * original_rgb[:, :, 2]).astype(np.uint8)
            y_rec  = wz_oseg_decode.reconstruct_from_dct_sparse(
                dct_coeffs, width, height, bw, bh, side_gray)
            wz_rgb = wz_oseg_decode.gray_to_rgb(y_rec)

            otsu_mask_arr = np.array(
                Image.open(paths["wz_otsu_mask"]).convert("L"))

            wz_metrics_vals = parse_metrics_txt(paths["wz_metrics"])
            iou, dice = qmetrics.compute_iou_dice(ref_mask, otsu_mask_arr)

            row = {
                "timestamp": timestamp, "node_id": node_id, "image_id": image_id,
                "algorithm": "WZ-OSEG", "profile": "-",
                "psnr_db":  round(qmetrics.compute_psnr(original_rgb, wz_rgb), 3),
                "ssim":     round(qmetrics.compute_ssim(original_rgb, wz_rgb), 4),
                "mask_iou": round(iou, 4), "mask_dice": round(dice, 4),
                "mask_ref_source": mask_ref_source,
                "cpu_time_ms":       wz_metrics_vals.get("cpu_time_ms"),
                "memory_kb":         wz_metrics_vals.get("memory_kb"),
                "compressed_bytes":  wz_metrics_vals.get("compressed_bytes"),
                "compression_ratio": wz_metrics_vals.get("compression_ratio"),
            }
        except Exception as exc:
            row = {
                "timestamp": timestamp, "node_id": node_id, "image_id": image_id,
                "algorithm": "WZ-OSEG", "profile": "ERROR",
                "psnr_db": None, "ssim": None,
                "mask_iou": None, "mask_dice": None,
                "mask_ref_source": mask_ref_source,
                "cpu_time_ms": None, "memory_kb": None,
                "compressed_bytes": None, "compression_ratio": None,
            }
            print(f"[WZ-OSEG] Erreur reconstruction {image_id}: {exc}")

        append_result_row(row)

        # --- Sample visuel (WZ-OSEG seul ne produit pas ADRES : on attend
        #     l'endpoint ADRES pour la planche complète).
        #     On sauvegarde ici uniquement original + wz_oseg + masques,
        #     la planche sera complétée/écrasée par l'endpoint ADRES si
        #     les deux branches tournent sur le même image_id. ---
        if is_sample(image_id) and wz_rgb is not None:
            sam_arr = None
            if uploaded_sam_path and uploaded_sam_path.exists():
                sam_arr = np.array(Image.open(uploaded_sam_path).convert("L"))
            elif (local := find_server_sam_mask(image_id)):
                sam_arr = np.array(Image.open(local).convert("L"))

            save_sample(
                image_id,
                original_rgb=original_rgb,
                wz_rgb=wz_rgb,
                adres_q_rgb=None,
                adres_e_rgb=None,
                sam_mask=sam_arr,
                otsu_mask=otsu_mask_arr,
            )

        return JSONResponse({
            "status": "ok", "image_id": image_id, "rows_appended": 1,
            "mask_ref_source": mask_ref_source,
            "sample_saved": is_sample(image_id) and wz_rgb is not None,
        })

    except Exception as exc:
        raise HTTPException(status_code=500,
                            detail=f"Erreur traitement {image_id}: {exc}")


# ---------------------------------------------------------------------------
# Endpoint ADRES
# ---------------------------------------------------------------------------

@app.post("/test/submit/adres/{node_id}/{image_id}")
async def test_submit_adres(
    node_id: str,
    image_id: str,
    original:           UploadFile = File(...),
    adres_q_compressed: UploadFile = File(...),
    adres_q_roi_mask:   UploadFile = File(...),
    adres_q_metrics:    UploadFile = File(...),
    adres_e_compressed: UploadFile = File(...),
    adres_e_roi_mask:   UploadFile = File(...),
    adres_e_metrics:    UploadFile = File(...),
    sam_mask:           UploadFile = File(None),
):
    run_dir = TEST_RUN_DIR / "adres" / node_id / image_id
    run_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "original":           run_dir / "original.png",
        "adres_q_compressed": run_dir / "adres_q_compressed.p2",
        "adres_q_roi_mask":   run_dir / "adres_q_roi_mask.png",
        "adres_q_metrics":    run_dir / "adres_q_metrics.txt",
        "adres_e_compressed": run_dir / "adres_e_compressed.p2",
        "adres_e_roi_mask":   run_dir / "adres_e_roi_mask.png",
        "adres_e_metrics":    run_dir / "adres_e_metrics.txt",
        "sam_mask":           run_dir / "sam_mask.png",
    }
    for key, upload in [
        ("original", original),
        ("adres_q_compressed", adres_q_compressed),
        ("adres_q_roi_mask",   adres_q_roi_mask),
        ("adres_q_metrics",    adres_q_metrics),
        ("adres_e_compressed", adres_e_compressed),
        ("adres_e_roi_mask",   adres_e_roi_mask),
        ("adres_e_metrics",    adres_e_metrics),
    ]:
        await save_upload(upload, paths[key])

    uploaded_sam_path = None
    if sam_mask is not None:
        await save_upload(sam_mask, paths["sam_mask"])
        uploaded_sam_path = paths["sam_mask"]

    _register_seen_id(image_id)

    try:
        original_img  = Image.open(paths["original"]).convert("RGB")
        original_rgb  = np.array(original_img)
        original_gray = np.array(original_img.convert("L"))
        timestamp     = datetime.now().isoformat()

        ref_mask, mask_ref_source = load_reference_mask(
            image_id, uploaded_sam_path, original_gray)

        results       = []
        adres_q_rgb   = None
        adres_e_rgb   = None
        otsu_mask_arr = None

        for profile, comp_key, mask_key, metrics_key in [
            ("Q", "adres_q_compressed", "adres_q_roi_mask", "adres_q_metrics"),
            ("E", "adres_e_compressed", "adres_e_roi_mask", "adres_e_metrics"),
        ]:
            try:
                roi_mask_blk, roi_png_bytes, bg_png_bytes, params = \
                    adres_decode.read_compressed(paths[comp_key])
                adres_rgb = adres_decode.reconstruct_image(
                    roi_mask_blk, roi_png_bytes, bg_png_bytes, params)

                if profile == "Q":
                    adres_q_rgb = adres_rgb
                    # Masque Otsu de l'encodeur (même pour Q et E,
                    # on prend Q comme référence visuelle)
                    otsu_mask_arr = np.array(
                        Image.open(paths[mask_key]).convert("L"))
                else:
                    adres_e_rgb = adres_rgb

                adres_metrics_vals = parse_metrics_txt(paths[metrics_key])
                adres_mask = np.array(Image.open(paths[mask_key]).convert("L"))
                iou, dice  = qmetrics.compute_iou_dice(ref_mask, adres_mask)

                results.append({
                    "timestamp": timestamp, "node_id": node_id, "image_id": image_id,
                    "algorithm": "ADRES", "profile": profile,
                    "psnr_db":  round(qmetrics.compute_psnr(original_rgb, adres_rgb), 3),
                    "ssim":     round(qmetrics.compute_ssim(original_rgb, adres_rgb), 4),
                    "mask_iou": round(iou, 4), "mask_dice": round(dice, 4),
                    "mask_ref_source": mask_ref_source,
                    "cpu_time_ms":       adres_metrics_vals.get("cpu_time_ms"),
                    "memory_kb":         adres_metrics_vals.get("memory_kb"),
                    "compressed_bytes":  adres_metrics_vals.get("compressed_bytes"),
                    "compression_ratio": adres_metrics_vals.get("compression_ratio"),
                })
            except Exception as exc:
                results.append({
                    "timestamp": timestamp, "node_id": node_id, "image_id": image_id,
                    "algorithm": "ADRES", "profile": f"{profile}-ERROR",
                    "psnr_db": None, "ssim": None,
                    "mask_iou": None, "mask_dice": None,
                    "mask_ref_source": mask_ref_source,
                    "cpu_time_ms": None, "memory_kb": None,
                    "compressed_bytes": None, "compression_ratio": None,
                })
                print(f"[ADRES-{profile}] Erreur reconstruction {image_id}: {exc}")

        for row in results:
            append_result_row(row)

        # --- Sample visuel complet (ADRES produit les deux profils + original) ---
        sample_saved = False
        if is_sample(image_id):
            sam_arr = None
            if uploaded_sam_path and uploaded_sam_path.exists():
                sam_arr = np.array(Image.open(uploaded_sam_path).convert("L"))
            elif (local := find_server_sam_mask(image_id)):
                sam_arr = np.array(Image.open(local).convert("L"))

            # Récupère wz_oseg.png s'il existe déjà (endpoint WZ-OSEG traité avant)
            wz_rgb = None
            wz_path = SAMPLES_DIR / image_id / "wz_oseg.png"
            if wz_path.exists():
                wz_rgb = np.array(Image.open(wz_path).convert("RGB"))

            save_sample(
                image_id,
                original_rgb=original_rgb,
                wz_rgb=wz_rgb,
                adres_q_rgb=adres_q_rgb,
                adres_e_rgb=adres_e_rgb,
                sam_mask=sam_arr,
                otsu_mask=otsu_mask_arr,
            )
            sample_saved = True

        return JSONResponse({
            "status": "ok", "image_id": image_id,
            "rows_appended": len(results),
            "mask_ref_source": mask_ref_source,
            "sample_saved": sample_saved,
        })

    except Exception as exc:
        raise HTTPException(status_code=500,
                            detail=f"Erreur traitement {image_id}: {exc}")


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
