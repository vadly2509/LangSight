import io
import re
import os
import json
import time
import math
import base64
import inspect
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from collections import deque, defaultdict

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, Response


# LOGGING
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("langsight")

# Library lain cenderung berisik — turunkan ke WARNING agar log app bersih.
for _noisy in [
    "httpx", "httpcore", "huggingface_hub", "sentence_transformers",
    "transformers", "urllib3", "filelock",
]:
    logging.getLogger(_noisy).setLevel(logging.WARNING)



# CONFIG
HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", 8000))

# Grounding DINO — model id dari HuggingFace. Bisa di-switch runtime via /admin/model.
GDINO_MODEL_ID = os.environ.get("GDINO_MODEL_ID", "IDEA-Research/grounding-dino-base")

# Resize gambar di CPU mode supaya inference tidak terlalu lambat.
CPU_INFERENCE_SIZE = int(os.environ.get("CPU_INFERENCE_SIZE", "640"))

# Threshold default DINO. Bisa dioverride per request maupun per kelas (lihat di bawah).
BOX_THRESHOLD  = float(os.environ.get("BOX_THRESHOLD",  "0.40"))
TEXT_THRESHOLD = float(os.environ.get("TEXT_THRESHOLD", "0.25"))

# NLP — sentence transformer multilingual untuk matching query bahasa natural ke kelas target.
ST_MODEL_NAME        = "paraphrase-multilingual-MiniLM-L12-v2"
SIMILARITY_THRESHOLD = 0.40  # cosine sim minimum agar prediksi diterima
SIMILARITY_GAP       = 0.08  # gap winner vs runner-up minimum

# Path & artifact storage
MODEL_DIR = Path("langsight/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
EMBED_CACHE = MODEL_DIR / "anchor_embeddings.pkl"

# YOLO config — file bisa berupa .pt biasa atau .pt.zip (torch.load mendukung keduanya).
YOLO_MODEL_PATH = Path(os.environ.get(
    "YOLO_MODEL_PATH",
    str(MODEL_DIR / "langsight_best.pt.zip"),
))
YOLO_CONF_THR = float(os.environ.get("YOLO_CONF_THR", "0.25"))

# Mapping nama kelas YOLO native -> nama kelas target 9-class (tissue & scissor di-drop).
YOLO_TO_TARGET = {
    "pen"            : "pen",
    "pencil"         : "pencil",
    "eraser"         : "eraser",
    "stapler"        : "stapler",
    "correction_tape": "correction_tape",
    "bottle"         : "bottle",
}

# Ensemble weights — output dari notebook 03_Ensemble_Eval.
ENSEMBLE_WEIGHTS_PATH = Path(os.environ.get(
    "ENSEMBLE_WEIGHTS_PATH",
    str(MODEL_DIR / "ensemble_weights.json"),
))
WBF_IOU_THR = float(os.environ.get("WBF_IOU_THR", "0.50"))
WBF_IOU_PER_CLASS = {
    "pen"            : 0.30,
    "pencil"         : 0.30,
    "clip"           : 0.30,
    "correction_tape": 0.35,
    "stapler"        : 0.45,
}

# 9 kelas final yang didukung sistem.
TARGET_CLASSES = [
    "pen", "pencil", "eraser", "sharpener", "correction_tape",
    "stapler", "clip", "bottle", "notebook",
]
NC = len(TARGET_CLASSES)



# PROMPT ENGINEERING UNTUK GROUNDING DINO
CLASS_PROMPTS = {
    "pen"            : "ballpoint pen . ink pen . writing pen . gel pen .",
    "pencil"         : "graphite pencil . wooden pencil . lead pencil . drafting pencil .",
    "eraser"         : "rubber eraser block . white eraser . pencil eraser .",
    "sharpener"      : "pencil sharpener . handheld sharpener .",
    "correction_tape": "correction tape . white out tape . correction roller .",
    "stapler"        : "office stapler . paper stapler . manual stapler .",
    "clip"           : "paper clip . binder clip . metal paperclip .",
    "bottle"         : "water bottle . drink bottle . beverage container .",
    "notebook"       : "spiral notebook . exercise book . notepad . lined notebook .",
}

# Negative cues — diberikan ke DINO sebagai "anti-prompt" untuk kelas yang sering
CLASS_NEGATIVE_CUES = {
    "eraser"         : "notebook . book .",          # eraser kadang dikira buku kecil
    "correction_tape": "stapler . tape dispenser .", # bentuk mirip stapler kecil
    "sharpener"      : "eraser . small box .",       # sama-sama kotak kecil
    "clip"           : "pen tip . screw . correction tape ",# objek kecil metalik
    "stapler"        : "correction tape . pencil case .",
}

# IoU threshold untuk matching detection saat evaluasi TP/FP/FN.
# Objek panjang-tipis (pen, pencil) memakai threshold lebih longgar supaya
# partial-overlap dengan ground truth tetap dihitung sebagai TP.
ADAPTIVE_IOU = {
    "pen"            : 0.30,
    "pencil"         : 0.30,
    "clip"           : 0.30,
    "correction_tape": 0.35,
    "stapler"        : 0.40,
}
IOU_THRESHOLD = 0.50  # default untuk kelas lain

# Box threshold override per kelas. Kelas yang sulit dideteksi oleh DINO
# (objek kecil atau ambigu) diturunkan threshold-nya supaya recall meningkat.
PER_CLASS_BOX_THR = {
    "pencil"         : 0.25,
    "clip"           : 0.20,
    "correction_tape": 0.25,
    "sharpener"      : 0.25,
    "stapler"        : 0.25,
}

# Text threshold override per kelas. Kelas dengan prompt punya banyak sinonim
# perlu text_threshold sedikit lebih ketat untuk mencegah false trigger pada token generik.
PER_CLASS_TEXT_THR = {
    "pen"     : 0.20,   # "pen" sangat umum, ketatkan
    "bottle"  : 0.20,   # idem
    "notebook": 0.20,
    "clip"    : 0.18,   # objek kecil, sedikit lebih permisif
    "sharpener": 0.18,
}

# Threshold cosine similarity per kelas (override SIMILARITY_THRESHOLD default).
# Kelas yang punya banyak sinonim percakapan (mis. pen -> pulpen/bolpoin/biro)
# pakai threshold sedikit lebih tinggi karena anchor lebih kaya.
CLASS_THRESHOLDS = {
    "pen"            : 0.42,
    "pencil"         : 0.42,
    "clip"           : 0.48,
    "correction_tape": 0.40,
    "stapler"        : 0.40,
    "sharpener"      : 0.42,
}

# Aspect-ratio sanity check per kelas: tuple (min_ar, max_ar) untuk width/height.
# Deteksi dengan aspect ratio di luar range ini di-downweight 30%.
# None = tidak ada constraint.
CLASS_AR_RANGE = {
    "pen"            : (1.5, 12.0),  # panjang horizontal/vertikal
    "pencil"         : (1.5, 12.0),
    "clip"           : (0.5, 2.5),   # kecil dan kompak
    "bottle"         : (0.25, 1.5),  # tinggi vertikal (tapi tetap longgar)
    "notebook"       : (0.5, 2.5),
    "correction_tape": (0.6, 3.0),
    "stapler"        : (0.8, 4.0),
    "eraser"         : (0.4, 3.5),
    "sharpener"      : (0.5, 2.5),
}

# Minimum dan maximum luas box sebagai persen image area.
# Dipakai untuk menolak box yang jelas noise (terlalu kecil) atau
# over-trigger (hampir seluruh frame).
MIN_BOX_AREA_RATIO = 0.0005   # 0.05% area
MAX_BOX_AREA_RATIO = 0.85     # 85% area

# Warna bbox per kelas untuk visualisasi (RGB, akan dikonversi ke BGR saat draw).
CLASS_COLORS = {
    "pen"            : (96, 165, 250),
    "pencil"         : (167, 139, 250),
    "eraser"         : (192, 132, 252),
    "sharpener"      : (163, 230, 53),
    "correction_tape": (244, 114, 182),
    "stapler"        : (251, 191, 36),
    "clip"           : (148, 163, 184),
    "bottle"         : (248, 113, 113),
    "notebook"       : (134, 239, 172),
}



# ANCHOR SEEDS Inggris (WordNet) + Indonesia (curated)
# Seeds ini menjadi kalimat-kalimat "anchor" yang di-embed oleh Sentence Transformer
# untuk matching query bahasa natural -> kelas target. Semakin kaya seed-nya,
# semakin baik recall NLP terhadap variasi cara user menyebut objek.
_WN_SEEDS = {
    "pen": [
        "pen", "ballpoint pen", "ball pen", "ink pen", "biro pen", "biro",
        "rollerball pen", "fountain pen", "gel pen", "fineliner pen",
        "writing pen", "drawing pen", "technical pen", "felt pen",
        "ink writing instrument",
    ],
    "pencil": [
        "pencil", "lead pencil", "graphite pencil", "wooden pencil",
        "mechanical pencil", "automatic pencil", "drawing pencil",
        "carpenter pencil", "writing pencil", "sketching pencil",
        "HB pencil", "2B pencil", "graphite drawing tool",
    ],
    "eraser": [
        "eraser", "rubber eraser", "pencil eraser", "india rubber",
        "rubber", "gum eraser", "kneaded eraser", "art eraser",
        "block eraser", "white eraser",
    ],
    "sharpener": [
        "pencil sharpener", "sharpener", "rotary sharpener",
        "manual sharpener", "handheld sharpener", "wedge sharpener",
        "blade sharpener",
    ],
    "correction_tape": [
        "correction tape", "correction fluid", "white-out", "whiteout",
        "liquid paper", "tipp-ex", "tipex", "tippex",
        "correction roller", "white correction strip",
    ],
    "stapler": [
        "stapler", "paper stapler", "office stapler", "desk stapler",
        "stapling machine", "spring-loaded stapler",
        "heavy duty stapler", "binding stapler",
    ],
    "clip": [
        "paper clip", "paperclip", "metal clip", "binder clip",
        "bulldog clip", "foldback clip", "document clip",
        "wire clip", "fastener clip", "stationery clip",
    ],
    "bottle": [
        "bottle", "water bottle", "drink bottle", "tumbler",
        "drinking bottle", "plastic bottle", "thermos bottle",
        "flask", "reusable bottle", "sports bottle", "vacuum flask",
    ],
    "notebook": [
        "notebook", "exercise book", "writing book", "spiral notebook",
        "composition book", "memo book", "memo pad", "journal",
        "diary book", "school notebook", "writing journal",
        "binder notebook", "ring binder",
    ],
}

# Seed bahasa Indonesia  varian sehari-hari, brand umum, dan pola kalimat
_ID_SEEDS = {
    "pen": [
        "pulpen", "polpen", "pulpén", "pena", "pena tinta",
        "bolpoin", "bolpen", "ballpoint", "ballpen", "ball point",
        "biro", "pen tinta", "alat tulis bertinta", "pen menulis",
        "gel pen", "pena gel", "pulpen gel", "pulpen hitam", "pulpen biru",
        "pulpen merah", "pulpen warna", "fineliner", "drawing pen",
        "pen drawing", "spidol kecil", "pen kantor", "pulpen sekolah",
        "alat tulis pulpen", "benda menulis tinta", "pen pilot", "pen standard",
        "pulpen saya", "pulpen ku", "pulpenku",
        "mana pulpenku", "di mana pulpen saya",
    ],
    "pencil": [
        "pensil", "pinsil", "pensel", "pensil kayu", "pensil grafit",
        "pensil mekanik", "mechanical pencil", "pensil HB", "pensil 2B",
        "pensil 2H", "pensil H", "pensil B", "pensil tulis", "pensil gambar",
        "pensil sketsa", "pensil arsir", "batang grafit",
        "pensil staedtler", "pensil faber castell", "pensil joyko",
        "pensil panjang", "pensil pendek", "pensil ujian", "pensil sekolah",
        "alat tulis pensil", "pensil hitam", "pensilku",
        "pensil saya", "di mana pensil", "mana pensilku",
    ],
    "eraser": [
        "penghapus", "penghapus karet", "karet penghapus", "karet hapus",
        "karet hapus pensil", "setip", "stip", "karet setip", "stipo",
        "eraser", "penghapus tulisan", "penghapus pensil", "penghapus kotak",
        "penghapus putih", "karet hapus tulisan", "penghapus tombow",
        "penghapus staedtler", "penghapus papermate",
        "penghapus saya", "penghapus ku", "penghapusku",
        "di mana penghapus", "mana stip",
    ],
    "sharpener": [
        "rautan", "rautan pensil", "serutan", "serutan pensil",
        "peraut pensil", "peruncing pensil", "alat raut",
        "alat meruncing pensil", "alat menajamkan pensil", "sharpener",
        "rautan kotak", "rautan plastik", "rautan besi", "rautan kecil",
        "rautan staedtler", "rautan joyko",
        "rautan saya", "rautanku", "di mana rautan", "mana rautan",
    ],
    "correction_tape": [
        "tipex", "tipex roll", "tipex roller", "tip-ex", "tipe-x", "tipp-ex",
        "tip x", "tip-x", "tippex", "stipo", "stipoe",
        "correction tape", "pita koreksi", "pita tipex",
        "penghapus tulisan pita putih", "pita penutup tulisan",
        "alat menutupi tulisan salah", "pita warna putih untuk koreksi",
        "kertas tipex", "tipex putih", "alat koreksi",
        "kenko tipex", "joyko tipex", "tipex roller pita",
        "tipex saya", "tipexku", "di mana tipex",
    ],
    "stapler": [
        "stapler", "stepler", "stapless", "steples", "staples",
        "hekter", "hecter", "jegrek", "jeklek",
        "alat menjepit kertas dengan kawat staples",
        "alat staples kertas", "penjepret kertas", "alat hekter",
        "stapler kantor", "stapler kecil", "stapler kayu", "stapler besi",
        "kenko stapler", "joyko stapler", "max stapler",
        "stapler saya", "staplerku", "di mana stapler", "mana hekter",
    ],
    "clip": [
        "klip", "klip kertas", "paper clip", "paperclip", "kliper",
        "binder clip", "klip binder", "bulldog clip", "klip bulldog",
        "klip jepit", "klip jepit kertas", "klip hitam", "klip warna",
        "jepitan kertas", "jepitan kertas logam", "jepitan kawat",
        "klip besi", "klip kawat", "penjepit dokumen", "penjepit kertas",
        "klip dokumen", "klip foldback", "foldback clip",
        "klip saya", "klipku", "di mana klip", "mana paper clip",
    ],
    "bottle": [
        "botol", "botol minum", "botol minuman", "botol air",
        "botol air minum", "tumbler", "tumblr", "tumbler air",
        "botol plastik", "botol kaca", "botol air mineral",
        "botol bekal", "tempat minum", "wadah minum", "wadah air",
        "botol reusable", "thermos", "thermos botol", "termos",
        "flask", "botol olahraga", "botol sport", "botol kuliah",
        "botol sekolah", "botol kantor", "botol aqua", "botol mineral",
        "botol saya", "botolku", "di mana botol", "mana tumbler",
    ],
    "notebook": [
        "buku", "buku tulis", "buku catatan", "buku spiral",
        "notebook", "notebook spiral", "buku notes", "notes",
        "binder", "buku binder", "buku binder kuliah",
        "buku pelajaran", "buku bergaris", "buku tugas", "buku PR",
        "buku gambar", "buku sketsa", "buku kosong", "buku coretan",
        "buku tulis sekolah", "buku catatan kuliah", "buku diary",
        "diary", "jurnal", "buku jurnal", "buku agenda", "agenda",
        "buku saya", "bukuku", "di mana buku", "mana notebook",
    ],
}



# SESSION LOG & RUNTIME STATE

# In-memory log untuk export sesi (debugging dan demo). Tidak persistent.
SESSION_LOG: list[dict] = []

# Riwayat jumlah deteksi per frame untuk dynamic threshold adjustment.
FRAME_HISTORY = deque(maxlen=10)

# Distribusi confidence per kelas untuk statistik.
CLASS_CONF_HISTORY: dict = defaultdict(list)

# Kalibrasi Platt scaling per kelas. Default empty = passthrough.
# Format: {class_name: {"a": float, "b": float}}, formula 1/(1+exp(a*s+b)).
_CALIBRATION: dict = {}


def log_session(query, target_class, method, n_det, elapsed_ms, similarity=None):
    """Tambahkan satu baris ke SESSION_LOG dengan rotasi 500 entry."""
    SESSION_LOG.append({
        "ts"          : time.strftime("%H:%M:%S"),
        "query"       : query,
        "target_class": target_class,
        "method"      : method,
        "n_detections": n_det,
        "elapsed_ms"  : elapsed_ms,
        "similarity"  : round(similarity, 4) if similarity else None,
    })
    if len(SESSION_LOG) > 500:
        SESSION_LOG.pop(0)


def calibrate_score(cls: str, raw_score: float) -> float:
    """Terapkan Platt scaling kalau ada kalibrasi terdaftar untuk kelas ini."""
    if cls not in _CALIBRATION:
        return raw_score
    a = _CALIBRATION[cls].get("a", -1.0)
    b = _CALIBRATION[cls].get("b", 0.0)
    try:
        return 1.0 / (1.0 + math.exp(a * raw_score + b))
    except OverflowError:
        return 0.0 if a * raw_score + b > 0 else 1.0


def get_dynamic_threshold(base_threshold: float) -> float:
    """Naikkan threshold otomatis kalau frame-frame sebelumnya banyak deteksi
    (mengindikasikan kemungkinan false positive bertubi-tubi)."""
    if len(FRAME_HISTORY) < 5:
        return base_threshold
    recent = [f["n_det"] for f in FRAME_HISTORY]
    avg = sum(recent) / len(recent)
    if avg > 4:
        return min(base_threshold + 0.10, 0.65)
    if avg > 2:
        return min(base_threshold + 0.05, 0.60)
    return base_threshold



# TEMPORAL SMOOTHING
_TEMPORAL_WINDOW   = 3
_TEMPORAL_ALPHA    = 0.6
_temporal_history: dict = defaultdict(list)


def temporal_smooth(target_class: str, new_dets: list) -> list:
    now = time.time()
    history = _temporal_history[target_class]

    # Buang history yang lebih lama dari 4 detik.
    history = [h for h in history if now - h.get("ts", 0) < 4.0]

    # Kasus: tidak ada deteksi baru tapi ada riwayat -> mungkin objek sebenarnya masih ada.
    if not new_dets and history:
        last_confs = [h["conf"] for h in history[-_TEMPORAL_WINDOW:]]
        if len(last_confs) >= 2 and sum(last_confs) / len(last_confs) > 0.35:
            ghost = history[-1].copy()
            ghost["confidence"] = round(ghost["conf"] * 0.7, 4)
            ghost["smoothed"]   = True
            ghost["ghost"]      = True
            history.append({"bbox": ghost["bbox"], "conf": ghost["confidence"], "ts": now})
            _temporal_history[target_class] = history[-_TEMPORAL_WINDOW * 2:]
            return [ghost]
        _temporal_history[target_class] = history
        return []

    smoothed = []
    for det in new_dets:
        bbox = det["bbox"]
        conf = det["confidence"]

        matching = [h for h in history if _iou(bbox, h["bbox"]) > 0.40]

        if len(matching) >= 2:
            # Muncul konsisten -> exponential moving average + bonus konsistensi.
            avg_hist = sum(h["conf"] for h in matching) / len(matching)
            smoothed_conf = _TEMPORAL_ALPHA * conf + (1 - _TEMPORAL_ALPHA) * avg_hist
            consistency_bonus = min(0.08, len(matching) * 0.03)
            smoothed_conf = min(0.99, smoothed_conf + consistency_bonus)
        elif len(matching) == 1:
            smoothed_conf = _TEMPORAL_ALPHA * conf + (1 - _TEMPORAL_ALPHA) * matching[0]["conf"]
        else:
            # Baru muncul -> turunkan untuk meredam noise frame tunggal.
            smoothed_conf = conf * 0.85

        d = det.copy()
        d["confidence"] = round(smoothed_conf, 4)
        d["smoothed"]   = True
        smoothed.append(d)
        history.append({"bbox": bbox, "conf": smoothed_conf, "ts": now})

    _temporal_history[target_class] = history[-_TEMPORAL_WINDOW * 2:]
    return smoothed



# ANCHOR BUILDER — WordNet (EN/ID) + manual seeds

def build_anchors() -> dict:
    """Susun list anchor sentences per kelas dengan menggabungkan:
        1) Seed manual Inggris (curated)
        2) Lemma WordNet (Inggris, Indonesia, Malay)
        3) Seed manual Indonesia (curated, banyak slang)
    """
    try:
        import nltk
        from nltk.corpus import wordnet as wn

        for res, path in [("wordnet", "corpora/wordnet"), ("omw-1.4", "corpora/omw-1.4")]:
            try:
                nltk.data.find(path)
            except LookupError:
                log.info(f"  Downloading NLTK: {res}")
                nltk.download(res, quiet=True)

        anchors = {}
        for cls in TARGET_CLASSES:
            seeds  = _WN_SEEDS.get(cls, [cls.replace("_", " ")])
            en_set = set(seeds)
            for seed in seeds:
                for syn in wn.synsets(seed.replace(" ", "_"), pos=wn.NOUN):
                    for lemma in syn.lemmas():
                        en_set.add(lemma.name().replace("_", " ").lower())
                    for lang in ("ind", "zsm"):
                        try:
                            for lemma in syn.lemmas(lang=lang):
                                en_set.add(lemma.name().replace("_", " ").lower())
                        except Exception:
                            pass
            id_words = _ID_SEEDS.get(cls, [])
            anchors[cls] = list(en_set) + [w for w in id_words if w not in en_set]
        return anchors
    except Exception as e:
        log.warning(f"WordNet error ({e}) - fallback ke seed manual saja")
        return {
            cls: (_WN_SEEDS.get(cls, []) + _ID_SEEDS.get(cls, []))
            for cls in TARGET_CLASSES
        }



# NLP ENGINE — query bahasa natural -> target class

class LangSightNLP:
    """Menerjemahkan kalimat bebas user (ID/EN) menjadi salah satu dari 9 kelas
    target dengan dua mekanisme:
        1) Semantic match via Sentence Transformer (utama)
        2) Keyword match dari anchor seeds (fallback)

    Multi-object query ("pulpen dan pensil") dipecah lewat parse_multi().
    """

    # Kata-kata yang muncul di query tapi BUKAN kelas target -> auto-reject.
    _BLACKLIST = {
        "person", "people", "orang", "manusia", "tangan", "hand", "face", "wajah",
        "meja", "table", "chair", "kursi", "dinding", "wall", "floor", "lantai",
        "laptop", "phone", "hp", "handphone", "komputer", "computer", "monitor",
        "cat", "dog", "anjing", "kucing", "bird", "burung", "car", "mobil",
        "sofa", "television", "tv", "keyboard", "mouse", "headphone", "earphone",
        "glass", "piring", "sendok", "garpu", "baju", "celana", "sepatu", "tas",
        "drawing", "painting", "art", "lukisan",
    }

    # Conjunction untuk memecah multi-object query.
    _CONJUNCTIONS = {"dan", "and", "dengan", "sama", "juga", "plus", "beserta", "atau", "or"}

    def __init__(self, model_name: str = ST_MODEL_NAME, anchors: dict = None):
        self.anchors = anchors or {}
        self._active = {k: v for k, v in self.anchors.items() if k in TARGET_CLASSES and v}
        log.info(f"NLP kelas aktif: {len(self._active)}")

        try:
            from sentence_transformers import SentenceTransformer, util
            self._st   = SentenceTransformer(model_name)
            self._util = util
            self._use_semantic = True
            self._embs = {}
            for cls, terms in self._active.items():
                self._embs[cls] = self._st.encode(
                    terms, convert_to_tensor=True, normalize_embeddings=True,
                )
            log.info(f"Semantic NLP siap: {len(self._embs)} kelas")
        except ImportError:
            log.warning("sentence-transformers tidak tersedia - keyword fallback")
            self._use_semantic = False

    def parse_multi(self, query: str) -> list[str]:
        """Pecah query menjadi sub-queries kalau ada conjunction (dan/atau/etc)."""
        q = query.strip().lower()
        for conj in self._CONJUNCTIONS:
            pattern = r"\b" + re.escape(conj) + r"\b"
            if re.search(pattern, q):
                parts = re.split(pattern, q, flags=re.IGNORECASE)
                return [p.strip() for p in parts if p.strip()]
        return [query]

    def predict(self, query: str):
        """Return (target_class, method, ranked_scores). target_class None kalau ditolak."""
        q = query.strip()
        if not q:
            return None, "empty", []

        q_lower = q.lower()
        for bw in self._BLACKLIST:
            if bw in q_lower:
                log.info(f"  NLP '{q}' -> BLACKLIST ({bw})")
                return None, "blacklist", []

        if not self._use_semantic:
            return self._keyword_predict(q)

        try:
            qvec = self._st.encode(q, convert_to_tensor=True, normalize_embeddings=True)
            scores = {
                c: float(self._util.cos_sim(qvec, e)[0].max())
                for c, e in self._embs.items()
            }
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            top_cls, top_score = ranked[0]
            gap = top_score - ranked[1][1] if len(ranked) > 1 else 1.0

            log.info(
                f"  NLP '{q}' -> Top-3: "
                f"{[(c, round(s, 3)) for c, s in ranked[:3]]} gap={gap:.3f}"
            )

            thresh = CLASS_THRESHOLDS.get(top_cls, SIMILARITY_THRESHOLD)

            # Skor di bawah threshold -> coba keyword sebagai fallback dulu.
            if top_score < thresh:
                kw = self._keyword_predict(q)
                if kw[0]:
                    log.info(f"  NLP keyword fallback -> {kw[0]}")
                    return kw
                return None, "semantic", ranked

            # Gap terlalu tipis antara top-1 dan top-2 -> ambigu, coba keyword.
            if gap < SIMILARITY_GAP:
                kw = self._keyword_predict(q)
                if kw[0]:
                    log.info(f"  NLP gap kecil ({gap:.3f}) -> keyword fallback -> {kw[0]}")
                    return kw
                log.info(f"  NLP gap terlalu kecil ({gap:.3f}) - rejected")
                return None, "semantic", ranked

            return top_cls, "semantic", ranked

        except Exception as e:
            log.error(f"NLP error: {e}")
            return self._keyword_predict(q)

    def _keyword_predict(self, query: str):
        """Exact-substring match query terhadap anchor seeds."""
        n = re.sub(r"[^a-z0-9\s]", " ", query.lower())
        n = re.sub(r"\s+", " ", n).strip()
        for cls, terms in self._active.items():
            for t in terms:
                if t.lower() in n:
                    return cls, "keyword", [(cls, 1.0)]
        return None, "keyword", []



# YOLOv11n DETECTOR — closed-set model fine-tuned di dataset alat tulis

class YOLODetector:
    """Wrapper YOLOv11n untuk inference 6 kelas alat tulis. Satu kali predict
    menghasilkan semua kelas yang model dukung, lalu di-map ke nama target
    9-class (tissue dan scissor di-drop)."""

    def __init__(self, model_path: Path):
        import torch
        from ultralytics import YOLO

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"Loading YOLOv11n: {model_path} | device={self.device}")

        if not model_path.exists():
            raise FileNotFoundError(
                f"YOLO model tidak ditemukan: {model_path}. "
                "Set env var YOLO_MODEL_PATH atau letakkan file di langsight/models/"
            )

        # torch.load bisa baca .pt biasa maupun .pt.zip langsung.
        ckpt = torch.load(str(model_path), map_location="cpu", weights_only=False)
        model_raw = ckpt["model"].float().to(self.device).eval()

        # ultralytics.YOLO() butuh file .pt -> dump ke file temp lalu load.
        tmp_pt = MODEL_DIR / "_yolo_runtime.pt"
        torch.save({"model": model_raw, "names": model_raw.names}, str(tmp_pt))

        self.model = YOLO(str(tmp_pt))
        self.model.to(self.device)
        self.native_names = self.model.names
        log.info(f"YOLO native names: {self.native_names}")

        # Tampilkan metric latihan kalau ada (informasi saja).
        if isinstance(ckpt, dict) and ckpt.get("train_metrics"):
            tm = ckpt["train_metrics"]
            log.info(
                f"YOLO train metrics: "
                f"mAP@50={tm.get('metrics/mAP50(B)', 0):.4f}  "
                f"Recall={tm.get('metrics/recall(B)', 0):.4f}  "
                f"Precision={tm.get('metrics/precision(B)', 0):.4f}"
            )

    def detect(self, img_bgr, target_classes=None, conf=YOLO_CONF_THR):
        """Predict lalu filter & remap ke target 9-class.
        Return list of {class, confidence, bbox, source="yolo"}."""
        results = self.model.predict(
            img_bgr, verbose=False, conf=conf, device=self.device,
        )
        out = []
        if not results or results[0].boxes is None:
            return out

        boxes = results[0].boxes
        for i in range(len(boxes)):
            cls_id   = int(boxes.cls[i].item())
            cls_name = self.native_names[cls_id]
            target   = YOLO_TO_TARGET.get(cls_name)
            if not target:
                continue
            if target_classes and target not in target_classes:
                continue

            score = float(boxes.conf[i].item())
            xyxy  = boxes.xyxy[i].tolist()
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            out.append({
                "class"     : target,
                "confidence": round(score, 4),
                "bbox"      : [x1, y1, x2, y2],
                "source"    : "yolo",
            })
        return out



# WEIGHTED BOX FUSION — gabungkan YOLO + DINO per target_class

class WBFEnsemble:
    """Fuse deteksi YOLO dan DINO untuk satu target_class:

        1) YOLO inference (kalau kelas ada di YOLO native).
        2) DINO inference dengan prompt khusus kelas tersebut.
        3) Cluster box yang IoU >= wbf_iou_thr dan fuse koordinat dengan
           weighted average (weight = w_model * box_score).
        4) Score akhir: weighted sum dari skor masing-masing model.

    Bobot w_yolo dan w_dino per-kelas berasal dari evaluasi:
        w = (F1 x AP@50)^2 / sum-kuadrat-keduanya

    Source tag pada output:
        - "ensemble" : dua model nyumbang ke cluster yang sama
        - "yolo"     : hanya YOLO yang nyumbang
        - "dino"     : hanya DINO yang nyumbang
        - "dino_only": kelas yang tidak ada di YOLO (sharpener/clip/notebook)
    """

    def __init__(self, yolo_det, dino_det, weights_payload: dict,
                 wbf_iou: float = WBF_IOU_THR):
        self.yolo = yolo_det
        self.dino = dino_det
        self.meta = weights_payload.get("meta", {})
        self.weights = weights_payload.get("weights", {})
        self.wbf_iou = wbf_iou

        for cls in TARGET_CLASSES:
            if cls not in self.weights:
                log.warning(f"Bobot untuk '{cls}' tidak ada di ensemble_weights.json")

        self.yolo_classes      = set(self.meta.get("yolo_eval_classes",
                                                    list(YOLO_TO_TARGET.values())))
        self.dino_only_classes = set(self.meta.get("dino_only_classes", []))

        log.info(f"WBFEnsemble loaded - yolo classes: {sorted(self.yolo_classes)}")
        log.info(f"                   - dino-only   : {sorted(self.dino_only_classes)}")

    def _weights_for(self, cls: str):
        w = self.weights.get(cls, {"w_yolo": 0.5, "w_dino": 0.5})
        return float(w.get("w_yolo", 0.5)), float(w.get("w_dino", 0.5))

    def detect(self, img_bgr, target_class: str,
               yolo_conf: float = None,
               dino_box_thr: float = None,
               dino_text_thr: float = None):
        """Inference YOLO + DINO untuk satu target lalu fuse hasilnya."""
        # YOLO inference (skip kalau kelas tidak ada di model YOLO).
        if target_class in self.yolo_classes:
            yolo_dets = self.yolo.detect(
                img_bgr,
                target_classes=[target_class],
                conf=yolo_conf if yolo_conf is not None else YOLO_CONF_THR,
            )
        else:
            yolo_dets = []

        # DINO selalu jalan (juga untuk kelas DINO-only).
        dino_dets = self.dino.detect(
            img_bgr, target_class,
            box_threshold=dino_box_thr,
            text_threshold=dino_text_thr,
        )
        for d in dino_dets:
            d["source"] = "dino"

        # DINO-only: skip WBF, langsung return karena tidak ada partner YOLO.
        if target_class in self.dino_only_classes:
            for d in dino_dets:
                d["source"] = "dino_only"
            return dino_dets

        w_yolo, w_dino = self._weights_for(target_class)
        return self._fuse(yolo_dets, dino_dets, w_yolo, w_dino, target_class)

    def _fuse(self, yolo_dets, dino_dets, w_yolo, w_dino, target_class):
        """Cluster overlapping boxes dan fuse koordinat + score per cluster."""
        items = []
        for d in yolo_dets:
            items.append({"score": d["confidence"], "bbox": d["bbox"],
                          "src": "yolo", "w": w_yolo})
        for d in dino_dets:
            items.append({"score": d["confidence"], "bbox": d["bbox"],
                          "src": "dino", "w": w_dino})

        if not items:
            return []

        # Sort skor terbesar pertama supaya cluster di-seed dari box paling yakin.
        items.sort(key=lambda x: -x["score"])
        used = [False] * len(items)
        fused = []

        for i in range(len(items)):
            if used[i]:
                continue
            cluster = [items[i]]
            used[i] = True
            for j in range(i + 1, len(items)):
                if used[j]:
                    continue
                iou_thr = WBF_IOU_PER_CLASS.get(target_class, self.wbf_iou)
                if _iou(items[i]["bbox"], items[j]["bbox"]) >= iou_thr:
                    cluster.append(items[j])
                    used[j] = True

            # Weighted-average koordinat.
            ws    = [c["w"] * c["score"] for c in cluster]
            sum_w = sum(ws)
            if sum_w == 0:
                continue

            fx1 = sum(c["bbox"][0] * w for c, w in zip(cluster, ws)) / sum_w
            fy1 = sum(c["bbox"][1] * w for c, w in zip(cluster, ws)) / sum_w
            fx2 = sum(c["bbox"][2] * w for c, w in zip(cluster, ws)) / sum_w
            fy2 = sum(c["bbox"][3] * w for c, w in zip(cluster, ws)) / sum_w

            # Score per sumber (max kalau ada beberapa box dari sumber sama).
            per_src = {}
            for c in cluster:
                per_src[c["src"]] = max(per_src.get(c["src"], 0), c["score"])

            srcs = set(c["src"] for c in cluster)
            if len(srcs) == 1:
                src = next(iter(srcs))
                ens_score  = per_src[src]
                source_tag = src
            else:
                ens_score  = w_yolo * per_src.get("yolo", 0) + w_dino * per_src.get("dino", 0)
                source_tag = "ensemble"

            fused.append({
                "class"       : target_class,
                "confidence"  : round(min(1.0, ens_score), 4),
                "bbox"        : [int(fx1), int(fy1), int(fx2), int(fy2)],
                "source"      : source_tag,
                "contributors": sorted(list(srcs)),
            })

        fused.sort(key=lambda x: -x["confidence"])
        # Safety NMS untuk kasus fused boxes yang masih overlap, lalu cap di 5.
        fused = _nms(fused)[:5]
        return temporal_smooth(target_class, fused)


def _load_ensemble_weights(path: Path):
    """Load ensemble_weights.json atau buat fallback 50:50."""
    if not path.exists():
        log.warning(
            f"ensemble_weights.json tidak ditemukan di {path}. "
            "Fallback ke bobot default 50:50 (kecuali DINO-only classes)."
        )
        return {
            "meta": {
                "yolo_eval_classes": list(YOLO_TO_TARGET.values()),
                "dino_only_classes": [
                    c for c in TARGET_CLASSES if c not in YOLO_TO_TARGET.values()
                ],
                "fallback_default": True,
            },
            "weights": {
                cls: {
                    "w_yolo": 0.0 if cls not in YOLO_TO_TARGET.values() else 0.5,
                    "w_dino": 1.0 if cls not in YOLO_TO_TARGET.values() else 0.5,
                    "reason": "fallback_default",
                } for cls in TARGET_CLASSES
            },
        }
    with open(path) as f:
        return json.load(f)

def preprocess_adaptive(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if gray.mean() < 60:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    elif gray.mean() > 200:
        return cv2.convertScaleAbs(img_bgr, alpha=0.7, beta=0)
    elif gray.std() < 30:
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return img_bgr

# GROUNDING DINO DETECTOR — zero-shot text-prompted object detection
class GroundingDINODetector:
    """Wrapper Grounding DINO dengan beberapa optimisasi tambahan:

        1. Per-class box & text threshold override (lihat PER_CLASS_BOX_THR
           dan PER_CLASS_TEXT_THR di config).
        2. Aspect-ratio sanity check: bbox yang aspect-ratio-nya di luar
           rentang wajar untuk kelas tersebut akan di-downweight 30%.
        3. Box-area filter: box terlalu kecil (noise) atau hampir seluruh frame
           (over-trigger) langsung dibuang.
        4. Negative-cue filter: kalau label yang dikembalikan DINO match dengan
           kata yang sering bingung untuk kelas ini, skor di-downweight 40%.
        5. Confidence calibration (Platt scaling, opsional per-kelas).
        6. CPU mode auto-resize gambar ke CPU_INFERENCE_SIZE.
        7. Dynamic threshold dari frame history (anti-burst FP).
    """

    def __init__(self, model_id: str = GDINO_MODEL_ID):
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        self.device   = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        log.info(f"Loading Grounding DINO: {model_id} | device={self.device}")

        if self.device == "cpu":
            log.warning("CPU mode - latency ~2-5s per frame")

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model     = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        self.model     = self.model.to(self.device)
        self.model.eval()

        # Sejak transformers 4.51 signature post-process berubah:
        #   - lama : post_process_grounded_object_detection(..., box_threshold=, text_threshold=)
        #   - baru : post_process_grounded_object_detection(..., threshold=, text_labels=)
        # Deteksi versi via introspeksi parameter.
        params = inspect.signature(
            self.processor.post_process_grounded_object_detection
        ).parameters
        self._use_new_api = "box_threshold" not in params
        log.info(f"API mode: {'new (threshold=)' if self._use_new_api else 'old (box_threshold=)'}")

    # helper 
    def _resize(self, img_pil: Image.Image) -> Image.Image:
        """Downscale di CPU mode supaya inference muat di latency budget."""
        if self.device != "cpu" or CPU_INFERENCE_SIZE is None:
            return img_pil
        w, h = img_pil.size
        if max(w, h) <= CPU_INFERENCE_SIZE:
            return img_pil
        scale = CPU_INFERENCE_SIZE / max(w, h)
        return img_pil.resize((int(w * scale), int(h * scale)), Image.BILINEAR)

    @staticmethod
    def _aspect_penalty(target_class: str, x1: int, y1: int, x2: int, y2: int) -> float:
        """Beri faktor pengurang skor kalau aspect-ratio bbox di luar rentang
        wajar untuk kelas tersebut. Return 1.0 kalau aman, 0.7 kalau dipenalti."""
        ar_range = CLASS_AR_RANGE.get(target_class)
        if not ar_range:
            return 1.0
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        ar = max(w / h, h / w)  # always >= 1, mencakup objek vertikal & horizontal
        lo, hi = ar_range
        if lo <= ar <= hi:
            return 1.0
        # Penalti soft: ar di luar batas -> skor x 0.7
        return 0.7

    # inference 
    def detect(self, img_bgr, target_class, box_threshold=None, text_threshold=None):
        """Inference DINO untuk satu target_class dan return list of
        {class, label, confidence, bbox}. Sudah lewat:
            - per-class thresholds
            - dynamic frame-history threshold adjustment
            - area & aspect-ratio sanity check
            - Platt-scaling calibration (kalau di-set)
            - NMS adaptive per kelas
            - temporal smoothing
        """
        import torch
        img_bgr = preprocess_adaptive(img_bgr)

        # Threshold final: per-class override -> argument -> default global.
        base_bt = PER_CLASS_BOX_THR.get(
            target_class, box_threshold if box_threshold is not None else BOX_THRESHOLD
        )
        dyn_bt = get_dynamic_threshold(base_bt)
        tt = PER_CLASS_TEXT_THR.get(
            target_class, text_threshold if text_threshold is not None else TEXT_THRESHOLD
        )
        prompt = CLASS_PROMPTS.get(target_class, f"{target_class.replace('_', ' ')} .")

        img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        img_inf = self._resize(img_pil)

        try:
            # Tokenisasi cache: kalau prompt sama persis dan ukuran gambar inference sama,
            # tidak perlu rekonstruksi text_inputs (image_inputs tetap di-recompute karena beda gambar).
            inputs = self.processor(
                images=img_inf, text=prompt, return_tensors="pt",
            ).to(self.device)

            with torch.inference_mode():
                if self.device == "cuda":
                    with torch.autocast(device_type="cuda"):
                        outputs = self.model(**inputs)
                else:
                    outputs = self.model(**inputs)

            if self._use_new_api:
                results = self.processor.post_process_grounded_object_detection(
                    outputs, inputs.input_ids,
                    threshold=dyn_bt,
                    target_sizes=[img_inf.size[::-1]],
                )
            else:
                results = self.processor.post_process_grounded_object_detection(
                    outputs, inputs.input_ids,
                    box_threshold=dyn_bt, text_threshold=tt,
                    target_sizes=[img_inf.size[::-1]],
                )

            if not results or results[0]["boxes"] is None:
                return []

            # transformers 4.51+: "labels" int, "text_labels" string. Pakai yang string.
            lbl_key = "text_labels" if "text_labels" in results[0] else "labels"
            raw = list(zip(
                results[0]["scores"],
                results[0][lbl_key],
                results[0]["boxes"],
            ))
            log.info(
                f"  GDINO raw [{target_class}]: "
                f"{[(str(l)[:10], round(float(s), 3)) for s, l, _ in raw[:5]]}"
            )

            # Scale balik koordinat dari ukuran inference ke ukuran asli.
            sx = img_pil.width  / img_inf.width
            sy = img_pil.height / img_inf.height
            img_area = img_pil.width * img_pil.height

            # Negative cue keywords untuk kelas ini (kalau ada).
            # Diparsing sekali di luar loop supaya tidak diulang per box.
            neg_keywords = set()
            neg_str = CLASS_NEGATIVE_CUES.get(target_class, "")
            for token in re.split(r"\s*\.\s*|\s+", neg_str.strip()):
                token = token.strip().lower()
                if token and len(token) >= 3:
                    neg_keywords.add(token)

            dets = []
            for score, label, box in raw:
                x1, y1, x2, y2 = box.tolist()
                x1 = max(0, int(x1 * sx))
                y1 = max(0, int(y1 * sy))
                x2 = min(img_pil.width,  int(x2 * sx))
                y2 = min(img_pil.height, int(y2 * sy))

                bw = max(1, x2 - x1)
                bh = max(1, y2 - y1)
                area_ratio = (bw * bh) / img_area

                # Filter area: terlalu kecil = noise, terlalu besar = over-trigger.
                if area_ratio < MIN_BOX_AREA_RATIO or area_ratio > MAX_BOX_AREA_RATIO:
                    continue

                # Aspect-ratio sanity check: penalti skor 30% kalau di luar rentang.
                ar_factor = self._aspect_penalty(target_class, x1, y1, x2, y2)

                # Negative cue check: kalau label yang dikembalikan DINO match dengan
                # keyword yang sering bingung untuk kelas ini, downweight 40%.
                neg_factor = 1.0
                if neg_keywords:
                    lbl_lower = str(label).lower()
                    if any(nk in lbl_lower for nk in neg_keywords):
                        neg_factor = 0.6

                # Calibration kalau di-set.
                final_score = (calibrate_score(target_class, float(score))
                               * ar_factor * neg_factor)

                # Drop kalau setelah penalti turun di bawah threshold.
                if final_score < base_bt * 0.85:
                    continue

                dets.append({
                    "class"     : target_class,
                    "label"     : str(label),
                    "confidence": round(final_score, 4),
                    "raw_score" : round(float(score), 4),
                    "bbox"      : [x1, y1, x2, y2],
                })

            dets = _nms(dets)[:5]
            return temporal_smooth(target_class, dets)

        except Exception as e:
            log.error(f"DINO detect error ({target_class}): {e}")
            return []

    def detect_all(self, img_bgr, box_threshold=None, text_threshold=None):
        """Scan SEMUA 9 kelas. Lebih lambat (9x inference), dipakai oleh /detect/all."""
        all_dets = []
        for cls in TARGET_CLASSES:
            dets = self.detect(img_bgr, cls, box_threshold, text_threshold)
            all_dets.extend(dets)
        return _nms(all_dets, 0.5)[:15]



# HELPERS — IoU, NMS, drawing, image utils

def _iou(a, b):
    """Intersection-over-Union antara dua bbox [x1, y1, x2, y2]."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if not inter:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


# IoU threshold NMS adaptif per-kelas.
# Pen / pencil panjang dan tipis -> IoU lebih rendah supaya partial-overlap di-merge.
# Clip kecil dan sering berdekatan -> IoU lebih tinggi supaya tidak over-merge.
_CLASS_NMS_IOU = {
    "pen"   : 0.35,
    "pencil": 0.35,
    "clip"  : 0.60,
}


def _nms(dets, iou_thr=None):
    """Non-Maximum Suppression dengan IoU threshold per-kelas."""
    if len(dets) <= 1:
        return dets
    dets = sorted(dets, key=lambda x: x["confidence"], reverse=True)
    kept = []
    for d in dets:
        thr = _CLASS_NMS_IOU.get(d.get("class", ""), iou_thr or 0.50)
        if all(_iou(d["bbox"], k["bbox"]) < thr for k in kept):
            kept.append(d)
    return kept


def draw_detections(img_bgr, dets, show_source=True):
    """Gambar bbox + label + source tag ke gambar. show_source menampilkan
    singkatan sumber deteksi (yol/din/ens/d-o) buat visual debugging ensemble."""
    out = img_bgr.copy()
    for d in dets:
        cls   = d["class"]
        color = CLASS_COLORS.get(cls, (99, 202, 183))
        bgr   = (color[2], color[1], color[0])
        x1, y1, x2, y2 = d["bbox"]
        conf = d["confidence"]
        bw   = max(x2 - x1, 1)

        cv2.rectangle(out, (x1, y1), (x2, y2), bgr, 2)
        # Bar confidence di bawah bbox.
        cv2.rectangle(out, (x1, y2 - 5), (x2, y2), (30, 30, 30), -1)
        cv2.rectangle(out, (x1, y2 - 5), (x1 + int(bw * conf), y2), bgr, -1)

        src = d.get("source", "")
        if show_source and src:
            src_tag = {"yolo": "yol", "dino": "din",
                       "ensemble": "ens", "dino_only": "d-o"}.get(src, src[:3])
            lbl = f"{cls} {conf * 100:.0f}% [{src_tag}]"
        else:
            lbl = f"{cls} {conf * 100:.0f}%"

        (lw, lh), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        ly = y1 - 8 if y1 > lh + 12 else y1 + lh + 8
        cv2.rectangle(out, (x1, ly - lh - 6), (x1 + lw + 10, ly + 2), bgr, -1)
        cv2.putText(out, lbl, (x1 + 5, ly - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def img_to_b64(img_bgr, quality: int = 88):
    """BGR ndarray -> base64 JPEG string (untuk inline-display di frontend)."""
    buf = io.BytesIO()
    Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).save(
        buf, format="JPEG", quality=quality,
    )
    return base64.b64encode(buf.getvalue()).decode()


def read_image(raw: bytes):
    """Upload bytes -> BGR ndarray. PIL dipakai supaya format apa saja masuk."""
    return cv2.cvtColor(
        np.array(Image.open(io.BytesIO(raw)).convert("RGB")),
        cv2.COLOR_RGB2BGR,
    )



# APPLICATION LIFECYCLE

# Global singletons — diinisialisasi di lifespan dan dipakai oleh semua endpoint.
nlp:       LangSightNLP        = None
detector:  GroundingDINODetector = None
yolo_det:  YOLODetector        = None
ensemble:  WBFEnsemble         = None
ANCHORS:   dict                = {}
ENSEMBLE_WEIGHTS: dict         = {}


@asynccontextmanager
async def lifespan(app):
    """Inisialisasi semua model saat startup. Dipanggil otomatis oleh FastAPI."""
    global nlp, detector, yolo_det, ensemble, ANCHORS, ENSEMBLE_WEIGHTS

    log.info("LangSight starting...")

    # 1. Anchor builder (WordNet + ID seeds) untuk NLP.
    log.info("Building anchors (WordNet + OMW)...")
    ANCHORS = build_anchors()

    # 2. NLP engine.
    nlp = LangSightNLP(ST_MODEL_NAME, anchors=ANCHORS)

    # 3. Grounding DINO (selalu di-load karena dipakai juga oleh /detect/all).
    detector = GroundingDINODetector(GDINO_MODEL_ID)

    # 4. Warm-up DINO untuk hindari first-call latency spike.
    log.info("Warm-up DINO...")
    try:
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        detector.detect(dummy, "pen", 0.01, 0.01)
        log.info("Warm-up DINO selesai.")
    except Exception as e:
        log.warning(f"Warm-up DINO gagal (diabaikan): {e}")

    # 5. YOLO (opsional - skip kalau file model tidak ada).
    try:
        yolo_det = YOLODetector(YOLO_MODEL_PATH)
        log.info("Warm-up YOLO...")
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        yolo_det.detect(dummy, conf=0.01)
        log.info("Warm-up YOLO selesai.")
    except FileNotFoundError as e:
        log.warning(f"YOLO tidak tersedia: {e}")
        log.warning("/detect akan fallback ke DINO-only (tanpa ensemble).")
        yolo_det = None
    except Exception as e:
        log.error(f"Gagal load YOLO: {e}")
        yolo_det = None

    # 6. Bangun ensemble kalau YOLO tersedia.
    if yolo_det is not None:
        ENSEMBLE_WEIGHTS = _load_ensemble_weights(ENSEMBLE_WEIGHTS_PATH)
        try:
            ensemble = WBFEnsemble(yolo_det, detector, ENSEMBLE_WEIGHTS, wbf_iou=WBF_IOU_THR)
            log.info("Ensemble siap.")
        except Exception as e:
            log.error(f"Gagal build ensemble: {e}")
            ensemble = None
    else:
        ensemble = None
        log.info("Ensemble tidak dibuat (YOLO tidak tersedia).")

    log.info("-" * 55)
    log.info(f"  LangSight READY  http://localhost:{PORT}")
    log.info(f"  detect mode    : {'ENSEMBLE (YOLO+DINO)' if ensemble else 'DINO only'}")
    log.info(f"  detect/all mode: DINO only (by design)")
    log.info("-" * 55)
    yield
    log.info("Shutdown.")



# FASTAPI APP

app = FastAPI(
    title="LangSight API",
    description="Stationery Detection - Ensemble YOLOv11n + Grounding DINO",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# Favicon inline SVG supaya tidak perlu file fisik.
FAVICON_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
    '<rect width="32" height="32" rx="8" fill="#FFE66D"/>'
    '<text x="16" y="22" text-anchor="middle" font-size="18">LS</text>'
    '</svg>'
)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(content=FAVICON_SVG, media_type="image/svg+xml")


@app.get("/", include_in_schema=False)
async def root():
    """Serve frontend SPA kalau static/index.html ada."""
    p = Path("static/index.html")
    if p.exists():
        return FileResponse(p, media_type="text/html")
    return JSONResponse({"service": "LangSight API"})



# INFO & STATUS ENDPOINTS

@app.get("/health")
async def health():
    """Heartbeat & konfigurasi runtime. Dipakai frontend untuk cek konektivitas."""
    return {
        "status"               : "ok",
        "model_id"             : GDINO_MODEL_ID,
        "model_loaded"         : detector is not None,
        "yolo_loaded"          : yolo_det is not None,
        "ensemble_active"      : ensemble is not None,
        "detect_mode"          : "ensemble" if ensemble is not None else "dino_only",
        "detect_all_mode"      : "dino_only",
        "device"               : detector.device if detector else "unknown",
        "nlp_mode"             : "semantic" if (nlp and nlp._use_semantic) else "keyword",
        "classes"              : TARGET_CLASSES,
        "n_classes"            : NC,
        "yolo_eval_classes"    : list(YOLO_TO_TARGET.values()),
        "dino_only_classes"    : [c for c in TARGET_CLASSES if c not in YOLO_TO_TARGET.values()],
        "box_threshold"        : BOX_THRESHOLD,
        "text_threshold"       : TEXT_THRESHOLD,
        "yolo_conf_thr"        : YOLO_CONF_THR,
        "similarity_threshold" : SIMILARITY_THRESHOLD,
        "similarity_gap"       : SIMILARITY_GAP,
    }


@app.get("/classes")
async def get_classes():
    """List 9 target class plus konfigurasi NLP/DINO terkait."""
    return {
        "classes"         : TARGET_CLASSES,
        "class_prompts"   : CLASS_PROMPTS,
        "class_thresholds": CLASS_THRESHOLDS,
        "anchors"         : ANCHORS,
    }


@app.get("/classes/weights")
async def get_ensemble_weights():
    """Inspect bobot ensemble dari ensemble_weights.json. Frontend pakai ini
    untuk render tabel YOLO-vs-DINO di tab Admin."""
    if ensemble is None:
        return JSONResponse({
            "ensemble_active": False,
            "message": "Ensemble tidak aktif (YOLO tidak tersedia). /detect pakai DINO only.",
            "weights": None,
        })
    return JSONResponse({
        "ensemble_active"  : True,
        "meta"             : ensemble.meta,
        "weights"          : ensemble.weights,
        "yolo_classes"     : sorted(list(ensemble.yolo_classes)),
        "dino_only_classes": sorted(list(ensemble.dino_only_classes)),
        "wbf_iou"          : ensemble.wbf_iou,
    })


@app.get("/stats/classes")
async def class_stats():
    """Histogram confidence per kelas selama session berjalan."""
    stats = {}
    for cls, confs in CLASS_CONF_HISTORY.items():
        if confs:
            stats[cls] = {
                "n_detected": len(confs),
                "avg_conf"  : round(sum(confs) / len(confs), 4),
                "max_conf"  : round(max(confs), 4),
                "min_conf"  : round(min(confs), 4),
            }
    return {"class_stats": stats}



# NLP TEST ENDPOINT

@app.post("/nlp/test")
async def test_nlp(query: str = Form(...)):
    """Uji NLP saja: parse query -> target_class + ranked scores. Tidak inference visi."""
    if nlp is None:
        raise HTTPException(503, "NLP belum siap")
    if not query.strip():
        raise HTTPException(422, "Query kosong")

    sub_queries = nlp.parse_multi(query)
    results = []
    for sq in sub_queries:
        tc, method, ranked = nlp.predict(sq)
        prompt = CLASS_PROMPTS.get(tc, "") if tc else ""
        results.append({
            "sub_query"   : sq,
            "target_class": tc,
            "method"      : method,
            "gdino_prompt": prompt,
            "recognized"  : tc is not None,
            "top_scores"  : [{"class": c, "score": round(s, 4)} for c, s in ranked],
        })
    return JSONResponse({
        "query"      : query,
        "is_multi"   : len(sub_queries) > 1,
        "sub_results": results,
        "threshold"  : SIMILARITY_THRESHOLD,
        "gap"        : SIMILARITY_GAP,
    })



# DETECTION ENDPOINTS

@app.post("/detect")
async def detect(
    file          : UploadFile = File(...),
    query         : str        = Form(...),
    box_threshold : float      = Form(BOX_THRESHOLD),
    text_threshold: float      = Form(TEXT_THRESHOLD),
    annotate      : bool       = Form(True),
):
    """Endpoint utama: NLP -> target_class -> ensemble (YOLO + DINO) -> bbox.

    Fallback ke DINO-only kalau ensemble belum aktif (mis. file YOLO tidak ada).
    Mendukung multi-object query ("pulpen dan pensil") via parse_multi.
    """
    t0 = time.perf_counter()
    if not query.strip():
        raise HTTPException(422, "Query kosong")
    if nlp is None or detector is None:
        raise HTTPException(503, "Model belum siap")

    try:
        raw     = await file.read()
        img_bgr = read_image(raw)
    except Exception as e:
        raise HTTPException(400, f"Gambar tidak valid: {e}")

    mode = "ensemble" if ensemble is not None else "dino_only"

    sub_queries  = nlp.parse_multi(query)
    is_multi     = len(sub_queries) > 1
    all_dets     = []
    all_ranked   = []
    target_class = None
    method       = "semantic"
    top_score    = 0.0

    for sq in sub_queries:
        tc, meth, ranked = nlp.predict(sq)
        if not all_ranked:
            all_ranked = ranked
        if tc:
            target_class = tc
            method       = meth
            top_score    = ranked[0][1] if ranked else 0.0

            # Inference: pakai ensemble kalau aktif, kalau tidak fallback DINO.
            if ensemble is not None:
                dets = ensemble.detect(
                    img_bgr, tc,
                    yolo_conf=YOLO_CONF_THR,
                    dino_box_thr=box_threshold,
                    dino_text_thr=text_threshold,
                )
            else:
                dets = detector.detect(img_bgr, tc, box_threshold, text_threshold)
                for d in dets:
                    d.setdefault("source", "dino")

            all_dets.extend(dets)
            for d in dets:
                CLASS_CONF_HISTORY[tc].append(d["confidence"])
                if len(CLASS_CONF_HISTORY[tc]) > 200:
                    CLASS_CONF_HISTORY[tc] = CLASS_CONF_HISTORY[tc][-100:]

    # NMS cross-class kalau ada multiple sub-queries (cegah duplikat di area sama).
    if is_multi:
        all_dets = _nms(all_dets, 0.5)[:10]

    FRAME_HISTORY.append({"n_det": len(all_dets), "ts": time.time()})

    annotated_b64 = None
    if annotate:
        ann = draw_detections(img_bgr, all_dets)
        annotated_b64 = img_to_b64(ann)

    elapsed = round((time.perf_counter() - t0) * 1000, 1)

    # Susun pesan ringkas untuk frontend (breakdown sumber + label kelas).
    if target_class is None:
        msg = "Query tidak dikenali."
    elif all_dets:
        src_counts = defaultdict(int)
        for d in all_dets:
            src_counts[d.get("source", "unknown")] += 1
        src_summary = ", ".join(f"{n} {s}" for s, n in src_counts.items())
        cls_names   = ", ".join(set(d["class"] for d in all_dets))
        msg = (f"Ditemukan {len(all_dets)} objek ({cls_names}) "
               f"[{src_summary}, sim={top_score:.2f}]")
    else:
        msg = f"'{target_class}' tidak terdeteksi. Coba turunkan threshold."

    log.info(f"[{elapsed}ms] mode={mode} '{query}' -> '{target_class}' | {len(all_dets)} det")
    log_session(query, target_class, method, len(all_dets), elapsed, top_score)

    return JSONResponse({
        "success"        : target_class is not None,
        "mode"           : mode,
        "query"          : query,
        "is_multi"       : is_multi,
        "target_class"   : target_class,
        "method"         : method,
        "similarity"     : round(top_score, 4),
        "top_scores"     : [{"class": c, "score": round(s, 4)} for c, s in all_ranked],
        "detections"     : all_dets,
        "message"        : msg,
        "annotated_image": annotated_b64,
        "elapsed_ms"     : elapsed,
    })


@app.post("/detect/all")
async def detect_all(
    file          : UploadFile = File(...),
    box_threshold : float      = Form(BOX_THRESHOLD),
    text_threshold: float      = Form(TEXT_THRESHOLD),
    annotate      : bool       = Form(True),
):
    """Scan SEMUA 9 kelas memakai Grounding DINO saja. Tidak pakai ensemble
    karena YOLO hanya mengcover 6 dari 9 kelas (sengaja didesain begitu)."""
    t0 = time.perf_counter()
    if detector is None:
        raise HTTPException(503, "Model belum siap")
    try:
        raw     = await file.read()
        img_bgr = read_image(raw)
    except Exception as e:
        raise HTTPException(400, f"Gambar tidak valid: {e}")

    dets = detector.detect_all(img_bgr, box_threshold, text_threshold)
    for d in dets:
        d.setdefault("source", "dino")

    FRAME_HISTORY.append({"n_det": len(dets), "ts": time.time()})

    annotated_b64 = None
    if annotate:
        ann = draw_detections(img_bgr, dets)
        annotated_b64 = img_to_b64(ann)

    elapsed = round((time.perf_counter() - t0) * 1000, 1)
    log.info(f"[detect/all] mode=dino_only {len(dets)} det | {elapsed}ms")
    log_session("SCAN_ALL", "all", "scan", len(dets), elapsed)

    return JSONResponse({
        "success"        : True,
        "mode"           : "dino_only",
        "detections"     : dets,
        "message"        : f"Ditemukan {len(dets)} objek dari seluruh kelas (DINO).",
        "annotated_image": annotated_b64,
        "elapsed_ms"     : elapsed,
    })


@app.post("/detect/batch")
async def detect_batch(
    files         : list[UploadFile] = File(...),
    query         : str              = Form(...),
    box_threshold : float            = Form(BOX_THRESHOLD),
    text_threshold: float            = Form(TEXT_THRESHOLD),
):
    """Batch detect — apply query yang sama ke beberapa gambar sekaligus."""
    if len(files) > 10:
        raise HTTPException(422, "Max 10 gambar")
    if nlp is None or detector is None:
        raise HTTPException(503, "Model belum siap")

    target_class, method, _ = nlp.predict(query)
    mode = "ensemble" if ensemble is not None else "dino_only"
    results = []

    for f in files:
        try:
            raw     = await f.read()
            img_bgr = read_image(raw)
            if not target_class:
                dets = []
            elif ensemble is not None:
                dets = ensemble.detect(
                    img_bgr, target_class,
                    yolo_conf=YOLO_CONF_THR,
                    dino_box_thr=box_threshold,
                    dino_text_thr=text_threshold,
                )
            else:
                dets = detector.detect(img_bgr, target_class, box_threshold, text_threshold)
                for d in dets:
                    d.setdefault("source", "dino")
        except Exception as e:
            log.warning(f"Batch error {f.filename}: {e}")
            dets = []
        results.append({
            "filename"    : f.filename,
            "target_class": target_class,
            "detections"  : dets,
            "found"       : len(dets) > 0,
        })

    return JSONResponse({
        "query"      : query,
        "mode"       : mode,
        "results"    : results,
        "method"     : method,
        "total_found": sum(1 for r in results if r["found"]),
    })



# ADMIN ENDPOINTS — runtime configuration tanpa restart

@app.post("/admin/prompt")
async def update_prompt(cls: str = Form(...), prompt: str = Form(...)):
    """Edit prompt Grounding DINO untuk satu kelas tanpa perlu restart server.
    Dipakai frontend tab Admin untuk fine-tuning prompt secara live."""
    if cls not in TARGET_CLASSES:
        raise HTTPException(400, f"Kelas '{cls}' tidak dikenal")
    if not prompt.strip():
        raise HTTPException(422, "Prompt kosong")
    CLASS_PROMPTS[cls] = prompt.strip()
    log.info(f"Prompt updated: [{cls}] -> {prompt[:60]}")
    return JSONResponse({"ok": True, "class": cls, "new_prompt": CLASS_PROMPTS[cls]})


@app.post("/admin/model")
async def switch_model(model_id: str = Form(...)):
    """Switch antara grounding-dino-tiny dan grounding-dino-base tanpa restart."""
    global detector, GDINO_MODEL_ID
    allowed = [
        "IDEA-Research/grounding-dino-tiny",
        "IDEA-Research/grounding-dino-base",
    ]
    if model_id not in allowed:
        raise HTTPException(400, f"Model tidak diizinkan. Pilihan: {allowed}")
    if detector and detector.model_id == model_id:
        return JSONResponse({"ok": True, "message": "Model sudah aktif", "model_id": model_id})

    log.info(f"Switching model -> {model_id}")
    try:
        detector = GroundingDINODetector(model_id)
        GDINO_MODEL_ID = model_id
        return JSONResponse({"ok": True, "model_id": model_id})
    except Exception as e:
        raise HTTPException(500, f"Gagal load model: {e}")


@app.post("/admin/calibrate")
async def update_calibration(
    cls: str   = Form(...),
    a  : float = Form(-1.0),
    b  : float = Form(0.0),
):
    """Update parameter Platt scaling untuk satu kelas.
    Formula: calibrated = 1 / (1 + exp(a * raw_score + b))
    Default a=-1, b=0 -> passthrough (tidak ada kalibrasi).
    Turunkan |a| atau turunkan b untuk shift skor ke atas."""
    if cls not in TARGET_CLASSES:
        raise HTTPException(400, f"Kelas '{cls}' tidak dikenal")
    _CALIBRATION[cls] = {"a": a, "b": b}
    log.info(f"Calibration updated: [{cls}] a={a}, b={b}")

    # Preview konversi untuk beberapa raw score sebagai sanity check ke user.
    preview = {}
    for s in [0.20, 0.35, 0.50, 0.65, 0.80]:
        try:
            cal = 1.0 / (1.0 + math.exp(a * s + b))
        except OverflowError:
            cal = 0.0
        preview[str(s)] = round(cal, 4)

    return JSONResponse({
        "ok": True, "class": cls, "a": a, "b": b,
        "calibration_preview": preview,
    })


@app.get("/admin/calibration")
async def get_calibration():
    """List semua kalibrasi yang aktif."""
    return JSONResponse({"calibrations": _CALIBRATION})



# ACTIVE LEARNING — kumpulkan feedback user untuk iterasi berikutnya

_AL_DIR = MODEL_DIR / "active_learning"
_AL_DIR.mkdir(parents=True, exist_ok=True)

#A6 (utk optimasi pake anchor embedding)
ANCHOR_OVERRIDE_PATH = MODEL_DIR / "anchor_overrides.json"

def load_anchor_overrides():
    if not ANCHOR_OVERRIDE_PATH.exists():
        return {}
    with open(ANCHOR_OVERRIDE_PATH) as f:
        return json.load(f)

def save_anchor_override(cls: str, new_terms: list):
    overrides = load_anchor_overrides()
    overrides[cls] = new_terms
    with open(ANCHOR_OVERRIDE_PATH, "w") as f:
        json.dump(overrides, f, ensure_ascii=False, indent=2)

@app.post("/feedback")
async def submit_feedback(
    query        : str        = Form(...),
    target_class : str        = Form(...),
    is_correct   : bool       = Form(...),
    feedback_note: str        = Form(""),
    file         : UploadFile = File(None),
):
    """Terima feedback user (benar/salah) untuk deteksi tertentu.
    - is_correct=False -> simpan sebagai hard negative + tambahkan query
      ke anchor seeds kelas tersebut (NLP auto-improve).
    - is_correct=True  -> simpan sebagai konfirmasi.
    Gambar opsional disimpan ke disk untuk analisis manual nanti.
    """
    ts = time.strftime("%Y%m%d_%H%M%S")
    entry = {
        "timestamp"    : ts,
        "query"        : query,
        "target_class" : target_class,
        "is_correct"   : is_correct,
        "feedback_note": feedback_note,
    }

    # Simpan gambar kalau di-attach.
    if file:
        try:
            img_bytes = await file.read()
            img_name  = f"{ts}_{target_class}_{'pos' if is_correct else 'neg'}.jpg"
            img_path  = _AL_DIR / img_name
            with open(img_path, "wb") as f:
                f.write(img_bytes)
            entry["image_path"] = str(img_path)
        except Exception as e:
            log.warning(f"Feedback image save error: {e}")

    # Append JSONL log.
    log_path = _AL_DIR / "feedback_log.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")

    # Active learning: kalau false negative, tambahkan query ke anchor kelas
    # tersebut supaya next time NLP lebih kenal.
    if not is_correct and nlp and query.strip():
        existing = nlp._active.get(target_class, [])
        if query.lower() not in [t.lower() for t in existing]:
            existing.append(query.lower())
            nlp._active[target_class] = existing
            try:
                new_emb = nlp._st.encode(
                    existing, convert_to_tensor=True, normalize_embeddings=True,
                )
                nlp._embs[target_class] = new_emb
                log.info(f"Active learning: anchor '{query}' ditambah ke [{target_class}]")
                save_anchor_override(target_class, existing)
            except Exception as e:
                log.warning(f"Active learning re-encode error: {e}")

    mark = "correct" if is_correct else "incorrect"
    log.info(f"Feedback: [{target_class}] {mark} | '{query}'")
    return JSONResponse({
        "ok"            : True,
        "class"         : target_class,
        "is_correct"    : is_correct,
        "anchor_updated": not is_correct,
        "message": "Terima kasih, feedback disimpan." + (
            f" Anchor '{query}' ditambah ke kelas {target_class}." if not is_correct else ""
        ),
    })


@app.get("/feedback/summary")
async def feedback_summary():
    """Ringkasan feedback yang sudah masuk (total, per kelas, dan recent)."""
    log_path = _AL_DIR / "feedback_log.jsonl"
    if not log_path.exists():
        return JSONResponse({"total": 0, "entries": []})

    entries = []
    with open(log_path) as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except Exception:
                pass

    correct   = sum(1 for e in entries if e.get("is_correct"))
    incorrect = len(entries) - correct
    by_class  = defaultdict(lambda: {"correct": 0, "incorrect": 0})
    for e in entries:
        cls = e.get("target_class", "unknown")
        if e.get("is_correct"):
            by_class[cls]["correct"] += 1
        else:
            by_class[cls]["incorrect"] += 1

    return JSONResponse({
        "total"    : len(entries),
        "correct"  : correct,
        "incorrect": incorrect,
        "by_class" : dict(by_class),
        "recent"   : entries[-10:],
    })



# SESSION EXPORT

@app.get("/session/export")
async def export_session():
    """Export riwayat query + statistik kelas sebagai JSON (untuk debug/demo)."""
    return JSONResponse({
        "exported_at"  : time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_queries": len(SESSION_LOG),
        "session_log"  : SESSION_LOG,
        "class_stats"  : {
            cls: {
                "n_detected": len(confs),
                "avg_conf"  : round(sum(confs) / len(confs), 4),
            }
            for cls, confs in CLASS_CONF_HISTORY.items() if confs
        },
    })



# ENTRY POINT

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=HOST, port=PORT, reload=False, log_level="info")