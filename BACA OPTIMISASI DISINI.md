# LANGSIGHT — PANDUAN OPTIMISASI LENGKAP
# "Sampe Mentok" Edition
# Semua layer optimisasi dari yang paling mudah sampai paling dalam
# ─────────────────────────────────────────────────────────────────


════════════════════════════════════════════════════════════
OVERVIEW: 5 DOMAIN OPTIMISASI
════════════════════════════════════════════════════════════

  A. PROMPT & NLP        → Ketajaman pemahaman bahasa
  B. DINO INFERENCE      → Ketajaman deteksi visual
  C. ENSEMBLE & FUSION   → Ketajaman penggabungan model
  D. POST-PROCESSING     → Ketajaman output akhir
  E. EVALUASI & FEEDBACK → Ketajaman pengukuran & belajar

Total optimisasi: 24 item
Estimasi dampak per item tersedia di setiap section


════════════════════════════════════════════════════════════
A. PROMPT & NLP
════════════════════════════════════════════════════════════

────────────────────────────────────────────────────────────
A1. Attribute-Based Prompt
Effort: RENDAH | Dampak: TINGGI | Prioritas: 1
────────────────────────────────────────────────────────────

MASALAH:
Prompt sekarang hanya berisi nama objek. DINO bekerja jauh
lebih baik kalau prompt mendeskripsikan tampilan fisik objek.

SEBELUM:
  "clip": "paper clip . binder clip . metal paperclip ."

SESUDAH:
  "clip": (
      "small metal paper clip . tiny silver wire clip . "
      "flat metallic paperclip . small binder clip . "
      "thin metal document fastener ."
  )
  "sharpener": (
      "pencil sharpener . small plastic sharpener . "
      "cylindrical pencil sharpener . handheld sharpener . "
      "small manual pencil sharpening tool ."
  )
  "pencil": (
      "graphite pencil . wooden pencil . long thin pencil . "
      "yellow pencil . hexagonal wooden pencil . "
      "lead pencil . drafting pencil ."
  )
  "correction_tape": (
      "correction tape . white correction roller . "
      "small white rectangular correction tape dispenser . "
      "white out tape . tipex roller . correction strip ."
  )

KELAS PRIORITAS UNTUK DIUPDATE:
  clip (F1=0.552), sharpener (F1=0.590), pencil (F1=0.867 bisa naik lagi)

CARA IMPLEMENTASI:
  Edit CLASS_PROMPTS di app.py dan semua notebook.
  Tidak butuh kode baru, cukup edit string.


────────────────────────────────────────────────────────────
A2. Prompt Ensembling (Multi-Prompt per Kelas)
Effort: MEDIUM | Dampak: TINGGI | Prioritas: 4
────────────────────────────────────────────────────────────

MASALAH:
Satu prompt per kelas artinya satu "sudut pandang" DINO.
Kalau prompt tidak cocok dengan kondisi gambar, missed.

SOLUSI:
Jalankan 2-3 variasi prompt, union hasilnya sebelum NMS.

IMPLEMENTASI:
  PROMPT_VARIANTS = {
      "clip": [
          "paper clip . binder clip . metal paperclip .",
          "small flat metal clip . silver wire fastener .",
          "tiny metallic document clip . small office clip .",
      ],
      "sharpener": [
          "pencil sharpener . handheld sharpener .",
          "small plastic sharpener . cylindrical pencil sharpener .",
          "manual pencil sharpening tool . small desk sharpener .",
      ],
  }

  def detect_multi_prompt(self, img_bgr, target_class, ...):
      variants = PROMPT_VARIANTS.get(target_class)
      if not variants:
          return self.detect(img_bgr, target_class, ...)  # default

      all_dets = []
      for prompt_variant in variants:
          CLASS_PROMPTS[target_class] = prompt_variant
          dets = self.detect(img_bgr, target_class, ...)
          all_dets.extend(dets)

      # Union + NMS untuk remove duplikat
      return _nms(all_dets, iou_thr=0.50)[:5]

CATATAN:
  Hanya terapkan untuk kelas DINO-only yang F1-nya rendah.
  Kelas dengan YOLO yang kuat tidak perlu ini.


────────────────────────────────────────────────────────────
A3. Query Expansion via Attribute Extraction
Effort: MEDIUM | Dampak: MEDIUM | Prioritas: 6
────────────────────────────────────────────────────────────

MASALAH:
"pulpen merah" → sistem match ke "pen" tapi kata "merah" dibuang.
Padahal bisa dipakai untuk mempertajam prompt DINO sementara.

IMPLEMENTASI:
  # Daftar atribut yang dikenali
  QUERY_ATTRIBUTES = {
      "colors" : ["merah","biru","hitam","hijau","kuning","putih",
                  "red","blue","black","green","yellow","white"],
      "sizes"  : ["kecil","besar","panjang","pendek",
                  "small","big","long","short"],
      "materials": ["plastik","besi","kayu","karet",
                    "plastic","metal","wooden","rubber"],
  }

  def extract_attributes(query: str) -> dict:
      found = {"colors": [], "sizes": [], "materials": []}
      q = query.lower()
      for attr_type, keywords in QUERY_ATTRIBUTES.items():
          for kw in keywords:
              if kw in q:
                  found[attr_type].append(kw)
      return found

  def build_enriched_prompt(base_prompt: str, attrs: dict) -> str:
      extras = []
      for color in attrs.get("colors", []):
          extras.append(f"{color} {base_prompt.split('.')[0].strip()}")
      if extras:
          return base_prompt + " . " + " . ".join(extras) + " ."
      return base_prompt

  # Di endpoint /detect, sebelum DINO inference:
  attrs = extract_attributes(query)
  if any(attrs.values()):
      enriched = build_enriched_prompt(CLASS_PROMPTS[target_class], attrs)
      # Pakai enriched prompt hanya untuk request ini (tidak mengubah global)
      dets = dino.detect(img, target_class, prompt_override=enriched)


────────────────────────────────────────────────────────────
A4. Negation Handling
Effort: RENDAH | Dampak: MEDIUM | Prioritas: 7
────────────────────────────────────────────────────────────

MASALAH:
"bukan pensil" atau "selain pulpen" sekarang tetap match
ke kelas tersebut karena kata dominan tetap terdeteksi.

IMPLEMENTASI:
  NEGATION_WORDS = [
      "bukan", "bukan yang", "selain", "kecuali",
      "not", "except", "other than", "not the"
  ]

  def detect_negation(query: str) -> tuple[bool, str]:
      """Return (is_negated, cleaned_query)"""
      q = query.lower()
      for neg in NEGATION_WORDS:
          if neg in q:
              cleaned = q.replace(neg, "").strip()
              return True, cleaned
      return False, query

  # Di predict():
  is_negated, clean_q = detect_negation(query)
  target_class, method, ranked = self._predict_internal(clean_q)
  if is_negated:
      # Hapus target_class dari hasil, ambil runner-up
      ranked = [(c, s) for c, s in ranked if c != target_class]
      target_class = ranked[0][0] if ranked else None


────────────────────────────────────────────────────────────
A5. Typo Tolerance
Effort: RENDAH | Dampak: MEDIUM | Prioritas: 8
────────────────────────────────────────────────────────────

MASALAH:
"steapler", "penghapuss", "tipecs" tidak ada di kamus
dan bisa gagal match kalau tidak ada di _ID_SEEDS.

IMPLEMENTASI:
  from difflib import get_close_matches

  def fuzzy_keyword_predict(self, query: str):
      """Keyword match dengan toleransi typo via edit distance."""
      q_words = query.lower().split()
      for cls, terms in self._active.items():
          for term in terms:
              term_words = term.lower().split()
              for qw in q_words:
                  matches = get_close_matches(qw, term_words,
                                              n=1, cutoff=0.75)
                  if matches:
                      return cls, "fuzzy_keyword", [(cls, 0.85)]
      return None, "fuzzy_keyword", []

  # Di predict(), tambahkan sebagai fallback ke-3:
  # semantic → keyword → fuzzy_keyword


────────────────────────────────────────────────────────────
A6. Anchor Embedding Persistent (Feedback → File)
Effort: RENDAH | Dampak: MEDIUM | Prioritas: 5
────────────────────────────────────────────────────────────

MASALAH:
Anchor yang diperkaya via feedback hilang saat server restart.

IMPLEMENTASI:
  ANCHOR_OVERRIDE_PATH = MODEL_DIR / "anchor_overrides.json"

  def load_anchor_overrides():
      """Load anchor tambahan dari file kalau ada."""
      if not ANCHOR_OVERRIDE_PATH.exists():
          return {}
      with open(ANCHOR_OVERRIDE_PATH) as f:
          return json.load(f)

  def save_anchor_override(cls: str, new_terms: list):
      """Simpan anchor baru ke file supaya persist."""
      overrides = load_anchor_overrides()
      overrides[cls] = new_terms
      with open(ANCHOR_OVERRIDE_PATH, "w") as f:
          json.dump(overrides, f, ensure_ascii=False, indent=2)

  # Di /feedback endpoint, setelah update anchor:
  save_anchor_override(target_class, existing)


════════════════════════════════════════════════════════════
B. DINO INFERENCE
════════════════════════════════════════════════════════════

────────────────────────────────────────────────────────────
B1. Adaptive Preprocessing (Brightness & Contrast)
Effort: RENDAH | Dampak: TINGGI | Prioritas: 2
────────────────────────────────────────────────────────────

MASALAH:
Performa DINO turun drastis di kondisi pencahayaan buruk.
Tidak ada preprocessing sekarang.

IMPLEMENTASI:
  def preprocess_adaptive(img_bgr: np.ndarray) -> np.ndarray:
      gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
      mean_brightness = gray.mean()
      std_brightness  = gray.std()

      if mean_brightness < 60:
          # Terlalu gelap — CLAHE untuk enhance contrast lokal
          clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
          lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
          lab[:, :, 0] = clahe.apply(lab[:, :, 0])
          return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

      elif mean_brightness > 200:
          # Terlalu terang — reduce exposure
          return cv2.convertScaleAbs(img_bgr, alpha=0.7, beta=0)

      elif std_brightness < 30:
          # Kontras sangat rendah (gambar flat) — histogram equalization
          lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
          lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
          return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

      return img_bgr  # sudah bagus

  # Di GroundingDINODetector.detect(), tambahkan sebelum inference:
  img_bgr = preprocess_adaptive(img_bgr)


────────────────────────────────────────────────────────────
B2. Two-Stage Detection untuk Objek Kecil
Effort: MEDIUM | Dampak: TINGGI | Prioritas: 3
────────────────────────────────────────────────────────────

MASALAH:
Clip dan sharpener sering missed karena terlalu kecil di frame.
DINO di gambar penuh resolusinya tidak cukup untuk objek kecil.

IMPLEMENTASI:
  def detect_two_stage(self, img_bgr, target_class, ...):
      h, w = img_bgr.shape[:2]

      # Stage 1: DINO di gambar penuh dengan threshold rendah
      rough = self.detect(img_bgr, target_class,
                          box_threshold=0.15, text_threshold=0.10)

      if not rough:
          return []

      refined_all = []
      for det in rough:
          x1, y1, x2, y2 = det["bbox"]

          # Expand bbox 40% ke semua sisi
          pw = int((x2 - x1) * 0.4)
          ph = int((y2 - y1) * 0.4)
          cx1 = max(0, x1 - pw)
          cy1 = max(0, y1 - ph)
          cx2 = min(w, x2 + pw)
          cy2 = min(h, y2 + ph)

          crop = img_bgr[cy1:cy2, cx1:cx2]
          if crop.size == 0:
              continue

          # Stage 2: DINO di crop yang sudah diperbesar
          crop_dets = self.detect(crop, target_class,
                                  box_threshold=0.35)

          # Scale balik koordinat ke gambar asli
          for cd in crop_dets:
              bx1, by1, bx2, by2 = cd["bbox"]
              cd["bbox"] = [bx1+cx1, by1+cy1, bx2+cx1, by2+cy1]
              cd["two_stage"] = True
              refined_all.append(cd)

      return _nms(refined_all, iou_thr=0.50)[:5] if refined_all else rough

  # Hanya aktifkan untuk kelas DINO-only yang kecil:
  TWO_STAGE_CLASSES = {"clip", "sharpener"}

  # Di detect():
  if target_class in TWO_STAGE_CLASSES:
      return self.detect_two_stage(img_bgr, target_class, ...)


────────────────────────────────────────────────────────────
B3. Multi-Scale Inference
Effort: MEDIUM | Dampak: MEDIUM | Prioritas: 9
────────────────────────────────────────────────────────────

MASALAH:
DINO dijalankan di satu resolusi. Objek kecil lebih baik
dideteksi di scale besar, objek besar di scale kecil.

IMPLEMENTASI:
  MULTISCALE_CLASSES = {"clip", "sharpener", "pencil"}
  INFERENCE_SCALES   = [0.75, 1.0, 1.5]

  def detect_multiscale(self, img_bgr, target_class, ...):
      h, w = img_bgr.shape[:2]
      all_dets = []

      for scale in INFERENCE_SCALES:
          nw, nh = int(w * scale), int(h * scale)
          resized = cv2.resize(img_bgr, (nw, nh))
          dets = self.detect(resized, target_class, ...)
          # Scale balik koordinat
          for d in dets:
              d["bbox"] = [
                  int(d["bbox"][0] / scale),
                  int(d["bbox"][1] / scale),
                  int(d["bbox"][2] / scale),
                  int(d["bbox"][3] / scale),
              ]
          all_dets.extend(dets)

      return _nms(all_dets, iou_thr=0.50)[:5]

  # Di detect(), hanya untuk kelas yang butuh:
  if target_class in MULTISCALE_CLASSES:
      return self.detect_multiscale(img_bgr, target_class, ...)


────────────────────────────────────────────────────────────
B4. Confidence Calibration via Platt Scaling
Effort: MEDIUM | Dampak: MEDIUM | Prioritas: 10
────────────────────────────────────────────────────────────

MASALAH:
Skor DINO keluar mentah dari model. Threshold 0.35 yang sekarang
arbitrary — tidak ada basis probabilistik.

CARA HITUNG DARI NOTEBOOK:
  from sklearn.linear_model import LogisticRegression
  from sklearn.calibration import calibration_curve
  import numpy as np

  # raw_scores = semua skor DINO di evaluation set untuk kelas X
  # labels     = 1 kalau TP (IoU >= threshold), 0 kalau FP
  raw_scores = [...]
  labels     = [...]

  lr = LogisticRegression(C=1.0)
  lr.fit(np.array(raw_scores).reshape(-1, 1), labels)

  a = float(lr.coef_[0][0])
  b = float(lr.intercept_[0])
  # calibrated = 1 / (1 + exp(a * raw + b))

  print(f"Kelas X: a={a:.4f}, b={b:.4f}")

CARA APPLY KE SISTEM:
  # Via endpoint yang sudah ada:
  POST /admin/calibrate
  { "cls": "clip", "a": -2.5, "b": 0.8 }

  # Atau hardcode di config setelah dihitung:
  _CALIBRATION = {
      "clip"      : {"a": -2.5, "b": 0.8},
      "sharpener" : {"a": -2.1, "b": 0.6},
      "eraser"    : {"a": -1.8, "b": 0.3},
  }


════════════════════════════════════════════════════════════
C. ENSEMBLE & FUSION
════════════════════════════════════════════════════════════

────────────────────────────────────────────────────────────
C1. Lazy DINO — Skip DINO Kalau YOLO Sudah Confident
Effort: RENDAH | Dampak: SANGAT TINGGI | Prioritas: 1
────────────────────────────────────────────────────────────

MASALAH:
DINO selalu jalan ~500ms meskipun YOLO sudah sangat yakin.
Ini juga penyebab correction_tape turun di ensemble.

IMPLEMENTASI:
  # Di WBFEnsemble.detect():
  def detect(self, img_bgr, target_class, ...):
      w_yolo, w_dino = self._weights_for(target_class)

      # Jalankan YOLO dulu
      yolo_dets = self.yolo.detect(img_bgr, [target_class], ...)
      best_yolo_conf = max(
          (d["confidence"] for d in yolo_dets), default=0.0
      )

      # Kalau YOLO sudah sangat dominan DAN confident:
      # skip DINO sepenuhnya
      LAZY_THRESHOLD_W    = 0.80  # bobot YOLO minimal
      LAZY_THRESHOLD_CONF = 0.75  # confidence YOLO minimal

      if (w_yolo >= LAZY_THRESHOLD_W
              and best_yolo_conf >= LAZY_THRESHOLD_CONF
              and target_class not in self.dino_only_classes):
          log.info(f"  Lazy DINO: skip [{target_class}] "
                   f"(w_yolo={w_yolo}, conf={best_yolo_conf:.2f})")
          return temporal_smooth(target_class, yolo_dets)

      # Kalau tidak, lanjut normal ke DINO + WBF
      dino_dets = self.dino.detect(...)
      return self._fuse(yolo_dets, dino_dets, w_yolo, w_dino, target_class)

DAMPAK GANDA:
  - correction_tape tidak turun lagi (tidak ada dilusi dari DINO)
  - Latency turun dari ~530ms ke ~30ms untuk kasus YOLO confident
  - Kelas yang terdampak: correction_tape (w_yolo=0.849)


────────────────────────────────────────────────────────────
C2. Per-Class WBF IoU Threshold
Effort: SANGAT RENDAH | Dampak: MEDIUM | Prioritas: 2
────────────────────────────────────────────────────────────

MASALAH:
Pen dan pencil bentuknya panjang tipis. IoU antara box YOLO dan
DINO untuk objek yang sama bisa di bawah 0.50 → tidak di-fuse
→ muncul sebagai dua deteksi terpisah.

IMPLEMENTASI:
  # Tambahkan di config (sudah ada sebagian di ADAPTIVE_IOU):
  WBF_IOU_PER_CLASS = {
      "pen"            : 0.30,
      "pencil"         : 0.30,
      "clip"           : 0.30,
      "correction_tape": 0.35,
      "stapler"        : 0.45,
      # notebook, bottle, eraser, sharpener → default 0.50
  }

  # Di WBFEnsemble._fuse(), ganti:
  # if _iou(items[i]["bbox"], items[j]["bbox"]) >= self.wbf_iou:
  # Jadi:
  iou_thr = WBF_IOU_PER_CLASS.get(target_class, self.wbf_iou)
  if _iou(items[i]["bbox"], items[j]["bbox"]) >= iou_thr:


────────────────────────────────────────────────────────────
C3. Dynamic Weight per Frame
Effort: MEDIUM | Dampak: MEDIUM | Prioritas: 8
────────────────────────────────────────────────────────────

MASALAH:
Bobot WBF statis dari evaluasi offline. Tapi kondisi tiap frame
berbeda — pencahayaan, sudut, oklusi.

IMPLEMENTASI:
  def dynamic_weights(yolo_score, dino_score,
                      base_w_yolo, base_w_dino,
                      delta=0.15):
      """
      Adjust bobot berdasarkan confidence relatif di frame ini.
      """
      if yolo_score > 0.80 and dino_score < 0.40:
          # YOLO jauh lebih yakin frame ini
          return (min(base_w_yolo + delta, 0.95),
                  max(base_w_dino - delta, 0.05))
      elif dino_score > 0.80 and yolo_score < 0.40:
          # DINO jauh lebih yakin frame ini
          return (max(base_w_yolo - delta, 0.05),
                  min(base_w_dino + delta, 0.95))
      return base_w_yolo, base_w_dino

  # Di _fuse(), sebelum kalkulasi ws:
  best_yolo = max((i["score"] for i in items if i["src"]=="yolo"),
                  default=0.0)
  best_dino = max((i["score"] for i in items if i["src"]=="dino"),
                  default=0.0)
  w_yolo, w_dino = dynamic_weights(best_yolo, best_dino,
                                    w_yolo, w_dino)


────────────────────────────────────────────────────────────
C4. NMS Threshold per Kelas — Lebih Presisi
Effort: SANGAT RENDAH | Dampak: LOW-MEDIUM | Prioritas: 3
────────────────────────────────────────────────────────────

MASALAH:
NMS sekarang tidak cover semua 9 kelas dengan threshold
yang tepat — clip bisa over-merged kalau ada banyak di frame.

IMPLEMENTASI:
  # Update _CLASS_NMS_IOU yang sudah ada:
  _CLASS_NMS_IOU = {
      "pen"            : 0.35,  # panjang tipis
      "pencil"         : 0.35,  # panjang tipis
      "clip"           : 0.60,  # kecil, bisa banyak berdekatan
      "correction_tape": 0.40,  # memanjang
      "stapler"        : 0.55,  # besar, jarang ada dua
      "notebook"       : 0.55,  # besar
      "bottle"         : 0.50,  # default
      "eraser"         : 0.50,  # default
      "sharpener"      : 0.50,  # default
  }


════════════════════════════════════════════════════════════
D. POST-PROCESSING & OUTPUT
════════════════════════════════════════════════════════════

────────────────────────────────────────────────────────────
D1. Re-ranking Post-Detection
Effort: RENDAH | Dampak: MEDIUM | Prioritas: 4
────────────────────────────────────────────────────────────

MASALAH:
Kalau ada 5 box untuk "pen", yang terpilih hanya berdasarkan
confidence score. Box di pojok frame yang jauh dari center
kurang relevan untuk user.

IMPLEMENTASI:
  def rerank_detections(dets: list, img_w: int, img_h: int,
                        target_class: str) -> list:
      if not dets:
          return dets

      scored = []
      for d in dets:
          x1, y1, x2, y2 = d["bbox"]
          cx = (x1 + x2) / 2
          cy = (y1 + y2) / 2
          bw = x2 - x1
          bh = y2 - y1

          # Seberapa dekat ke tengah frame (0.0–1.0)
          center_score = 1.0 - (
              abs(cx / img_w - 0.5) + abs(cy / img_h - 0.5)
          )

          # Seberapa "wajar" ukurannya (ideal ~5% area frame)
          area_ratio = (bw * bh) / (img_w * img_h)
          size_score = max(0.0, 1.0 - abs(area_ratio - 0.05) * 10)

          # Final rerank score
          final = (0.70 * d["confidence"]
                 + 0.20 * center_score
                 + 0.10 * size_score)

          scored.append((final, d))

      scored.sort(key=lambda x: -x[0])
      result = []
      for final_score, d in scored:
          d["rerank_score"] = round(final_score, 4)
          result.append(d)
      return result

  # Di detect() dan ensemble.detect(), sebelum return:
  h, w = img_bgr.shape[:2]
  dets = rerank_detections(dets, w, h, target_class)


────────────────────────────────────────────────────────────
D2. Temporal Smoothing per Kelas
Effort: RENDAH | Dampak: MEDIUM | Prioritas: 5
────────────────────────────────────────────────────────────

MASALAH:
Temporal smoothing sekarang pakai parameter sama untuk semua kelas.
Clip yang kecil dan sering muncul-hilang butuh window lebih besar.
Notebook yang besar dan stabil butuh alpha yang berbeda.

IMPLEMENTASI:
  TEMPORAL_CONFIG = {
      "clip"      : {"window": 5, "alpha": 0.50, "ghost_thr": 0.25},
      "sharpener" : {"window": 4, "alpha": 0.55, "ghost_thr": 0.28},
      "pen"       : {"window": 3, "alpha": 0.60, "ghost_thr": 0.30},
      "pencil"    : {"window": 3, "alpha": 0.60, "ghost_thr": 0.30},
      "notebook"  : {"window": 2, "alpha": 0.70, "ghost_thr": 0.35},
      "bottle"    : {"window": 2, "alpha": 0.70, "ghost_thr": 0.35},
      "stapler"   : {"window": 2, "alpha": 0.65, "ghost_thr": 0.32},
      "eraser"    : {"window": 3, "alpha": 0.60, "ghost_thr": 0.30},
      "correction_tape": {"window": 2, "alpha": 0.65, "ghost_thr": 0.32},
  }

  # Di temporal_smooth(), ganti parameter hardcoded:
  cfg    = TEMPORAL_CONFIG.get(target_class, {})
  window = cfg.get("window",    _TEMPORAL_WINDOW)
  alpha  = cfg.get("alpha",     _TEMPORAL_ALPHA)
  g_thr  = cfg.get("ghost_thr", 0.35)


────────────────────────────────────────────────────────────
D3. Aspect Ratio Range yang Lebih Ketat
Effort: SANGAT RENDAH | Dampak: LOW-MEDIUM | Prioritas: 6
────────────────────────────────────────────────────────────

MASALAH:
CLASS_AR_RANGE sekarang cukup longgar untuk beberapa kelas.
Bisa diperketat berdasarkan analisis ground truth bbox di dataset.

UPDATE CLASS_AR_RANGE:
  CLASS_AR_RANGE = {
      "pen"            : (2.0, 10.0),  # lebih ketat dari (1.5, 12.0)
      "pencil"         : (2.0, 10.0),
      "clip"           : (0.6, 2.0),   # lebih ketat dari (0.5, 2.5)
      "bottle"         : (0.20, 1.2),  # lebih ketat (lebih vertikal)
      "notebook"       : (0.6, 2.0),
      "correction_tape": (0.8, 2.5),
      "stapler"        : (1.0, 3.5),
      "eraser"         : (0.5, 3.0),
      "sharpener"      : (0.6, 2.0),
  }

CARA HITUNG NILAI OPTIMAL:
  # Di notebook evaluasi, tambahkan analisis:
  for det in ground_truth_boxes:
      w = det["bbox"][2] - det["bbox"][0]
      h = det["bbox"][3] - det["bbox"][1]
      ar = max(w/h, h/w)
      # Plot histogram AR per kelas
      # Ambil percentile 5-95 sebagai range


════════════════════════════════════════════════════════════
E. EVALUASI & FEEDBACK
════════════════════════════════════════════════════════════

────────────────────────────────────────────────────────────
E1. Confusion Matrix antar Kelas
Effort: MEDIUM | Dampak: TINGGI (untuk improve) | Prioritas: 1
────────────────────────────────────────────────────────────

MASALAH:
Kita tahu Macro F1 per kelas, tapi tidak tahu kelas mana yang
sering salah dikira kelas lain. Tanpa ini, tidak bisa tahu
prompt mana yang perlu diperbaiki.

IMPLEMENTASI DI NOTEBOOK:
  import numpy as np
  import matplotlib.pyplot as plt
  from sklearn.metrics import ConfusionMatrixDisplay

  # Kumpulkan semua prediksi vs ground truth
  y_true = []  # kelas ground truth
  y_pred = []  # kelas yang diprediksi sistem

  for img_path, gt_boxes in evaluation_set.items():
      for gt in gt_boxes:
          # Cari prediksi yang paling match (IoU > threshold)
          best_pred = find_best_matching_pred(gt, predictions[img_path])
          y_true.append(gt["class"])
          y_pred.append(best_pred["class"] if best_pred else "background")

  # Plot confusion matrix
  from sklearn.metrics import confusion_matrix
  cm = confusion_matrix(y_true, y_pred, labels=TARGET_CLASSES)
  disp = ConfusionMatrixDisplay(cm, display_labels=TARGET_CLASSES)
  disp.plot(xticks_rotation=45)
  plt.title("LangSight Confusion Matrix")
  plt.tight_layout()
  plt.savefig("confusion_matrix.png")

YANG DICARI:
  - Apakah "eraser" sering dikira "notebook"? → update negative cue
  - Apakah "pen" sering dikira "pencil"? → perkuat perbedaan prompt
  - Apakah banyak yang dikira "background"? → turunkan threshold


────────────────────────────────────────────────────────────
E2. Threshold Sensitivity Analysis
Effort: MEDIUM | Dampak: TINGGI | Prioritas: 2
────────────────────────────────────────────────────────────

MASALAH:
Threshold 0.35 sekarang hasil grid search sederhana.
Belum ada analisis sensitivitas yang sistematis per kelas.

IMPLEMENTASI DI NOTEBOOK:
  results = {}
  thresholds = np.arange(0.15, 0.65, 0.05)

  for thr in thresholds:
      f1_per_class = evaluate_all_classes(box_threshold=thr)
      results[thr] = f1_per_class

  # Plot F1 vs threshold per kelas
  fig, axes = plt.subplots(3, 3, figsize=(15, 12))
  for idx, cls in enumerate(TARGET_CLASSES):
      ax = axes[idx // 3][idx % 3]
      f1_values = [results[t][cls] for t in thresholds]
      ax.plot(thresholds, f1_values)
      ax.axvline(x=PER_CLASS_BOX_THR.get(cls, 0.35),
                 color='r', linestyle='--', label='current')
      ax.set_title(cls)
      ax.set_xlabel("box_threshold")
      ax.set_ylabel("F1")
      ax.legend()

  # Dari grafik ini, bisa update PER_CLASS_BOX_THR secara presisi


────────────────────────────────────────────────────────────
E3. Per-Kondisi Breakdown
Effort: TINGGI | Dampak: TINGGI | Prioritas: 3
────────────────────────────────────────────────────────────

MASALAH:
Tidak tahu performa sistem di kondisi berbeda:
gelap, ramai, objek tertutup, single object.

IMPLEMENTASI DI NOTEBOOK:
  def classify_condition(img_path: str, gt_boxes: list) -> str:
      img = cv2.imread(img_path)
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

      if gray.mean() < 80:
          return "dark"
      elif len(gt_boxes) == 1:
          return "single_object"
      elif len(gt_boxes) >= 4:
          return "cluttered"
      else:
          return "normal"

  # Evaluasi per kondisi:
  conditions = ["dark", "single_object", "cluttered", "normal"]
  results_by_condition = {c: {} for c in conditions}

  for img_path, gt_boxes in evaluation_set.items():
      condition = classify_condition(img_path, gt_boxes)
      f1 = evaluate_single_image(img_path, gt_boxes)
      for cls in TARGET_CLASSES:
          results_by_condition[condition][cls] = f1.get(cls, 0)


────────────────────────────────────────────────────────────
E4. Latency Benchmark Sistematis
Effort: RENDAH | Dampak: MEDIUM | Prioritas: 4
────────────────────────────────────────────────────────────

MASALAH:
Ada elapsed_ms di response tapi tidak ada evaluasi sistematis
pengaruh ukuran gambar, jumlah kelas, dan mode terhadap latency.

IMPLEMENTASI:
  import time
  import statistics

  def benchmark_latency(n_runs=50):
      results = {}

      test_sizes = [(320, 240), (640, 480), (1280, 720)]
      test_classes = TARGET_CLASSES
      modes = ["yolo_only", "dino_only", "ensemble"]

      for size in test_sizes:
          dummy = np.zeros((*size[::-1], 3), dtype=np.uint8)
          latencies = []
          for _ in range(n_runs):
              t0 = time.perf_counter()
              detector.detect(dummy, "pen")
              latencies.append((time.perf_counter() - t0) * 1000)

          results[f"{size[0]}x{size[1]}"] = {
              "mean_ms" : round(statistics.mean(latencies), 1),
              "p50_ms"  : round(statistics.median(latencies), 1),
              "p95_ms"  : round(sorted(latencies)[int(0.95*n_runs)], 1),
              "p99_ms"  : round(sorted(latencies)[int(0.99*n_runs)], 1),
          }

      return results


════════════════════════════════════════════════════════════
MASTER CHECKLIST — URUTAN IMPLEMENTASI
════════════════════════════════════════════════════════════

FASE 1 — ZERO KODE BARU (edit config saja)
─────────────────────────────────────────
  [ ] A1. Update CLASS_PROMPTS dengan atribut fisik
          → Edit string di app.py dan semua notebook
          → Kelas prioritas: clip, sharpener, pencil, correction_tape

  [ ] C4. Update _CLASS_NMS_IOU untuk semua 9 kelas
          → Tambah 3 kelas yang belum ada: correction_tape, stapler,
            notebook, bottle, eraser, sharpener

  [ ] C2. Update WBF IoU per kelas
          → Tambahkan WBF_IOU_PER_CLASS dan pakai di _fuse()

  [ ] D3. Perketat CLASS_AR_RANGE
          → Update nilai min/max berdasarkan analisis dataset


FASE 2 — KODE KECIL (< 20 baris per item)
─────────────────────────────────────────
  [ ] C1. Lazy DINO
          → ~10 baris di WBFEnsemble.detect()
          → Langsung fix correction_tape dan percepat latency

  [ ] B1. Adaptive Preprocessing
          → ~20 baris fungsi preprocess_adaptive()
          → Panggil di awal detect()

  [ ] D1. Re-ranking Post-Detection
          → ~25 baris fungsi rerank_detections()
          → Panggil sebelum return di detect()

  [ ] D2. Temporal Smoothing per Kelas
          → Tambah TEMPORAL_CONFIG dict
          → Update 3 baris di temporal_smooth()

  [ ] A6. Anchor Persistent
          → ~15 baris save/load ke JSON file
          → Update di /feedback endpoint

  [ ] A5. Typo Tolerance
          → ~15 baris fuzzy_keyword_predict()
          → Tambah sebagai fallback ke-3 di predict()


FASE 3 — KODE MEDIUM (30–100 baris)
────────────────────────────────────
  [ ] B2. Two-Stage Detection
          → ~50 baris detect_two_stage()
          → Aktifkan hanya untuk: clip, sharpener

  [ ] A2. Prompt Ensembling
          → ~30 baris detect_multi_prompt()
          → Aktifkan hanya untuk kelas DINO-only

  [ ] C3. Dynamic Weight per Frame
          → ~20 baris dynamic_weights()
          → Update di _fuse()

  [ ] A3. Query Expansion
          → ~40 baris extract_attributes() + build_enriched_prompt()
          → Update di /detect endpoint

  [ ] A4. Negation Handling
          → ~20 baris detect_negation()
          → Update di predict()


FASE 4 — ANALISIS (notebook)
─────────────────────────────
  [ ] E1. Confusion Matrix
          → Tambahkan di notebook 02 dan 03
          → Output: confusion_matrix_dino.png, confusion_matrix_ensemble.png

  [ ] E2. Threshold Sensitivity Analysis
          → Tambahkan di semua notebook
          → Output: threshold_sensitivity_{kelas}.png

  [ ] B4. Hitung Platt Scaling dari notebook
          → Setelah E2 selesai, hitung a,b per kelas
          → Apply via /admin/calibrate atau hardcode di _CALIBRATION

  [ ] E3. Per-Kondisi Breakdown
          → Tambahkan classifier kondisi di notebook
          → Output: per_condition_f1.json

  [ ] E4. Latency Benchmark
          → Jalankan setelah semua optimisasi Fase 1-3 selesai
          → Dokumentasikan sebelum vs sesudah


FASE 5 — ADVANCED (opsional, kalau masih mau)
──────────────────────────────────────────────
  [ ] B3. Multi-Scale Inference
          → Hanya kalau two-stage belum cukup untuk clip/sharpener
          → 3x lebih lambat — pakai dengan bijak

  [ ] Pakai Grounding DINO tiny untuk speed, base untuk accuracy
          → Endpoint /admin/model sudah ada, tinggal benchmark


════════════════════════════════════════════════════════════
EKSPEKTASI PENINGKATAN F1 PER FASE
════════════════════════════════════════════════════════════

  Baseline sekarang    : Macro F1 = 0.736

  Setelah Fase 1       : ~0.760–0.775
    (prompt lebih tajam, NMS dan WBF lebih presisi)

  Setelah Fase 2       : ~0.790–0.810
    (lazy DINO fix correction_tape, preprocessing fix dark images,
     reranking dan temporal per kelas)

  Setelah Fase 3       : ~0.820–0.840
    (two-stage untuk clip/sharpener, prompt ensembling)

  Setelah Fase 4       : ~0.840–0.860
    (threshold optimal dari sensitivity analysis,
     Platt scaling untuk calibrated confidence)

  Target realistis akhir: Macro F1 ~0.85+
  (dari 0.736 sekarang — peningkatan ~15%)

  Kelas yang paling berubah:
    clip       : 0.552 → ~0.680 (two-stage + attribute prompt)
    sharpener  : 0.590 → ~0.700 (two-stage + attribute prompt)
    correction_tape: 0.909 → ~0.950+ (lazy DINO)
    pencil     : 0.867 → ~0.900 (per-class WBF IoU + AR range)


════════════════════════════════════════════════════════════
SATU ATURAN PENTING
════════════════════════════════════════════════════════════

Setelah setiap fase, SELALU jalankan ulang 3 notebook evaluasi
dan bandingkan Macro F1 baru vs sebelumnya.

Kalau F1 turun setelah perubahan → rollback perubahan tersebut.
Tidak semua optimisasi cocok untuk semua dataset.
Angka evaluasi adalah hakim terakhir.
