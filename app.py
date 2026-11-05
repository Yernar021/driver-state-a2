"""
Assignment 2 - веб-приложение (Gradio): видео с камеры или файл -> состояние водителя + живость + объяснение.

Что показывает:
  1. Состояние по секундам: сонливость (alert / eyes_closed / yawning) и внимание (looking_road / not),
     модели EB из шага 20 (DINOv2) или из шага 10 (MobileNetV2), если DINOv2-моделей ещё нет.
  2. Живость: правило моргания из статьи (EAR < 0.8 x своя медиана за 10 с; живой, если моргнул за 10 с).
  3. Объяснение: карты внимания DINOv2 (CLS -> патчи, последний слой) на лице в ключевые моменты.
  4. Задержку по этапам (мс на кадр) и окружение (CPU/GPU, ОС, браузер пишет сам Gradio).
Первые 10 с видео - калибровка (смотреть вперёд, как на дорогу). Нужно видео от ~20 с.

Запуск (из C:\\driver_project):
  pip install gradio torch transformers mediapipe opencv-python scikit-learn joblib matplotlib
  python code\\app.py            -> открыть http://127.0.0.1:7860
  python code\\app.py --share    -> временная публичная ссылка (для защиты с телефона)
"""
import argparse
import os
import platform
import sys
import time

import cv2
import joblib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dino import Dino, overlay  # noqa: E402

MODEL_PATH = os.path.join(HERE, "face_landmarker.task")
PROC_FPS = 15          # кадров/с для моргания (как в записях спуфинга)
STATE_EVERY = 3        # состояние раз в 3 кадра = 5 кадров/с (как в обучении)
BLINK_R, LIVE_T = 0.8, 10.0

# ---- формулы из 02_extract_features.py / 11_demo_live.py (не менять) ----
R_EYE = [33, 160, 158, 133, 153, 144]
L_EYE = [362, 385, 387, 263, 373, 380]
MOUTH_V = [(81, 178), (13, 14), (311, 402)]
MOUTH_H = (78, 308)
POSE_IDX = [1, 152, 263, 33, 291, 61]
MODEL_3D = np.array([(0.0, 0.0, 0.0), (0.0, -63.6, -12.5), (-43.3, 32.7, -26.0),
                     (43.3, 32.7, -26.0), (-28.9, -28.9, -24.1), (28.9, -28.9, -24.1)], dtype=np.float64)


def dist(a, b):
    return float(np.linalg.norm(a - b))


def ear(pts, idx):
    p = [pts[i] for i in idx]
    return (dist(p[1], p[5]) + dist(p[2], p[4])) / (2.0 * dist(p[0], p[3]) + 1e-9)


def mar(pts):
    v = sum(dist(pts[a], pts[b]) for a, b in MOUTH_V)
    return v / (3.0 * dist(pts[MOUTH_H[0]], pts[MOUTH_H[1]]) + 1e-9)


def head_pose(pts, w, h):
    img = np.array([pts[i] for i in POSE_IDX], dtype=np.float64)
    cam = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(MODEL_3D, img, cam, None, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return np.nan, np.nan, np.nan
    rot, _ = cv2.Rodrigues(rvec)
    angles = cv2.RQDecomp3x3(rot)[0]
    wrap = lambda a: float((a + 90.0) % 180.0 - 90.0)
    return wrap(angles[1]), wrap(angles[0]), wrap(angles[2])


def crop_face(frame, pts, margin=0.25, size=224):
    h, w = frame.shape[:2]
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(x1 - x0, y1 - y0) * (1 + margin)
    xa, ya = int(max(cx - side / 2, 0)), int(max(cy - side / 2, 0))
    xb, yb = int(min(cx + side / 2, w)), int(min(cy + side / 2, h))
    if xb - xa < 10 or yb - ya < 10:
        return None
    return cv2.resize(frame[ya:yb, xa:xb], (size, size))
# ---- конец формул ----


class OnlineFeatures:
    """Калибровка + окно 1 с, как в обучении (копия из 11_demo_live.py)."""
    BASE = ["ear", "mar", "yaw", "pitch", "roll"]

    def __init__(self, n_cal, window):
        self.n_cal, self.window = n_cal, window
        self.cal_base, self.cal_blend, self.cal_emb = [], [], []
        self.med, self.last_base, self.nan_run = None, None, 0
        self.hist = []

    def ffill(self, base):
        if np.isnan(base).any():
            if self.last_base is not None and self.nan_run < 3:
                self.nan_run += 1
                base = np.where(np.isnan(base), self.last_base, base)
            else:
                return None
        else:
            self.nan_run = 0
        self.last_base = base
        return base

    def add(self, base, blend, emb):
        if self.med is None:
            self.cal_base.append(base); self.cal_blend.append(blend); self.cal_emb.append(emb)
            if len(self.cal_base) >= self.n_cal:
                b = np.array(self.cal_blend, np.float64)
                self.med = dict(base=np.median(np.array(self.cal_base), axis=0),
                                blend=np.nanmedian(b, axis=0) if np.isfinite(b).any() else np.zeros(52),
                                emb=np.median(np.array(self.cal_emb), axis=0))
                self.hist = [x - self.med["base"] for x in self.cal_base[-self.window:]]
            return None
        bc = base - self.med["base"]
        self.hist = (self.hist + [bc])[-self.window:]
        H, k = np.array(self.hist), self.window
        feats = dict(zip(self.BASE, bc))
        std = lambda v: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
        for j, c in [(0, "ear"), (1, "mar")]:
            feats[f"{c}_mean{k}"] = H[:, j].mean(); feats[f"{c}_min{k}"] = H[:, j].min()
            feats[f"{c}_max{k}"] = H[:, j].max(); feats[f"{c}_std{k}"] = std(H[:, j])
        for j, c in [(2, "yaw"), (3, "pitch")]:
            feats[f"{c}_mean{k}"] = H[:, j].mean(); feats[f"{c}_std{k}"] = std(H[:, j])
        bcal = np.nan_to_num(blend - self.med["blend"], nan=0.0)
        for i in range(52):
            feats[f"bs_{i}"] = bcal[i]
        feats["bs_ok"] = float(np.isfinite(blend).all())
        return feats, emb - self.med["emb"]


def make_vector(b, feats, emb_c):
    xa = [feats[n] for n in b["feats_A"]]
    xb = [feats[n] for n in b["feats_B"]]
    z = b["pca"].transform(b["scaler"].transform(emb_c[None, :]))[0]
    return np.r_[xa, xb, z].astype(np.float32)[None, :]


# ---------------- загрузка моделей (один раз) ----------------
class Engine:
    def __init__(self, models_dir):
        self.models, self.backbone = {}, None
        for bb in ["dino", "mnv2"]:
            fns = {t: os.path.join(models_dir, f"eb_{t}_{bb}.joblib") for t in ["drowsy", "attention"]}
            if bb == "mnv2":   # старые файлы из шага 10
                fns = {t: fn if os.path.exists(fn) else os.path.join(models_dir, f"eb_{t}.joblib")
                       for t, fn in fns.items()}
            if all(os.path.exists(f) for f in fns.values()):
                self.backbone = bb
                for t, fn in fns.items():
                    b = joblib.load(fn); b["rf"].n_jobs = 1
                    self.models[t] = b
                break
        if not self.models:
            sys.exit("Нет моделей: python code\\20_train_final_a2.py --task drowsy/attention --backbone dino")
        self.dino = Dino()
        self.mnv2 = None
        if self.backbone == "mnv2":
            import tensorflow as tf
            net = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet",
                                                    input_shape=(224, 224, 3), pooling="avg")
            prep = tf.keras.applications.mobilenet_v2.preprocess_input
            f = tf.function(lambda x: net(x, training=False),
                            input_signature=[tf.TensorSpec([1, 224, 224, 3], tf.float32)])
            self.mnv2 = lambda rgb: f(tf.constant(prep(rgb.astype(np.float32))[None])).numpy()[0]
        self.dino.embed([np.zeros((224, 224, 3), np.uint8)])   # прогрев
        print(f"Модели: EB + {self.backbone} | устройство DINOv2: {self.dino.device}")

    def embed(self, crop_rgb):
        if self.backbone == "dino":
            return self.dino.embed([crop_rgb])[0][0]
        return self.mnv2(crop_rgb)


def landmarker():
    import mediapipe as mp
    return mp, mp.tasks.vision.FaceLandmarker.create_from_options(mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp.tasks.vision.RunningMode.VIDEO, num_faces=2, output_face_blendshapes=True))


def stamp(img, text):
    """Водяной знак на картинке (синтетика / демо)."""
    img = img.copy()
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, h - 22), (w, h), (0, 0, 0), -1)
    cv2.putText(img, text, (5, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return img


def analyse(video, consent, synthetic, eng):
    import pandas as pd
    if not consent:
        return "**Нужно подтвердить согласие человека в кадре.**", None, [], None
    if video is None:
        return "Загрузите видео или запишите с камеры.", None, [], None
    cap = cv2.VideoCapture(video)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not np.isfinite(src_fps) or src_fps <= 0 or src_fps > 240:
        src_fps = 30.0
    mp, lm = landmarker()
    ref = eng.models["drowsy"]
    of = OnlineFeatures(int(round(ref["calib_sec"] * ref["fps"])), ref["window"])
    names = ref["blend_names"]
    rows, crops, times = [], [], {"MediaPipe": [], "features": [], f"embedding ({eng.backbone})": [],
                                  "RF x2": [], "total": []}
    i, nxt, k, t_last = 0, 0.0, 0, -1.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # время кадра: из метки контейнера (запись с веб-камеры в браузере часто с неверным FPS)
        pm = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        t = pm if (pm > t_last and pm > 0) else i / src_fps
        i += 1
        if t + 1e-6 < nxt or t <= t_last:
            continue
        nxt = max(nxt, t) + 1.0 / PROC_FPS
        t_last = t
        t0 = time.perf_counter()
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)),
                                  int(t * 1000))
        t1 = time.perf_counter()
        r = dict(t=t, face=0, ear=np.nan)
        pts = None
        if res.face_landmarks:
            P = [np.array([(p.x * w, p.y * h) for p in f]) for f in res.face_landmarks]
            areas = [np.ptp(p[:, 0]) * np.ptp(p[:, 1]) for p in P]
            o = np.argsort(areas)[::-1]
            if len(P) == 1 or areas[o[1]] <= 0.3 * areas[o[0]]:   # второе лицо > 30% - кадр не берём
                pts, fi = P[o[0]], int(o[0])
        if pts is not None:
            r.update(face=1, ear=(ear(pts, R_EYE) + ear(pts, L_EYE)) / 2)
        do_state = (k % STATE_EVERY == 0)
        k += 1
        if do_state and pts is not None:
            crop = crop_face(frame, pts)
            raw = np.array([r["ear"], mar(pts), *head_pose(pts, w, h)], np.float64)
            base = of.ffill(raw) if crop is not None else None
            if base is not None:
                blend = np.full(52, np.nan)
                if res.face_blendshapes:
                    sc = {c.category_name: c.score for c in res.face_blendshapes[fi]}
                    nm = names or [c.category_name for c in res.face_blendshapes[fi]]
                    blend = np.array([sc.get(n, np.nan) for n in nm[:52]], np.float64)
                _, jpg = cv2.imencode(".jpg", crop)   # как при обучении: лицо хранилось в JPG
                crop_rgb = cv2.cvtColor(cv2.imdecode(jpg, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                t2 = time.perf_counter()
                emb = eng.embed(crop_rgb)
                t3 = time.perf_counter()
                out = of.add(base, blend, emb)
                if out is not None:
                    feats, emb_c = out
                    for task, b in eng.models.items():
                        pr = b["rf"].predict_proba(make_vector(b, feats, emb_c))[0]
                        r.update({f"p_{c}": float(p) for c, p in zip(b["classes"], pr)})
                        r[task] = b["classes"][int(pr.argmax())]
                    t4 = time.perf_counter()
                    for key, v in zip(times, [t1 - t0, t2 - t1, t3 - t2, t4 - t3, t4 - t0]):
                        times[key].append(v)
                    crops.append((len(rows), crop_rgb))
        elif do_state and of.med is not None:
            r["drowsy"], r["attention"] = ref["noface_label"], eng.models["attention"]["noface_label"]
        rows.append(r)
    cap.release(); lm.close()
    if not rows:
        return "Не удалось прочитать видео.", None, [], None
    D = pd.DataFrame(rows)
    dur = D["t"].iloc[-1]
    if of.med is None:
        return (f"Видео {dur:.0f} с: лицо найдено слишком редко или видео короче калибровки "
                f"({ref['calib_sec']} с). Нужно ~20 с, лицо в кадре."), None, [], None

    # ---- живость: правило из статьи ----
    e = D["ear"].where(D["face"] == 1)
    norm = e.rolling(int(round(10 * PROC_FPS)), min_periods=int(round(2 * PROC_FPS))).median().shift(1)
    hit = (e < BLINK_R * norm).fillna(False).values
    onset = np.r_[False, hit[1:] & ~hit[:-1]]
    D["blink"] = onset
    sec = np.arange(LIVE_T, dur + 1e-9, 1.0)
    bt = D["t"].values[onset]
    live = np.array([((bt > s - LIVE_T) & (bt <= s)).any() for s in sec])

    # ---- сводка ----
    S = D.dropna(subset=["drowsy"]) if "drowsy" in D else D.iloc[:0]
    share = lambda col, v: 100 * (S[col] == v).mean() if len(S) else float("nan")
    med = {k_: 1000 * np.median(v) for k_, v in times.items() if v}
    env = (f"{platform.system()} {platform.release()}, {platform.processor() or platform.machine()}, "
           f"Python {platform.python_version()}, DINOv2 on {eng.dino.device}")
    tag = "⚠️ **SYNTHETIC / AI-GENERATED MEDIA** - результаты относятся к сгенерированному видео.\n\n" if synthetic else ""
    verdict = ("**LIVE** - моргания есть" if live.size and live[-1]
               else "**ATTACK SUSPECTED** - нет моргания за последние 10 с") if live.size else "мало данных"
    md = tag + f"""### Результат ({dur:.0f} с видео, модели EB + {eng.backbone})
| | |
|---|---|
| Сонливость: alert / eyes_closed / yawning | {share('drowsy','alert'):.0f}% / {share('drowsy','eyes_closed'):.0f}% / {share('drowsy','yawning'):.0f}% секунд-кадров |
| Внимание: смотрит на дорогу | {share('attention','looking_road'):.0f}% |
| Живость (моргание за 10 с) | {verdict}; живых решений {100 * live.mean() if live.size else 0:.0f}% , морганий {int(onset.sum())} ({60 * onset.sum() / max(dur, 1):.0f}/мин) |
| Задержка, мс/кадр (медиана) | """ + ", ".join(f"{k_} {v:.1f}" for k_, v in med.items()) + f""" |
| Скорость | ~{1000 / med.get('total', np.nan):.0f} кадров/с (системе нужно 5) |
| Окружение | {env} |

Первые {ref['calib_sec']} с - калибровка (состояние не выводится). Карты внимания показывают, на какие участки лица
смотрит DINOv2; это объяснение признаков, а не причинное доказательство решения леса."""

    # ---- график ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 1, figsize=(10, 6.4), sharex=True)
    for c, col in [("p_eyes_closed", "#7030a0"), ("p_yawning", "#c55a11"), ("p_not_looking_road", "#2e75b6")]:
        if c in D:
            q = D.dropna(subset=[c])
            ax[0].plot(q["t"], q[c], label=c[2:], color=col, lw=1.2)
    ax[0].set_ylim(0, 1); ax[0].set_ylabel("probability"); ax[0].legend(loc="upper right", fontsize=8)
    ax[0].axvspan(0, D.loc[D["drowsy"].notna(), "t"].min() if "drowsy" in D and D["drowsy"].notna().any() else 10,
                  color="grey", alpha=0.15)
    ax[0].set_title("Driver state (grey = calibration)")
    ax[1].plot(D["t"], D["ear"], color="#404040", lw=0.8, label="EAR")
    ax[1].plot(D["t"], BLINK_R * norm, color="#c00000", lw=0.8, ls="--", label=f"{BLINK_R} x own 10 s median")
    ax[1].scatter(D.loc[onset, "t"], D.loc[onset, "ear"], color="#c00000", s=12, zorder=3, label="blink")
    ax[1].set_ylabel("EAR"); ax[1].legend(loc="upper right", fontsize=8)
    ax[2].step(sec, live.astype(int), where="post", color="#548235")
    ax[2].set_yticks([0, 1], ["attack?", "live"]); ax[2].set_xlabel("time, s"); ax[2].set_ylim(-0.2, 1.2)
    ax[2].set_title("Liveness: blink in last 10 s")
    if synthetic:
        fig.text(0.5, 0.5, "SYNTHETIC", fontsize=60, color="red", alpha=0.15, ha="center", va="center", rotation=20)
    fig.tight_layout()

    # ---- объяснение: карты внимания DINOv2 в ключевые моменты ----
    gallery = []
    if crops:
        idx = np.array([c[0] for c in crops])
        pick = []
        for c in ["p_eyes_closed", "p_yawning", "p_not_looking_road", "p_alert"]:
            if c in D:
                v = D[c].values[idx]
                if np.isfinite(v).any():
                    pick.append((c[2:], int(np.nanargmax(v)), float(np.nanmax(v))))
        ims = [crops[j][1] for _, j, _ in pick]
        _, att = eng.dino.embed(ims, with_attention=True)
        label = "SYNTHETIC - AI-generated" if synthetic else "demo - consented subject"
        for (name, j, p), im, a_ in zip(pick, ims, att):
            t_ = D["t"].values[crops[j][0]]
            gallery.append((stamp(np.hstack([im, overlay(im, a_)]), label),
                            f"max p({name}) = {p:.2f} at {t_:.1f} s"))
    return md, fig, gallery, D.round(4)


def main():
    import gradio as gr
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="models")
    ap.add_argument("--share", action="store_true")
    a = ap.parse_args()
    eng = Engine(a.models)
    with gr.Blocks(title="Driver State + Liveness") as demo:
        gr.Markdown("# Driver state monitoring + liveness (Assignment 2)\n"
                    "Upload a 20–60 s video or record with the camera. Look ahead for the first 10 s (calibration). "
                    "Video is processed locally and not stored.")
        with gr.Row():
            with gr.Column(scale=1):
                vid = gr.Video(sources=["upload", "webcam"], label="Video (camera or file)")
                consent = gr.Checkbox(label="The person in the video agreed to be recorded and analysed", value=False)
                synth = gr.Checkbox(label="This video is synthetic / AI-generated (will be watermarked)", value=False)
                btn = gr.Button("Analyse", variant="primary")
            with gr.Column(scale=2):
                md = gr.Markdown()
                plot = gr.Plot(label="Timeline")
        gal = gr.Gallery(label="Explanation: DINOv2 attention (CLS → patches, last layer)", columns=4, height=300)
        table = gr.Dataframe(label="Per-frame output", wrap=False)
        btn.click(lambda v, c, s: analyse(v, c, s, eng), [vid, consent, synth], [md, plot, gal, table])
    demo.launch(share=a.share)


if __name__ == "__main__":
    main()
