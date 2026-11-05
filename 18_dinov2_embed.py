"""
Шаг 18 (Assignment 2). Эмбеддинги DINOv2 для всех кадров DMD с лицом.
Порядок строк ТОТ ЖЕ, что у results\\emb_all_v2.npy (MobileNetV2): кадры с face_found == 1,
отсортированные по video_id, frame_idx; k-й такой кадр записи = frames\\<video_id>\\<k:06d>.jpg.
Поэтому оба файла можно подставлять в одни и те же скрипты.

Считается кусками по 2000 кадров, кусок сразу сохраняется -> если прервать (Ctrl+C), повторный
запуск продолжит с того же места.

Запуск (из C:\\driver_project):
  pip install torch transformers
  python code\\18_dinov2_embed.py
Результат: results\\emb_dinov2_s.npy  (N x 768, float16)
Время: на 12-ядерном CPU ожидается порядка 30-60 мин на ~75 тыс. кадров (печатается по ходу).
"""
import argparse
import glob
import os
import time

import cv2
import numpy as np
import pandas as pd

from dino import Dino, EMB_DIM


def face_table(labels, frames):
    lab = pd.read_csv(labels, keep_default_na=False, na_values=[""], low_memory=False,
                      usecols=lambda c: c in ("video_id", "frame_idx", "face_found"))
    lab["face_found"] = pd.to_numeric(lab["face_found"], errors="coerce").fillna(0)
    face = lab[lab["face_found"] == 1].sort_values(["video_id", "frame_idx"]).copy()
    face["k"] = face.groupby("video_id").cumcount()
    face["path"] = [os.path.join(frames, v, f"{k:06d}.jpg") for v, k in zip(face["video_id"], face["k"])]
    return face


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=r"data_processed\dmd\labels.csv")
    ap.add_argument("--frames", default=r"data_processed\dmd\frames")
    ap.add_argument("--out", default=r"results\emb_dinov2_s.npy")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=0, help="потоки CPU (0 = по умолчанию torch)")
    a = ap.parse_args()

    face = face_table(a.labels, a.frames)
    paths = face["path"].tolist()
    n = len(paths)
    miss = [p for p in paths[:: max(n // 500, 1)] if not os.path.exists(p)]
    print(f"Кадров с лицом: {n} | записей: {face['video_id'].nunique()}")
    if miss:
        raise SystemExit(f"Не найдены картинки, например {miss[0]}. Проверь --frames "
                         f"(папка с подпапками записей, внутри 000000.jpg, 000001.jpg, ...)")
    ref = r"results\emb_all_v2.npy"
    if os.path.exists(ref):
        n_ref = len(np.load(ref, mmap_mode="r"))
        print(f"Проверка порядка: emb_all_v2.npy = {n_ref} строк -> " + ("совпадает" if n_ref == n else "НЕ совпадает!"))

    part_dir = a.out + ".parts"
    os.makedirs(part_dir, exist_ok=True)
    dino = Dino(threads=a.threads or None)
    print("Устройство:", dino.device)
    t0, done_now = time.time(), 0
    for s in range(0, n, a.chunk):
        fn = os.path.join(part_dir, f"{s:07d}.npy")
        if os.path.exists(fn):
            continue
        idx = range(s, min(s + a.chunk, n))
        E = np.zeros((len(idx), EMB_DIM), np.float16)
        for j in range(0, len(idx), a.batch):
            ims = [cv2.imread(paths[i])[:, :, ::-1] for i in idx[j:j + a.batch]]   # BGR -> RGB
            E[j:j + len(ims)] = dino.embed(ims)[0]
        np.save(fn, E)
        done_now += len(idx)
        el = time.time() - t0
        left = n - s - len(idx)
        print(f"  {s + len(idx)}/{n}  |  {1000 * el / done_now:.1f} мс/кадр  |  осталось ~{left * el / done_now / 60:.0f} мин",
              flush=True)

    parts = sorted(glob.glob(os.path.join(part_dir, "*.npy")))
    E = np.concatenate([np.load(p) for p in parts])
    assert len(E) == n, f"собрано {len(E)} из {n}"
    np.save(a.out, E)
    print(f"Сохранено: {a.out} {E.shape}. Папку {part_dir} можно удалить.")


if __name__ == "__main__":
    main()
