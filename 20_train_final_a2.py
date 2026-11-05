"""
Шаг 20 (Assignment 2). Итоговые модели для веб-приложения: EB на выбранном backbone, обучение на ВСЕХ людях.
Для сонливости кадры разговора добавляются как alert (EB+T), как в статье.
Оценки качества не даёт: она в 19 (LOPO).

Запуск (из C:\\driver_project):
  python code\\20_train_final_a2.py --task drowsy --backbone dino
  python code\\20_train_final_a2.py --task attention --backbone dino
Результат: models\\eb_<task>_<backbone>.joblib   (app.py берёт их автоматически)
"""
import argparse
import glob
import importlib
import os
import sys

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
X19 = importlib.import_module("19_a2_experiments")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(X19.TASKS), required=True)
    ap.add_argument("--backbone", choices=["dino", "mnv2"], default="dino")
    ap.add_argument("--labels", default=r"data_processed\dmd\labels.csv")
    ap.add_argument("--landmarks", default=r"data_processed\dmd\landmarks")
    ap.add_argument("--emb-mnv2", default=r"results\emb_all_v2.npy")
    ap.add_argument("--emb-dino", default=r"results\emb_dinov2_s.npy")
    ap.add_argument("--out", default="models")
    a = ap.parse_args()
    label_col, classes = X19.TASKS[a.task]
    fk, XF, Ec, talk, _ = X19.load(a, label_col, classes)
    y = fk[label_col].values.astype(object)
    use = np.isin(y, classes) | (talk if a.task == "drowsy" else False)
    y = np.where(talk, "alert", y)[use]
    E = Ec[a.backbone][use]
    sub = np.random.RandomState(0).permutation(len(E))[:X19.PCA_FIT_MAX]
    sc = StandardScaler().fit(E[sub])
    pca = PCA(X19.N_PCA, random_state=0).fit(sc.transform(E[sub]))
    X = np.hstack([XF[use], pca.transform(sc.transform(E))]).astype(np.float32)
    clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced",
                                 random_state=0, n_jobs=-1).fit(X, y)
    _, feats_A = X19.add_window_features(fk.head(10), X19.WINDOW)
    blend_names = None
    for fn in sorted(glob.glob(os.path.join(a.landmarks, "*.npz"))):
        d = np.load(fn)
        if len(d["blend_names"]):
            blend_names = [str(x) for x in d["blend_names"]]
            break
    bundle = dict(task=a.task, backbone=a.backbone, classes=list(clf.classes_),
                  noface_label=X19.NOFACE_RULE[a.task], feats_A=feats_A, feats_B=X19.BS + ["bs_ok"],
                  blend_names=blend_names, window=X19.WINDOW, calib_sec=X19.CALIB_SEC, fps=X19.FPS,
                  scaler=sc, pca=pca, rf=clf)
    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, f"eb_{a.task}_{a.backbone}.joblib")
    joblib.dump(bundle, out, compress=3)
    print(f"{a.task}/{a.backbone}: {len(y)} кадров, {X.shape[1]} признаков -> {out}")


if __name__ == "__main__":
    main()
