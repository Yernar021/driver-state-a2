"""
Шаг 19 (Assignment 2). Бейзлайн A1 против SOTA (DINOv2), кривые обучения, абляция. Всё по LOPO.

Варианты (один человек в тесте, всё подстраиваемое обучается только на остальных):
  A1_mnv2_lr   бейзлайн A1: эмбеддинги MobileNetV2 (ImageNet) + логистическая регрессия
  S_dino_lr    то же, но эмбеддинги DINOv2 ViT-S/14  -> чистая замена backbone
  EB_mnv2      гибрид из статьи: геометрия + окно 1 с + 52 blendshapes + PCA32(MobileNetV2), RandomForest
  EB_dino      тот же гибрид, но PCA32(DINOv2)
  MLP_dino     признаки EB_dino, маленькая нейросеть (128-64) -> ДЛЯ КРИВЫХ ОБУЧЕНИЯ (loss и F1 по эпохам)
Абляция (только drowsy): EB_mnv2 / EB_dino с кадрами разговора в обучении (+T) и без.
  Для неё же: какая доля кадров разговора тестового человека названа зевком.

Калибровка, окна и правило для кадров без лица - как в статье (вычитание медианы первых 10 с записи).
Метрика: macro-F1 по каждому человеку, среднее по людям; парный тест Вилкоксона по людям.

Запуск (из C:\\driver_project), нужен results\\emb_dinov2_s.npy из шага 18:
  python code\\19_a2_experiments.py --task drowsy
  python code\\19_a2_experiments.py --task attention
  быстрее (без логистической регрессии, она самая медленная):  --skip A1_mnv2_lr,S_dino_lr
Результат: results\\a2\\  (summary_*.csv, per_person_*.csv, curves_*.png/.csv, cm_*.png, bars_*.png)
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = ["ear", "mar", "yaw", "pitch", "roll"]
TASKS = {"drowsy": ("drowsy_label", ["alert", "eyes_closed", "yawning"]),
         "attention": ("attention_label", ["looking_road", "not_looking_road"])}
NOFACE_RULE = {"drowsy": "alert", "attention": "not_looking_road"}
BS = [f"bs_{i}" for i in range(52)]
WINDOW, CALIB_SEC, FPS, N_PCA, PCA_FIT_MAX = 5, 10, 5, 32, 20000
EPOCHS = 30


def add_window_features(df, k):
    df = df.sort_values(["video_id", "frame_idx"]).copy()
    g = df.groupby("video_id")
    feats = list(BASE)
    for c in ["ear", "mar"]:
        for name in ["mean", "min", "max", "std"]:
            col = f"{c}_{name}{k}"
            df[col] = g[c].transform(lambda s, n=name: getattr(s.rolling(k, min_periods=1), n)())
            feats.append(col)
    for c in ["yaw", "pitch"]:
        for name in ["mean", "std"]:
            col = f"{c}_{name}{k}"
            df[col] = g[c].transform(lambda s, n=name: getattr(s.rolling(k, min_periods=1), n)())
            feats.append(col)
    wcols = [x for x in feats if x not in BASE]
    df[wcols] = df[wcols].fillna(0)
    return df, feats


def load(a, label_col, classes):
    lab = pd.read_csv(a.labels, keep_default_na=False, na_values=[""], low_memory=False)
    for c in BASE:
        lab[c] = pd.to_numeric(lab[c], errors="coerce")
    lab["face_found"] = pd.to_numeric(lab["face_found"], errors="coerce").fillna(0)
    face = lab[lab["face_found"] == 1].sort_values(["video_id", "frame_idx"]).copy()
    face["eidx"] = np.arange(len(face))
    embs = {}
    for name, fn in [("mnv2", a.emb_mnv2), ("dino", a.emb_dino)]:
        E = np.load(fn, mmap_mode="r")
        if len(E) != len(face):
            raise SystemExit(f"{fn}: {len(E)} строк, а кадров с лицом {len(face)}")
        embs[name] = E

    Bm = np.full((len(face), 52), np.nan, np.float32)
    pos = {k: i for i, k in enumerate(zip(face["video_id"], face["frame_idx"]))}
    for vid in face["video_id"].unique():
        fn = os.path.join(a.landmarks, vid + ".npz")
        if not os.path.exists(fn):
            continue
        d = np.load(fn)
        rows = np.array([pos.get((vid, int(x)), -1) for x in d["frame_idx"]])
        ok = (rows >= 0) & (d["face"] == 1)
        Bm[rows[ok]] = d["blend"][ok][:, :52]
    face[BS] = Bm

    n_cal = int(round(CALIB_SEC * FPS))
    f = face
    f[BASE] = f.groupby("video_id")[BASE].ffill(limit=3)
    f = f.dropna(subset=BASE).sort_values(["video_id", "frame_idx"])
    f["is_cal"] = (f.groupby("video_id").cumcount() < n_cal).values
    f["bs_ok"] = f[BS].notna().all(axis=1).astype(float)
    ref = f[f["is_cal"]].groupby("video_id")[BASE].median()
    f[BASE] = f[BASE].values - ref.reindex(f["video_id"]).values
    ref_b = f[f["is_cal"]].groupby("video_id")[BS].median()
    f[BS] = (f[BS].values - ref_b.reindex(f["video_id"]).values)
    f[BS] = f[BS].fillna(0.0)
    f, feats_A = add_window_features(f, WINDOW)
    feats = feats_A + BS + ["bs_ok"]

    lab_mask = f[label_col].isin(classes).values
    talk = np.zeros(len(f), bool)
    if "talking" in f.columns:
        tk = f["talking"].fillna("").astype(str).str.strip().str.lower()
        talk = (tk.str.contains("talk") & ~tk.str.contains("not")).values & ~lab_mask
    keep = lab_mask | talk
    vids = f["video_id"].values
    cal_rows = np.flatnonzero(f["is_cal"].values)
    rows = np.flatnonzero(keep)
    Ec = {}
    for name, E in embs.items():   # калибровка эмбеддингов: минус медиана первых 10 с записи
        X = np.asarray(E[f["eidx"].values[rows]], dtype=np.float32)
        for v in np.unique(vids[rows]):
            cr = cal_rows[vids[cal_rows] == v]
            X[vids[rows] == v] -= np.median(np.asarray(E[f["eidx"].values[cr]], np.float32), axis=0)
        Ec[name] = X
    fk = f.iloc[rows].reset_index(drop=True)
    labeled = lab[lab[label_col].isin(classes)]
    noface = labeled[~labeled.index.isin(f.index[lab_mask])]
    return fk, fk[feats].values.astype(np.float32), Ec, talk[rows], noface


def rf():
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced",
                                  random_state=0, n_jobs=-1)


def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced"))


def pca_block(E, tr, te):
    sub = tr[np.random.RandomState(0).permutation(len(tr))[:PCA_FIT_MAX]]
    sc = StandardScaler().fit(E[sub])
    pca = PCA(N_PCA, random_state=0).fit(sc.transform(E[sub]))
    return pca.transform(sc.transform(E[tr])), pca.transform(sc.transform(E[te]))


def macro_f1_people(y, pr, people, persons):
    out = []
    for p in persons:
        m = people == p
        out.append(f1_score(y[m], pr[m], labels=sorted(set(y[m])), average="macro", zero_division=0))
    return np.array(out)


def run_mlp_curves(Xtr, ytr, Xte, yte, lab_te, classes, seed=0):
    """Нейросеть 128-64 по эпохам: loss на обучении, macro-F1 на обучении и на тестовом человеке."""
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    rng = np.random.RandomState(seed)
    m = MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3, learning_rate_init=1e-3,
                      batch_size=256, random_state=seed)
    # баланс классов: на каждой эпохе каждый класс берётся одинаковое число раз (как class_weight)
    idx_c = [np.flatnonzero(ytr == c) for c in classes if (ytr == c).any()]
    per = int(np.median([len(i) for i in idx_c]))
    sub_tr = rng.permutation(len(ytr))[:5000]
    hist = []
    for ep in range(EPOCHS):
        bal = np.concatenate([rng.choice(i, per, replace=len(i) < per) for i in idx_c])
        rng.shuffle(bal)
        m.partial_fit(Xtr[bal], ytr[bal], classes=np.array(classes))
        hist.append(dict(epoch=ep + 1, train_loss=m.loss_,
                         train_f1=f1_score(ytr[sub_tr], m.predict(Xtr[sub_tr]), average="macro", zero_division=0),
                         test_f1=f1_score(yte[lab_te], m.predict(Xte[lab_te]), labels=sorted(set(yte[lab_te])),
                                          average="macro", zero_division=0)))
    return m.predict(Xte), hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASKS), required=True)
    ap.add_argument("--labels", default=r"data_processed\dmd\labels.csv")
    ap.add_argument("--landmarks", default=r"data_processed\dmd\landmarks")
    ap.add_argument("--emb-mnv2", default=r"results\emb_all_v2.npy")
    ap.add_argument("--emb-dino", default=r"results\emb_dinov2_s.npy")
    ap.add_argument("--out", default=r"results\a2")
    ap.add_argument("--skip", default="", help="через запятую: какие варианты не считать")
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    label_col, classes = TASKS[a.task]
    skip = set(s.strip() for s in a.skip.split(",") if s.strip())

    fk, XF, Ec, talk, noface = load(a, label_col, classes)
    y = fk[label_col].values.astype(object)
    people = fk["person"].values
    is_lab = np.isin(y, classes)
    persons = sorted(set(people[is_lab]) | set(noface["person"]))
    print(f"Задача {a.task}: кадров с меткой {is_lab.sum()}, без лица {len(noface)}, "
          f"разговора (только для +T) {talk.sum()}, людей {len(persons)}", flush=True)

    # варианты: (как строить признаки, модель, добавлять ли разговор в обучение)
    def emb_only(name):
        return lambda tr, te: (Ec[name][tr], Ec[name][te])

    def hybrid(name):
        def b(tr, te):
            ztr, zte = pca_block(Ec[name], tr, te)
            return np.hstack([XF[tr], ztr]), np.hstack([XF[te], zte])
        return b

    configs = {"A1_mnv2_lr": (emb_only("mnv2"), lr, False),
               "S_dino_lr": (emb_only("dino"), lr, False),
               "EB_mnv2": (hybrid("mnv2"), rf, False),
               "EB_dino": (hybrid("dino"), rf, False),
               "MLP_dino": (hybrid("dino"), None, False)}
    if a.task == "drowsy" and talk.any():
        configs["EB_mnv2+T"] = (hybrid("mnv2"), rf, True)
        configs["EB_dino+T"] = (hybrid("dino"), rf, True)
    configs = {k: v for k, v in configs.items() if k not in skip}

    y_nf, p_nf = noface[label_col].values.astype(object), noface["person"].values
    rule = np.array([NOFACE_RULE[a.task]] * len(noface), dtype=object)
    y_all, p_all = np.concatenate([y[is_lab], y_nf]), np.concatenate([people[is_lab], p_nf])
    yawn_type = None
    if a.task == "drowsy" and "yawning" in fk.columns:
        yawn_type = np.concatenate([fk["yawning"].astype(str).values[is_lab], noface["yawning"].astype(str).values])

    summary, per_person, curves, preds = [], {}, [], {}
    for name, (build, model, with_talk) in configs.items():
        ts = time.time()
        pred = np.empty(len(y), dtype=object)
        for p in persons:
            te = np.flatnonzero((people == p) & (is_lab | talk))
            tr = np.flatnonzero((people != p) & (is_lab | (talk if with_talk else False)))
            if not len(te):
                continue
            ytr = np.where(talk[tr], "alert", y[tr]).astype(object)
            Xtr, Xte = build(tr, te)
            if model is None:
                pr, hist = run_mlp_curves(Xtr, ytr, Xte, y[te], is_lab[te], classes)
                curves += [dict(person=p, **h) for h in hist]
            else:
                pr = model().fit(Xtr, ytr).predict(Xte)
            pred[te] = pr
        pr_all = np.concatenate([pred[is_lab], rule])
        preds[name] = pr_all
        f1s = macro_f1_people(y_all, pr_all, p_all, persons)
        per_person[name] = f1s
        row = dict(config=name, macro_F1=round(f1s.mean(), 3), std_people=round(f1s.std(ddof=1), 3))
        if yawn_type is not None:
            is_y = y_all == "yawning"
            for t, short in [("Yawning with hand", "yawn_hand_found_%"), ("Yawning without hand", "yawn_nohand_found_%")]:
                mt = is_y & (yawn_type == t)
                row[short] = round(100 * (pr_all[mt] == "yawning").mean(), 1) if mt.any() else np.nan
            row["false_yawn_frames"] = int(((pr_all == "yawning") & ~is_y).sum())
            if talk.any():
                done = talk & (pred != None)   # noqa: E711  (разговор людей, у которых есть разметка сонливости)
                row["talking_called_yawn_%"] = round(100 * (pred[done] == "yawning").mean(), 1)
        summary.append(row)
        print(f"  {name:12s} macro-F1 {row['macro_F1']:.3f} ± {row['std_people']:.3f}   ({(time.time() - ts) / 60:.1f} мин)",
              flush=True)

    S = pd.DataFrame(summary)
    P = pd.DataFrame(per_person, index=persons).round(3)
    P.index.name = "test_person"
    tests = []
    for a_, b_ in [("S_dino_lr", "A1_mnv2_lr"), ("EB_dino", "EB_mnv2"), ("EB_dino", "A1_mnv2_lr"),
                   ("EB_mnv2", "A1_mnv2_lr"), ("MLP_dino", "EB_dino"),
                   ("EB_dino+T", "EB_dino"), ("EB_mnv2+T", "EB_mnv2")]:
        if a_ in P and b_ in P:
            d = P[a_].values - P[b_].values
            pv = wilcoxon(P[a_], P[b_]).pvalue if np.any(d != 0) else 1.0
            tests.append(dict(comparison=f"{a_} vs {b_}", mean_diff=round(d.mean(), 3),
                              better_people=f"{int((d > 0).sum())}/{len(d)}", wilcoxon_p=round(pv, 4)))
    T = pd.DataFrame(tests)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print("\nИТОГ (macro-F1, среднее по людям):\n" + S.to_string(index=False))
    print("\nПарные сравнения по людям:\n" + T.to_string(index=False))
    S.to_csv(os.path.join(a.out, f"summary_{a.task}.csv"), index=False)
    T.to_csv(os.path.join(a.out, f"tests_{a.task}.csv"), index=False)
    P.to_csv(os.path.join(a.out, f"per_person_{a.task}.csv"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if curves:
        C = pd.DataFrame(curves)
        C.to_csv(os.path.join(a.out, f"curves_{a.task}.csv"), index=False)
        M = C.groupby("epoch")[["train_loss", "train_f1", "test_f1"]].agg(["mean", "std"])
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        ax[0].plot(M.index, M[("train_loss", "mean")], color="#1f4e79")
        ax[0].set_title("Training loss (mean over LOPO folds)"); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("log-loss")
        for col, lbl, c in [("train_f1", "train people", "#1f4e79"), ("test_f1", "held-out person", "#c55a11")]:
            m_, s_ = M[(col, "mean")], M[(col, "std")]
            ax[1].plot(M.index, m_, label=lbl, color=c)
            ax[1].fill_between(M.index, m_ - s_, m_ + s_, color=c, alpha=0.15)
        ax[1].set_title("Macro-F1 per epoch"); ax[1].set_xlabel("epoch"); ax[1].set_ylim(0, 1); ax[1].legend()
        for x in ax:
            x.grid(alpha=0.3)
        fig.suptitle(f"MLP on EB-DINOv2 features, task: {a.task}")
        fig.tight_layout(); fig.savefig(os.path.join(a.out, f"curves_{a.task}.png"), dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.barh(S["config"], S["macro_F1"], xerr=S["std_people"], color="#7f9fbf", capsize=3)
    for i, v in enumerate(S["macro_F1"]):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, 1.05); ax.set_xlabel("macro-F1 (LOPO, mean ± SD over people)"); ax.invert_yaxis()
    ax.set_title(f"A1 baseline vs DINOv2, task: {a.task}"); ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, f"bars_{a.task}.png"), dpi=200); plt.close(fig)

    for name in [n for n in ["A1_mnv2_lr", "EB_dino"] if n in preds]:
        cm = confusion_matrix(y_all, preds[name], labels=classes)
        cmn = cm / cm.sum(1, keepdims=True)
        fig, ax = plt.subplots(figsize=(3.6 + 0.6 * len(classes), 3.2 + 0.5 * len(classes)))
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, f"{100 * cmn[i, j]:.0f}%\n({cm[i, j]})", ha="center", va="center",
                        color="white" if cmn[i, j] > 0.6 else "black", fontsize=8)
        ax.set_xticks(range(len(classes)), classes, rotation=30, ha="right"); ax.set_yticks(range(len(classes)), classes)
        ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(name)
        fig.tight_layout(); fig.savefig(os.path.join(a.out, f"cm_{a.task}_{name}.png"), dpi=200); plt.close(fig)
    print(f"\nСохранено в {a.out}  | всего {(time.time() - t0) / 60:.1f} мин")


if __name__ == "__main__":
    main()
