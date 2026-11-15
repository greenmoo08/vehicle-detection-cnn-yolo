"""결과 시각화: CNN 그리드 히트맵·학습곡선·혼동행렬, YOLO 실험 비교·학습곡선·예측 이미지"""
import csv, glob, json, os
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False
for k in ["axes.spines.top", "axes.spines.right"]:
    plt.rcParams[k] = False
plt.rcParams.update({"axes.edgecolor": "#9a9993", "axes.grid": True, "grid.color": "#e6e5e0",
                     "grid.linewidth": 0.8, "axes.axisbelow": True})
B, O, A = "#2a78d6", "#eb6834", "#1baf7a"
OUT = "figures"; os.makedirs(OUT, exist_ok=True)

# ---------- CNN ----------
rows = list(csv.DictReader(open("results/cnn_grid.csv")))
lrs = ["0.01", "0.001", "0.0001"]
cols = [("False", "15"), ("False", "30"), ("True", "15"), ("True", "30")]
M = np.zeros((3, 4))
for r in rows:
    i = lrs.index(str(float(r["lr"]))) if str(float(r["lr"])) in lrs else ["0.01", "0.001", "0.0001"].index(r["lr"])
    M[i, cols.index((r["augment"], r["epochs"]))] = float(r["val_acc_mean"])
fig, ax = plt.subplots(figsize=(6.4, 3.4))
im = ax.imshow(M, cmap="Blues", vmin=0.5, vmax=0.9)
ax.set_xticks(range(4), ["증강X\n15ep", "증강X\n30ep", "증강O\n15ep", "증강O\n30ep"])
ax.set_yticks(range(3), ["lr=1e-2", "lr=1e-3", "lr=1e-4"]); ax.grid(False)
for i in range(3):
    for j in range(4):
        ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", color="white" if M[i, j] > 0.75 else "#0b0b0b")
ax.set_title("CNN 최적 조건 탐색: 검증 정확도 (3 seed 평균)", loc="left")
fig.colorbar(im, ax=ax, shrink=0.8)
fig.tight_layout(); fig.savefig(f"{OUT}/08_cnn_grid_heatmap.png", dpi=160); plt.close()

h = json.load(open("results/cnn_best_history.json"))["history"]
ep = [x["epoch"] for x in h]
fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
axs[0].plot(ep, [x["train_loss"] for x in h], color=B, lw=2, label="train"); axs[0].plot(ep, [x["val_loss"] for x in h], color=O, lw=2, label="valid")
axs[0].set_title("손실(Loss)", loc="left"); axs[0].set_xlabel("epoch"); axs[0].legend(frameon=False)
axs[1].plot(ep, [x["train_acc"] for x in h], color=B, lw=2, label="train"); axs[1].plot(ep, [x["val_acc"] for x in h], color=O, lw=2, label="valid")
axs[1].set_title("정확도(Accuracy)", loc="left"); axs[1].set_xlabel("epoch"); axs[1].set_ylim(0, 1.05); axs[1].legend(frameon=False)
fig.suptitle("CNN 최적 조건 학습 곡선", x=0.02, ha="left")
fig.tight_layout(); fig.savefig(f"{OUT}/09_cnn_learning_curve.png", dpi=160); plt.close()

cm = np.array(json.load(open("results/cnn_eval.json"))["cm"])
fig, ax = plt.subplots(figsize=(3.8, 3.4))
ax.imshow(cm, cmap="Blues"); ax.grid(False)
ax.set_xticks([0, 1], ["비차량", "차량"]); ax.set_yticks([0, 1], ["비차량", "차량"])
ax.set_xlabel("예측"); ax.set_ylabel("실제")
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14, color="white" if cm[i, j] > cm.max() / 2 else "#0b0b0b")
ax.set_title("CNN 혼동행렬 (valid)", loc="left")
fig.tight_layout(); fig.savefig(f"{OUT}/10_cnn_confusion.png", dpi=160); plt.close()

# ---------- YOLO ----------
if os.path.exists("results/yolo_experiments.csv"):
    Y = list(csv.DictReader(open("results/yolo_experiments.csv")))
    names = [r["exp"].split("_", 1)[0] for r in Y]
    labels = {"E1": "사전학습X\n640·증강O", "E2": "사전학습\n320·증강X", "E3": "사전학습\n320·증강O",
              "E4": "사전학습\n640·증강X", "E5": "사전학습\n640·증강O", "E6": "사전학습\n640·증강X\n30ep",
              "E7": "E6+잦은갱신\n(nbs=8)", "E8": "사전학습\n640·증강X\n백본고정", "E9": "사전학습\n640·증강O\n백본고정"}
    def last_map(exp):
        rc = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(open(f"runs/{exp}/results.csv"))]
        return float(rc[-1]["metrics/mAP50-95(B)"])
    for r in Y:
        r["last_mAP50_95"] = last_map(r["exp"])
    x = np.arange(len(Y)); w = 0.38
    m50 = [float(r["mAP50"]) for r in Y]; m95 = [float(r["mAP50_95"]) for r in Y]
    lm = [r["last_mAP50_95"] for r in Y]
    fig, ax = plt.subplots(figsize=(10.5, 4))
    b1 = ax.bar(x - w / 2 - 0.01, m95, w, color=B, label="최고 체크포인트(best.pt) mAP50-95")
    b2 = ax.bar(x + w / 2 + 0.01, lm, w, color=O, label="마지막 에폭(last.pt) mAP50-95")
    for bars in (b1, b2):
        for bb in bars:
            ax.text(bb.get_x() + bb.get_width() / 2, bb.get_height() + 0.01, f"{bb.get_height():.2f}", ha="center", fontsize=8.5)
    ax.set_xticks(x, [n + "\n" + labels.get(n, n) for n in names], fontsize=8.5)
    ax.set_ylim(0, 1.0); ax.set_title("YOLOv8n 실험 조건별 mAP50-95 (valid): 최고 시점 vs 학습 종료 시점", loc="left")
    ax.legend(frameon=False, ncol=2, loc="upper left"); ax.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(f"{OUT}/11_yolo_experiments.png", dpi=160); plt.close()

    fig, ax = plt.subplots(figsize=(6, 3.6))
    t = [float(r["train_sec"]) for r in Y]
    m50 = m95
    ax.scatter(t, m50, s=70, color=B, edgecolor="white", linewidth=1.5, zorder=3)
    for ti, mi, n in zip(t, m50, names):
        ax.annotate(n, (ti, mi), textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xlabel("학습 시간 (초, CPU)"); ax.set_ylabel("best mAP50-95")
    ax.set_title("학습 시간 대비 성능", loc="left")
    fig.tight_layout(); fig.savefig(f"{OUT}/12_yolo_time_vs_map.png", dpi=160); plt.close()

    # 최고 실험의 학습 곡선
    # 최적 조건: 최고 체크포인트와 마지막 에폭 성능을 함께 고려(안정성)
    best = max(Y, key=lambda r: float(r["mAP50_95"]) + r["last_mAP50_95"])
    import csv as _c
    with open("results/yolo_experiments_summary.csv", "w", newline="") as f:
        wr = _c.DictWriter(f, fieldnames=list(Y[0].keys())); wr.writeheader(); wr.writerows(Y)
    rc = list(csv.DictReader(open(f"runs/{best['exp']}/results.csv")))
    rc = [{k.strip(): v for k, v in r.items()} for r in rc]
    e = [int(float(r["epoch"])) for r in rc]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
    axs[0].plot(e, [float(r["train/box_loss"]) for r in rc], color=B, lw=2, label="train box_loss")
    axs[0].plot(e, [float(r["val/box_loss"]) for r in rc], color=O, lw=2, label="val box_loss")
    axs[0].set_title("Box loss", loc="left"); axs[0].legend(frameon=False, ncol=2, loc="lower right", bbox_to_anchor=(1, 1.0)); axs[0].set_xlabel("epoch")
    axs[1].plot(e, [float(r["metrics/mAP50(B)"]) for r in rc], color=B, lw=2, label="mAP50")
    axs[1].plot(e, [float(r["metrics/mAP50-95(B)"]) for r in rc], color=O, lw=2, label="mAP50-95")
    axs[1].set_title("검증 mAP", loc="left"); axs[1].legend(frameon=False); axs[1].set_xlabel("epoch"); axs[1].set_ylim(0, 1.05)
    fig.suptitle(f"최적 조건({best['exp']}) 학습 곡선", x=0.02, ha="left")
    fig.tight_layout(); fig.savefig(f"{OUT}/13_yolo_best_curve.png", dpi=160); plt.close()
    json.dump(best, open("results/yolo_best.json", "w"), indent=1)

    # 오류 수정 전/후 비교: E4(기본 설정) vs E7(잦은 갱신) vs E8(백본 고정)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for exp, col, lab in [("E4_pre_640_noaug_15ep", O, "E4 기본 설정 (수정 전)"), ("E7_pre_640_noaug_30ep_fix", "#9a9993", "E7 nbs=8 (1차 수정)"),
                          ("E8_pre_640_noaug_15ep_freeze", B, "E8 백본 고정 + 낮은 lr (2차 수정)")]:
        rc = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(open(f"runs/{exp}/results.csv"))][:15]
        ax.plot([int(float(r["epoch"])) for r in rc], [float(r["metrics/mAP50-95(B)"]) for r in rc], color=col, lw=2, marker="o", ms=4, label=lab)
    ax.set_xlabel("epoch"); ax.set_ylabel("검증 mAP50-95"); ax.set_ylim(0, 0.85)
    ax.set_title("오류 수정 전·후 학습 안정성 비교 (15 epoch)", loc="left"); ax.legend(frameon=False, fontsize=9, loc="center right")
    fig.tight_layout(); fig.savefig(f"{OUT}/15_fix_before_after.png", dpi=160); plt.close()
print("ok", sorted(os.listdir(OUT)))
