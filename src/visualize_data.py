"""데이터 시각화(EDA): 클래스 분포, 이미지당 객체 수, 박스 크기 분포, 라벨 샘플, 증강 예시, CNN 패치 샘플"""
import csv, glob, os
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False
for k in ["axes.spines.top", "axes.spines.right"]:
    plt.rcParams[k] = False
plt.rcParams["axes.edgecolor"] = "#9a9993"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.color"] = "#e6e5e0"
plt.rcParams["grid.linewidth"] = 0.8
plt.rcParams["axes.axisbelow"] = True

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # car, bus, truck, motorcycle
NAMES = ["car", "bus", "truck", "motorcycle"]
D = "data"; OUT = "figures"
os.makedirs(OUT, exist_ok=True)


def labels(split):
    rows = []
    for f in glob.glob(f"{D}/yolo/labels/{split}/*.txt"):
        for l in open(f):
            c, x, y, w, h = l.split()
            rows.append((int(c), float(w), float(h), os.path.basename(f)))
    return rows


orig_imgs = [f for f in glob.glob(f"{D}/yolo/images/*/*.jpg") if "_" not in os.path.basename(f)]
orig = [r for r in labels("train") if "_" not in r[3]] + labels("valid")

# 1) 클래스 분포
cnt = [sum(r[0] == k for r in orig) for k in range(4)]
fig, ax = plt.subplots(figsize=(6, 3.6))
b = ax.bar(NAMES, cnt, color=C, width=0.6)
for bar, v in zip(b, cnt):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.8, str(v), ha="center", color="#0b0b0b")
ax.set_title(f"클래스별 객체 수 (원본 {len(orig_imgs)}장)", loc="left")
ax.set_ylabel("객체 수"); ax.grid(axis="x", visible=False)
fig.tight_layout(); fig.savefig(f"{OUT}/01_class_distribution.png", dpi=160); plt.close()

# 2) 박스 크기(면적 비율) 분포 - 작은 객체가 많은지 확인
areas = np.array([r[1] * r[2] for r in orig])
fig, ax = plt.subplots(figsize=(6, 3.6))
ax.hist(np.sqrt(areas), bins=15, color="#2a78d6", edgecolor="white", linewidth=1.5)
ax.axvline(np.sqrt(32 * 32 / 640 / 640), color="#52514e", ls="--", lw=1.2)
ax.text(np.sqrt(32 * 32 / 640 / 640) + 0.01, ax.get_ylim()[1] * 0.85, "32px (소형 객체 기준)", color="#52514e", fontsize=9)
ax.set_title("바운딩 박스 크기 분포 (√면적 비율)", loc="left")
ax.set_xlabel("√(박스 면적 / 이미지 면적)"); ax.set_ylabel("박스 수"); ax.grid(axis="x", visible=False)
fig.tight_layout(); fig.savefig(f"{OUT}/02_box_size_hist.png", dpi=160); plt.close()

# 3) 박스 가로/세로 산점도 (클래스별)
fig, ax = plt.subplots(figsize=(5, 4.2))
for k in range(4):
    p = [(r[1] * 640, r[2] * 640) for r in orig if r[0] == k]
    if p:
        p = np.array(p)
        ax.scatter(p[:, 0], p[:, 1], s=36, color=C[k], label=NAMES[k], edgecolor="white", linewidth=1, alpha=0.9)
ax.set_xlabel("박스 너비 (px, 640 기준)"); ax.set_ylabel("박스 높이 (px)")
ax.set_title("클래스별 박스 크기", loc="left"); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(f"{OUT}/03_box_wh_scatter.png", dpi=160); plt.close()

# 4) 데이터셋 구성 (전처리 전/후)
nt = len(os.listdir(f"{D}/yolo/images/train")); nv = len(os.listdir(f"{D}/yolo/images/valid"))
fig, ax = plt.subplots(figsize=(6, 3.2))
cats = ["원본 COCO128", "차량 이미지 선별", "train (증강 후)", "valid"]
vals = [128, len(orig_imgs), nt, nv]
b = ax.barh(cats[::-1], vals[::-1], color="#2a78d6", height=0.55)
for bar, v in zip(b, vals[::-1]):
    ax.text(v + 1.5, bar.get_y() + bar.get_height() / 2, str(v), va="center")
ax.set_title("전처리 단계별 이미지 수", loc="left"); ax.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig(f"{OUT}/04_pipeline_counts.png", dpi=160); plt.close()


# 5) 라벨 샘플 시각화 (박스 그려보기 = 라벨 검증)
def draw(img_path, lbl_path):
    im = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    h, w = im.shape[:2]
    for l in open(lbl_path):
        c, x, y, bw, bh = map(float, l.split()); c = int(c)
        col = tuple(int(C[c][i:i + 2], 16) for i in (1, 3, 5))
        x1, y1, x2, y2 = int((x - bw / 2) * w), int((y - bh / 2) * h), int((x + bw / 2) * w), int((y + bh / 2) * h)
        cv2.rectangle(im, (x1, y1), (x2, y2), col, 3)
        cv2.putText(im, NAMES[c], (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
    return im

ims = sorted(f for f in glob.glob(f"{D}/yolo/images/train/*.jpg") if "_" not in os.path.basename(f))[:6]
fig, axs = plt.subplots(2, 3, figsize=(10, 6.6))
for a, p in zip(axs.flat, ims):
    a.imshow(draw(p, p.replace("images", "labels").replace(".jpg", ".txt"))); a.axis("off")
fig.suptitle("전처리 후 라벨 시각화 (640×640 letterbox)", x=0.02, ha="left")
fig.tight_layout(); fig.savefig(f"{OUT}/05_label_samples.png", dpi=130); plt.close()

# 6) 증강 예시
base = ims[0][:-4]
fig, axs = plt.subplots(1, 4, figsize=(12, 3.4))
for a, (suf, t) in zip(axs, [("", "원본"), ("_flip", "좌우반전"), ("_bright", "밝기/대비 (야간·역광)"), ("_blur", "블러 (빗길·모션)")]):
    p = base + suf + ".jpg"
    a.imshow(draw(p, p.replace("images", "labels").replace(".jpg", ".txt"))); a.set_title(t); a.axis("off")
fig.tight_layout(); fig.savefig(f"{OUT}/06_augmentation.png", dpi=130); plt.close()

# 7) CNN 패치 샘플
fig, axs = plt.subplots(2, 8, figsize=(12, 3.4))
for r, k in enumerate(["vehicle", "non_vehicle"]):
    fs = sorted(glob.glob(f"{D}/cnn/train/{k}/*.jpg"))[:8]
    for a, f in zip(axs[r], fs):
        a.imshow(cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)); a.axis("off")
    axs[r][0].set_title("차량" if k == "vehicle" else "비차량", loc="left")
fig.suptitle("CNN 입력 패치 (64×64)", x=0.02, ha="left")
fig.tight_layout(); fig.savefig(f"{OUT}/07_cnn_patches.png", dpi=130); plt.close()
print("saved", sorted(os.listdir(OUT)))
