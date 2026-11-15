"""
최적 YOLO 모델로 검증 이미지 예측 + 간단한 '주행 위험도' 분석
- 박스가 화면에서 차지하는 비율이 클수록 가까운 차량이라고 가정
- 화면 하단 중앙(자차 진행 경로)에 있고 면적 비율 > 5% 이면 '근접 경고'
결과: figures/14_predictions.png, results/predictions.csv
"""
import csv, glob, json, os
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO

plt.rcParams["font.family"] = "Noto Sans CJK JP"
best = json.load(open("results/yolo_best.json"))
model = YOLO(f"runs/{best['exp']}/weights/best.pt")
NAMES = ["car", "bus", "truck", "motorcycle"]
COL = [(42, 120, 214), (235, 104, 52), (27, 175, 122), (237, 161, 0)]

imgs = sorted(glob.glob("data/yolo/images/valid/*.jpg"))
rows, panels = [], []
for p in imgs:
    r = model.predict(p, imgsz=int(best["imgsz"]), conf=0.25, device="cpu", verbose=False)[0]
    im = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
    H, W = im.shape[:2]
    warn = False
    for b, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
        x1, y1, x2, y2 = map(int, b); c = int(c)
        area = (x2 - x1) * (y2 - y1) / (W * H)
        cx = (x1 + x2) / 2 / W
        in_path = 0.3 < cx < 0.7 and y2 / H > 0.5
        near = in_path and area > 0.05
        warn |= near
        cv2.rectangle(im, (x1, y1), (x2, y2), (227, 73, 72) if near else COL[c], 3)
        cv2.putText(im, f"{NAMES[c]} {s:.2f}", (x1, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COL[c], 2)
        rows.append(dict(image=os.path.basename(p), cls=NAMES[c], conf=round(s, 3), area_ratio=round(area, 4),
                         in_path=in_path, near_warning=near))
    gt = len(open(p.replace("images", "labels").replace(".jpg", ".txt")).read().split("\n")) - 1
    panels.append((im, f"{os.path.basename(p)[-7:-4]} | 정답 {gt}대 / 탐지 {len(r.boxes)}대" + (" | 근접경고" if warn else "")))

fig, axs = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4.4))
for a, (im, t) in zip(axs, panels):
    a.imshow(im); a.set_title(t, fontsize=10); a.axis("off")
fig.suptitle(f"최적 모델({best['exp']}) 검증 이미지 예측 결과 (conf ≥ 0.25)", x=0.01, ha="left")
fig.tight_layout(); fig.savefig("figures/14_predictions.png", dpi=120); plt.close()

with open("results/predictions.csv", "w", newline="") as f:
    if rows:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print(len(rows), "detections")
