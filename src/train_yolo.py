"""
YOLOv8n 차량 탐지 (Ultralytics) - 최적 조건 탐색
실험 조건(사전학습 여부, 증강 데이터 사용 여부, 입력 크기, 에폭)을 바꿔가며 학습하고
mAP50 / mAP50-95 / precision / recall / 학습시간을 results/yolo_experiments.csv 에 저장
"""
import argparse, csv, glob, json, os, time
from ultralytics import YOLO

ROOT = os.path.abspath("data/yolo")
NAMES = ["car", "bus", "truck", "motorcycle"]

EXPS = {
    # 이름: (가중치, 증강 데이터 사용, imgsz, epochs)
    "E1_scratch_640_aug_15ep": ("yolov8n.yaml", True, 640, 15),
    "E2_pre_320_noaug_15ep":   ("yolov8n.pt", False, 320, 15),
    "E3_pre_320_aug_15ep":     ("yolov8n.pt", True, 320, 15),
    "E4_pre_640_noaug_15ep":   ("yolov8n.pt", False, 640, 15),
    "E5_pre_640_aug_15ep":     ("yolov8n.pt", True, 640, 15),
    "E6_pre_640_noaug_30ep":   ("yolov8n.pt", False, 640, 30),
    # 2차 탐색: 소량 데이터에서는 기본값 nbs=64 때문에 가중치가 4에폭에 1번만 갱신되고,
    #           lr0=0.01이 사전학습 특징을 망가뜨리는 문제 발견 → nbs=8, lr0=0.001로 수정
    "E7_pre_640_noaug_30ep_fix": ("yolov8n.pt", False, 640, 30, dict(nbs=8, lr0=0.001, warmup_epochs=1)),
    # 3차 탐색: 갱신을 자주 해도 오히려 성능 붕괴 → 원인은 사전학습 특징 손상(망각)
    #           → 백본 10개 층을 고정(freeze)하고 SGD 낮은 학습률로 헤드만 미세조정
    "E8_pre_640_noaug_15ep_freeze": ("yolov8n.pt", False, 640, 15, dict(freeze=10, optimizer="SGD", lr0=0.001, nbs=8, warmup_epochs=0)),
    "E9_pre_640_aug_15ep_freeze":   ("yolov8n.pt", True, 640, 15, dict(freeze=10, optimizer="SGD", lr0=0.001, nbs=8, warmup_epochs=0)),
}


def make_yaml(aug):
    tr = sorted(glob.glob(f"{ROOT}/images/train/*.jpg"))
    if not aug:
        tr = [p for p in tr if "_" not in os.path.basename(p)]
    lst = f"{ROOT}/train_{'aug' if aug else 'orig'}.txt"
    open(lst, "w").write("\n".join(tr) + "\n")
    y = f"{ROOT}/data_{'aug' if aug else 'orig'}.yaml"
    open(y, "w").write(f"train: {lst}\nval: {ROOT}/images/valid\nnc: 4\nnames: {NAMES}\n")
    return y, len(tr)


def main(names):
    os.makedirs("results", exist_ok=True)
    out = "results/yolo_experiments.csv"
    done = {}
    if os.path.exists(out):
        done = {r["exp"]: r for r in csv.DictReader(open(out))}
    for n in names:
        w, aug, sz, ep, *extra = EXPS[n]
        extra = extra[0] if extra else {}
        y, ntr = make_yaml(aug)
        t0 = time.time()
        m = YOLO(w)
        m.train(data=y, imgsz=sz, epochs=ep, batch=8, device="cpu", workers=0, seed=0,
                project=os.path.abspath("runs"), name=n, exist_ok=True, plots=True, verbose=False,
                augment=True, mosaic=1.0, close_mosaic=3, patience=50, **extra)
        tt = time.time() - t0
        best = YOLO(f"runs/{n}/weights/best.pt")
        r = best.val(data=y, imgsz=sz, device="cpu", workers=0, plots=False, verbose=False)
        done[n] = dict(exp=n, weights=w, aug=aug, n_train=ntr, imgsz=sz, epochs=ep,
                       precision=round(r.box.mp, 4), recall=round(r.box.mr, 4),
                       mAP50=round(r.box.map50, 4), mAP50_95=round(r.box.map, 4),
                       train_sec=round(tt, 1), extra=json.dumps(extra))
        print(done[n])
        with open(out, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(done[n].keys())); wr.writeheader(); wr.writerows(done.values())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", nargs="*", default=list(EXPS))
    main(ap.parse_args().exps)
