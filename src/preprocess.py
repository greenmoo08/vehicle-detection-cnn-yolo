"""
차량 탐지 데이터 전처리 스크립트 (preprocess.py)
------------------------------------------------
1) 수집: COCO128(공개, YOLO 라벨 포함)에서 차량이 있는 이미지만 선별
2) 정제: 차량 클래스만 남기고 클래스 ID 재매핑 (car, bus, truck, motorcycle)
         너무 작은 박스(이미지 면적 0.05% 미만) 제거
3) 분할: train / valid (약 75:25, 고정 seed)
4) 증강: train 이미지만 좌우반전 / 밝기·대비 변화(주야간·역광 모사) / 블러(빗길·모션) 적용 → 3배
5) 리사이즈: 긴 변 640 기준 letterbox (YOLO 입력 규격)
6) CNN용 데이터: 바운딩 박스를 잘라 64x64 '차량' 패치, 차량 외 객체·배경 패치를 '비차량'으로 저장
결과: data/yolo/(images,labels)/(train,valid), data/cnn/(train,valid)/(vehicle,non_vehicle), data/stats.csv
"""
import argparse, csv, glob, os, random, shutil
import cv2
import numpy as np

COCO2NEW = {2: 0, 5: 1, 7: 2, 3: 3}          # COCO id -> 새 id
NAMES = ["car", "bus", "truck", "motorcycle"]
MIN_AREA = 0.0005                            # 박스 최소 면적 비율
IMG_SIZE = 640
SEED = 42


def read_labels(path):
    rows = []
    if os.path.exists(path):
        for line in open(path):
            p = line.split()
            if len(p) == 5:
                rows.append([int(p[0])] + [float(v) for v in p[1:]])
    return rows


def letterbox(img, size=IMG_SIZE):
    """비율을 유지한 채 긴 변을 size로 맞추고 회색(114)으로 패딩."""
    h, w = img.shape[:2]
    r = size / max(h, w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 114, np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, r, left, top, w, h


def remap_boxes(boxes, r, left, top, w, h, size=IMG_SIZE):
    out = []
    for c, x, y, bw, bh in boxes:
        X = (x * w * r + left) / size
        Y = (y * h * r + top) / size
        out.append([c, X, Y, bw * w * r / size, bh * h * r / size])
    return out


def augment(img, boxes, kind):
    if kind == "flip":
        return cv2.flip(img, 1), [[c, 1 - x, y, w, h] for c, x, y, w, h in boxes]
    if kind == "bright":   # 밝기/대비: 야간·역광 상황 모사
        a = random.choice([0.6, 1.4]); b = random.randint(-30, 30)
        return cv2.convertScaleAbs(img, alpha=a, beta=b), boxes
    if kind == "blur":     # 빗길·모션 블러 모사
        return cv2.GaussianBlur(img, (7, 7), 0), boxes
    return img, boxes


def save_yolo(img, boxes, img_path, lbl_path):
    cv2.imwrite(img_path, img)
    with open(lbl_path, "w") as f:
        for c, x, y, w, h in boxes:
            f.write(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def crop_patches(img, rows, vehicle, out_dir, stem, rng):
    h, w = img.shape[:2]
    n = 0
    for c, x, y, bw, bh in rows:
        x1, y1 = int((x - bw / 2) * w), int((y - bh / 2) * h)
        x2, y2 = int((x + bw / 2) * w), int((y + bh / 2) * h)
        if (x2 - x1) < 16 or (y2 - y1) < 16:
            continue
        patch = cv2.resize(img[max(0, y1):y2, max(0, x1):x2], (64, 64))
        cv2.imwrite(os.path.join(out_dir, f"{stem}_{n}.jpg"), patch); n += 1
    return n


def main(src, dst):
    random.seed(SEED)
    rng = np.random.default_rng(SEED)
    shutil.rmtree(dst, ignore_errors=True)
    for s in ["train", "valid"]:
        os.makedirs(f"{dst}/yolo/images/{s}", exist_ok=True)
        os.makedirs(f"{dst}/yolo/labels/{s}", exist_ok=True)
        for k in ["vehicle", "non_vehicle"]:
            os.makedirs(f"{dst}/cnn/{s}/{k}", exist_ok=True)

    imgs = sorted(glob.glob(f"{src}/images/train2017/*.jpg"))
    veh, nonveh = [], []
    for p in imgs:
        lbl = p.replace("images", "labels").replace(".jpg", ".txt")
        rows = read_labels(lbl)
        v = [[COCO2NEW[r[0]]] + r[1:] for r in rows if r[0] in COCO2NEW and r[3] * r[4] >= MIN_AREA]
        o = [r for r in rows if r[0] not in COCO2NEW and r[0] not in (1, 4, 6, 8)]
        (veh if v else nonveh).append((p, v, o))

    random.shuffle(veh)
    n_val = max(4, round(len(veh) * 0.25))
    split = {"valid": veh[:n_val], "train": veh[n_val:]}

    stats = []
    for s, items in split.items():
        for p, v, o in items:
            stem = os.path.splitext(os.path.basename(p))[0]
            img = cv2.imread(p)
            lb, r, l, t, w, h = letterbox(img)
            boxes = remap_boxes(v, r, l, t, w, h)
            save_yolo(lb, boxes, f"{dst}/yolo/images/{s}/{stem}.jpg", f"{dst}/yolo/labels/{s}/{stem}.txt")
            stats.append([s, stem, "orig", w, h, len(boxes)] + [sum(b[0] == k for b in boxes) for k in range(4)]
                         + [round(float(np.mean([b[3] * b[4] for b in boxes])), 5)])
            if s == "train":
                for kind in ["flip", "bright", "blur"]:
                    ai, ab = augment(lb, boxes, kind)
                    save_yolo(ai, ab, f"{dst}/yolo/images/{s}/{stem}_{kind}.jpg", f"{dst}/yolo/labels/{s}/{stem}_{kind}.txt")
            crop_patches(img, v, True, f"{dst}/cnn/{s}/vehicle", stem, rng)
            crop_patches(img, o, False, f"{dst}/cnn/{s}/non_vehicle", stem, rng)

    # 차량 없는 이미지 → 비차량 패치 (사람·동물·사물 등) 보충
    for i, (p, v, o) in enumerate(nonveh):
        s = "valid" if i % 4 == 0 else "train"
        crop_patches(cv2.imread(p), o, False, f"{dst}/cnn/{s}/non_vehicle", os.path.basename(p)[:-4], rng)

    # 비차량 패치가 지나치게 많으면 클래스 균형 맞추기 (차량의 2배 이내)
    for s in ["train", "valid"]:
        nv = len(os.listdir(f"{dst}/cnn/{s}/vehicle"))
        files = sorted(os.listdir(f"{dst}/cnn/{s}/non_vehicle"))
        random.shuffle(files)
        for f in files[nv * 2:]:
            os.remove(f"{dst}/cnn/{s}/non_vehicle/{f}")
        # 차량 패치도 좌우반전으로 2배 증강 (train만)
        if s == "train":
            for f in os.listdir(f"{dst}/cnn/{s}/vehicle"):
                im = cv2.imread(f"{dst}/cnn/{s}/vehicle/{f}")
                cv2.imwrite(f"{dst}/cnn/{s}/vehicle/flip_{f}", cv2.flip(im, 1))

    with open(f"{dst}/stats.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["split", "image", "type", "orig_w", "orig_h", "n_boxes"] + NAMES + ["mean_box_area"])
        wr.writerows(stats)

    with open(f"{dst}/yolo/data.yaml", "w") as f:
        f.write(f"path: {os.path.abspath(dst)}/yolo\ntrain: images/train\nval: images/valid\n"
                f"nc: 4\nnames: {NAMES}\n")

    for s in ["train", "valid"]:
        print(s, "YOLO images:", len(os.listdir(f"{dst}/yolo/images/{s}")),
              "| CNN vehicle:", len(os.listdir(f"{dst}/cnn/{s}/vehicle")),
              "non_vehicle:", len(os.listdir(f"{dst}/cnn/{s}/non_vehicle")))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="raw/coco128")
    ap.add_argument("--dst", default="data")
    a = ap.parse_args()
    main(a.src, a.dst)
