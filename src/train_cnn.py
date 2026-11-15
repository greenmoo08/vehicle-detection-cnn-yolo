"""
CNN 차량/비차량 분류기 (PyTorch)
- 입력: 64x64 RGB 패치 (preprocess.py가 만든 data/cnn)
- 모델: Conv(16)-Pool-Conv(32)-Pool-Conv(64)-GAP-FC(2)  (약 2.4만 파라미터)
- 최적 조건 탐색: 학습률 x 증강 x 에폭 그리드 → results/cnn_grid.csv
"""
import csv, glob, itertools, json, os, random, time
import cv2
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(2)
CLASSES = ["non_vehicle", "vehicle"]


def load(split):
    X, y = [], []
    for k, c in enumerate(CLASSES):
        for f in sorted(glob.glob(f"data/cnn/{split}/{c}/*.jpg")):
            im = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            X.append(im.transpose(2, 0, 1)); y.append(k)
    return torch.tensor(np.array(X)), torch.tensor(y)


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        self.f = nn.Sequential(blk(3, 16), blk(16, 32), blk(32, 64))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, 2))

    def forward(self, x):
        return self.head(self.f(x))


def augment_batch(x):
    # 학습 중 온라인 증강: 좌우반전 + 밝기 변화
    flip = torch.rand(x.size(0)) < 0.5
    x = x.clone(); x[flip] = x[flip].flip(-1)
    x = (x * (0.7 + 0.6 * torch.rand(x.size(0), 1, 1, 1))).clamp(0, 1)
    return x


def run(lr, aug, epochs, Xtr, ytr, Xva, yva, seed=0, log=False):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    m = SmallCNN()
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    hist = []
    for ep in range(epochs):
        m.train()
        idx = torch.randperm(len(Xtr))
        tl = 0
        for i in range(0, len(idx), 16):
            b = idx[i:i + 16]
            xb = augment_batch(Xtr[b]) if aug else Xtr[b]
            opt.zero_grad(); loss = lossf(m(xb), ytr[b]); loss.backward(); opt.step()
            tl += loss.item() * len(b)
        m.eval()
        with torch.no_grad():
            out = m(Xva); vl = lossf(out, yva).item()
            acc = (out.argmax(1) == yva).float().mean().item()
            tacc = (m(Xtr).argmax(1) == ytr).float().mean().item()
        hist.append(dict(epoch=ep + 1, train_loss=tl / len(Xtr), val_loss=vl, train_acc=tacc, val_acc=acc))
    return m, hist


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    Xtr, ytr = load("train"); Xva, yva = load("valid")
    print("train", Xtr.shape, "valid", Xva.shape)

    # ---- 최적 조건 탐색 (그리드) : 각 조건 3개 seed 평균 ----
    grid = list(itertools.product([1e-2, 1e-3, 1e-4], [False, True], [15, 30]))
    rows = []
    t0 = time.time()
    for lr, aug, ep in grid:
        accs = [run(lr, aug, ep, Xtr, ytr, Xva, yva, seed=s)[1][-1]["val_acc"] for s in range(3)]
        rows.append(dict(lr=lr, augment=aug, epochs=ep, val_acc_mean=np.mean(accs), val_acc_std=np.std(accs)))
        print(rows[-1])
    print("grid time %.1fs" % (time.time() - t0))
    with open("results/cnn_grid.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

    best = max(rows, key=lambda r: (r["val_acc_mean"], -r["val_acc_std"]))
    print("BEST", best)
    m, hist = run(best["lr"], best["augment"], best["epochs"], Xtr, ytr, Xva, yva, seed=0)
    with open("results/cnn_best_history.json", "w") as f:
        json.dump(dict(best=best, history=hist), f, indent=1)
    torch.save(m.state_dict(), "results/cnn_best.pt")

    # 혼동행렬 + 예측 결과 저장
    m.eval()
    with torch.no_grad():
        prob = torch.softmax(m(Xva), 1)[:, 1].numpy()
    pred = (prob > 0.5).astype(int)
    cm = [[int(((yva.numpy() == a) & (pred == b)).sum()) for b in range(2)] for a in range(2)]
    json.dump(dict(cm=cm, prob=prob.tolist(), y=yva.tolist()), open("results/cnn_eval.json", "w"))
    print("confusion", cm)
