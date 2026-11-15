#!/usr/bin/env bash
# 전체 파이프라인 한 번에 실행
#   bash run_all.sh         → 빠른 모드: YOLO는 최적 조건(E8)만 학습 (약 5분)
#   bash run_all.sh full    → 전체 모드: YOLO 실험 9개 모두 학습 (CPU 약 30~40분)
set -e
MODE=${1:-quick}
cd "$(dirname "$0")"

if [ ! -d raw/coco128 ]; then
  echo "[1/6] COCO128 데이터 다운로드"
  curl -sL -o coco128.zip https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip
  mkdir -p raw && python -c "import zipfile; zipfile.ZipFile('coco128.zip').extractall('raw')"
fi

echo "[2/6] 전처리";            python src/preprocess.py
echo "[3/6] 데이터 시각화";      python src/visualize_data.py
echo "[4/6] CNN 학습";           python src/train_cnn.py
if [ "$MODE" = "full" ]; then
  echo "[5/6] YOLO 실험 9개 학습"; python src/train_yolo.py
  python src/visualize_results.py
else
  echo "[5/6] YOLO 최적 조건(E8) 학습"; python src/train_yolo.py --exps E8_pre_640_noaug_15ep_freeze
fi
echo "[6/6] 예측 + 근접 차량 경고"; python src/predict.py
echo "완료: figures/ 와 results/ 폴더를 확인하세요."
