#!/bin/bash
# Phase 1: RENO 多 posQ 压缩扫描
# posQ 控制量化粒度: 越小=越精细=越多比特=越高质量
# 用法: bash tools/run_reno_sweep.sh <scene_id> [ckpt_path]
# 例如: bash tools/run_reno_sweep.sh 001 ./model/KITTIDetection/ckpt.pt

set -e

SCENE=${1:-001}
CKPT=${2:-./model/KITTIDetection/ckpt.pt}
INPUT_DIR="data/reno_input/${SCENE}"
BASE_OUTPUT="data/reno_output"
BASE_DECODED="data/reno_decoded"

echo "=== RENO posQ Sweep for Scene ${SCENE} ==="

# 检查输入
if [ ! -d "$INPUT_DIR" ] || [ -z "$(ls ${INPUT_DIR}/*.ply 2>/dev/null)" ]; then
    echo "[ERROR] No PLY files found in: $INPUT_DIR"
    echo "Run: python tools/pandaset_to_ply.py --scene_dir pandaset/${SCENE} --output_dir ${INPUT_DIR}"
    exit 1
fi

cd RENO

# posQ sweep: 8(finest) -> 16 -> 32 -> 64 -> 128(coarsest)
for POSQ in 8 16 32 64 128; do
    echo ""
    echo "--- posQ: ${POSQ} ---"

    OUTPUT_DIR="../${BASE_OUTPUT}/${SCENE}_posQ${POSQ}"
    DECODED_DIR="../${BASE_DECODED}/${SCENE}_posQ${POSQ}"

    mkdir -p "$OUTPUT_DIR" "$DECODED_DIR"

    # 压缩
    echo "Compressing (posQ=${POSQ})..."
    python compress.py \
        --input_glob "../${INPUT_DIR}/*.ply" \
        --output_folder "$OUTPUT_DIR" \
        --posQ ${POSQ} \
        --ckpt ${CKPT}

    # 解压
    echo "Decompressing..."
    python decompress.py \
        --input_glob "${OUTPUT_DIR}/*.bin" \
        --output_folder "$DECODED_DIR" \
        --ckpt ${CKPT}

    echo "Done: posQ=${POSQ} -> ${DECODED_DIR}"
done

# 评估几何质量
echo ""
echo "=== Evaluating geometry ==="
for POSQ in 8 16 32 64 128; do
    DECODED_DIR="../${BASE_DECODED}/${SCENE}_posQ${POSQ}"
    echo "--- posQ=${POSQ} ---"
    python eval.py \
        --input_glob "../${INPUT_DIR}/*.ply" \
        --decompressed_path "$DECODED_DIR" \
        --pcc_metric_path ./third_party/pc_error_d \
        --resolution 59.70
done

cd ..
echo ""
echo "=== RENO sweep complete ==="
