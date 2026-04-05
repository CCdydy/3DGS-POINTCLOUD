#!/bin/bash
# Phase 3: SplatAD Sensitivity Curve 批量训练+评估
# 用法: bash tools/run_splatad_sensitivity.sh <pandaset_scene_path>
# 例如: bash tools/run_splatad_sensitivity.sh /path/to/pandaset/001

set -e

PANDASET_SCENE=${1:?"Usage: $0 <pandaset_scene_path>"}
SCENE_NAME=$(basename "$PANDASET_SCENE")
NEURAD_DIR="neurad-studio"
MAX_ITER=30000

echo "=== SplatAD Sensitivity Sweep for ${SCENE_NAME} ==="

cd "$NEURAD_DIR"

# posQ values: 8(finest), 16(default), 32, 64, 128(coarsest), raw(uncompressed)
for POSQ in 8 16 32 64 128 raw; do
    echo ""
    echo "--- posQ: ${POSQ} ---"

    if [ "$POSQ" = "raw" ]; then
        DATA_PATH="$PANDASET_SCENE"
    else
        DATA_PATH="../data/pandaset_compressed/${SCENE_NAME}_posQ${POSQ}"
    fi

    if [ ! -d "$DATA_PATH" ]; then
        echo "[WARN] Data path not found, skipping: $DATA_PATH"
        continue
    fi

    OUTPUT_NAME="sensitivity_${SCENE_NAME}_posQ${POSQ}"

    # 训练 SplatAD
    echo "Training SplatAD (${MAX_ITER} iters)..."
    ns-train splatad \
        pandaset-dataparser \
        --data "$DATA_PATH" \
        --max-num-iterations ${MAX_ITER} \
        --output-dir "./outputs/${OUTPUT_NAME}"

    # 找到最新的 config.yml
    CONFIG=$(find "./outputs/${OUTPUT_NAME}" -name "config.yml" | head -1)
    if [ -z "$CONFIG" ]; then
        echo "[ERROR] No config.yml found for ${POSQ}"
        continue
    fi

    # 评估
    echo "Evaluating..."
    ns-eval \
        --load-config "$CONFIG" \
        --output-path "../results/sensitivity_${POSQ}.json"

    echo "Done: posQ=${POSQ} -> results/sensitivity_${POSQ}.json"
done

cd ..

echo ""
echo "=== Sensitivity sweep complete ==="
echo "Run: python tools/plot_sensitivity.py"
