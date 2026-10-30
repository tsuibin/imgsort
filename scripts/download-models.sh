#!/usr/bin/env bash
# 下载 CLIP ViT-B/32 的 ONNX 三个文件到 ./models
# huggingface.co 在部分网络环境（含本机）不通，统一走 hf-mirror.com
set -euo pipefail

REPO="Xenova/clip-vit-base-patch32"
BASE="https://hf-mirror.com/${REPO}/resolve/main"
DIR="$(cd "$(dirname "$0")/.." && pwd)/models"

mkdir -p "$DIR"
for f in onnx/vision_model.onnx onnx/text_model.onnx tokenizer.json; do
    out="$DIR/$(basename "$f")"
    if [ -f "$out" ]; then
        echo "已存在，跳过: $out"
        continue
    fi
    echo "下载 $f -> $out"
    curl -fL --retry 3 -o "$out.part" "$BASE/$f"
    mv "$out.part" "$out"
done

echo "完成，models 目录内容："
ls -lh "$DIR"
