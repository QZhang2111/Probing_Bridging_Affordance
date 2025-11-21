#!/bin/bash

# 1. 定义模型及对应层数
declare -A models
models=(
  ["dinov3_vit7b16"]="40"
  ["dino_vitb16"]="12"
  ["dinov2_vitb14"]="12"
  ["sam_vitb"]="12"
  ["clip_vitb16_laion"]="6 12"
  ["siglip_base_p16_384"]="6 12"
)

# 2. 定义不带层号的单独模型（例如 diffusion 系列）
single_models=( "sd21_up1" "sd21_up3")

# 3. 定义需要顺序执行的五个脚本
scripts=(
  "python scripts/extract_all.py"
  "python scripts/knife_roi_pipeline.py"
  #"python scripts/cross_domain_pca.py"
 # "python scripts/knife_patch_similarity.py"
 # "python scripts/cross_domain_similarity.py"
)

# 4. 运行多层模型
for model in "${!models[@]}"; do
  for layer in ${models[$model]}; do
    full_model="${model}_layer${layer}"
    echo "=============================="
    echo "Running model: ${full_model}"
    echo "=============================="
    for cmd in "${scripts[@]}"; do
      echo ">>> Executing: $cmd --model ${full_model}"
      $cmd --model ${full_model}
      if [ $? -ne 0 ]; then
        echo "Error occurred in ${cmd} for ${full_model}. Exiting."
        exit 1
      fi
    done
  done
done

# 5. 运行单层模型
for model in "${single_models[@]}"; do
  echo "=============================="
  echo "Running model: ${model}"
  echo "=============================="
  for cmd in "${scripts[@]}"; do
    echo ">>> Executing: $cmd --model ${model}"
    $cmd --model ${model}
    if [ $? -ne 0 ]; then
      echo "Error occurred in ${cmd} for ${model}. Exiting."
      exit 1
    fi
  done
done

echo "✅ All models and layers completed successfully!"