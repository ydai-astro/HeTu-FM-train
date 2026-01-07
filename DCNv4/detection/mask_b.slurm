#!/bin/bash
#SBATCH --job-name=HeTu_mask_b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --comment=hetu_ai
#SBATCH --gres=gpu:1

#SBATCH --output=log/%x_%A_%a.out
#SBATCH --error=log/%x_%A_%a.err

ROOT="/groups/g9600009/home/share/HeTu/racsmid/test/png"
IDX=$((SLURM_ARRAY_TASK_ID + 1))

# 取第 IDX 个 SB 目录（这一层就是 images_SBxxxxx）
SB_DIR=$(ls -d "${ROOT}"/images_SB* 2>/dev/null | sort -V | sed -n "${IDX}p")
if [ -z "${SB_DIR}" ]; then
  echo "No SB dir at index ${IDX}. Maybe array range is wrong. Exit."
  exit 0
fi

# 解析 SB_ID：images_SB20147 -> 20147
SB_ID=$(basename "${SB_DIR}" | sed -E 's/^images_SB([0-9]+)$/\1/')
if [ -z "${SB_ID}" ]; then
  echo "Failed to parse SB from ${SB_DIR}. Exit."
  exit 1
fi

# 可选但推荐：检查你的真实图片目录是否存在（和 python 的 img_dir 对齐）
IMG_DIR="${SB_DIR}/lin.cmap11.0"
if [ ! -d "${IMG_DIR}" ]; then
  echo "SB=${SB_ID}: missing ${IMG_DIR}. Skip."
  exit 0
fi

echo "JobID=${SLURM_JOB_ID} Task=${SLURM_ARRAY_TASK_ID} Host=$(hostname) GPU=${CUDA_VISIBLE_DEVICES} SB=${SB_ID}"
echo "IMG_DIR=${IMG_DIR}"

python inference_dcnv4_daiyao.py --input "${SB_ID}" --batch_size 16



#sbatch -p h3c-4rtx4090 --cpus-per-task=24 --array=0-799%8 mask_b.slurm
#sbatch -p insp-8V100 --cpus-per-task=4 --array=800-1399%8 mask_b.slurm
#sbatch -p insp-4V100 --cpus-per-task=11 --array=1400-1539%4 mask_b.slurm
#sbatch -p Sougen-4V100 --cpus-per-task=7 --array=1540-1577%4 mask_b.slurm





