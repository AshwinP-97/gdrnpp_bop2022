#!/usr/bin/env bash
set -x
this_dir=$(dirname "$0")
# commonly used opts:

# MODEL.WEIGHTS: resume or pretrained, or test checkpoint
CFG=$1
CUDA_VISIBLE_DEVICES=$2
CKPT=$4
IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
# GPUS=($(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n'))
NGPU=${#GPUS[@]}  # echo "${GPUS[0]}"
echo "use gpu ids: $CUDA_VISIBLE_DEVICES num gpus: $NGPU"
# CUDA_LAUNCH_BLOCKING=1
NCCL_DEBUG=INFO
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1

TYPE=$3
./lib/egl_renderer/compile_cpp_egl_renderer.sh

cd ./bop_renderer
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build
cd ..
if ["$TYPE" = "hb"]; then
    python $this_dir/tools/${TYPE}_bdp/${TYPE}_bdp_1_compute_fps.py
else
    python $this_dir/tools/$TYPE/${TYPE}_1_compute_fps.py
fi
#python $this_dir/tools/itodd/itodd_1_compute_fps.py
PYTHONPATH="$this_dir/../..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=$2  python $this_dir/main_gdrn.py \
    --config-file $CFG --num-gpus $NGPU --opts MODEL.WEIGHTS=$CKPT ${@:5}
