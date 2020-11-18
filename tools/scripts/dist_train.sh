#!/usr/bin/env bash

set -x
NGPUS=$1
PY_ARGS=${@:2}
output_dir=$3
interval=$4
model_path=$5


python -m torch.distributed.launch --nproc_per_node=${NGPUS} train.py --launcher pytorch ${PY_ARGS}
								      --extra_tag=${output_dir}
								      --ckpt_save_interval=${interval}
									  --pretrained_model=${model_path}


