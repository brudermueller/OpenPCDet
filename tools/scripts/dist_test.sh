#!/usr/bin/env bash

set -x
NGPUS=$1
PY_ARGS=${@:2}
batch=$3
eval_tag=$4
extra_tag=$5
ckpt=$6
# start_epoch=$6

python -m torch.distributed.launch --nproc_per_node=${NGPUS} test.py --launcher pytorch ${PY_ARGS}
                                   --batch_size=${batch}
                                   --eval_tag=${eval_tag}
                                   --extra_tag=${extra_tag}
                                   --ckpt=${ckpt}
                                   --save_to_file
                                #    --start_epoch=${start_epoch}
                                #    --eval_all

