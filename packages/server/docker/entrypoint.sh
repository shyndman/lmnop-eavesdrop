#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh

conda activate py_3.12
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

echo "ENVIRONMENT POST ACTIVATION"
echo "~~~ Variables ~~~"
env

/bin/uv run python $@
