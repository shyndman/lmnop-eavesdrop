#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
# conda activate py_3.12

conda run pip list -v
conda env list

conda activate py_3.12
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

python $@
