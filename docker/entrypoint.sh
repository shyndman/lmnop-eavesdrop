#!/bin/bash
RUN_ID=$(python3 -c "import uuid; print(uuid.uuid4())")
echo "ENTRYPOINT START - Run ID: $RUN_ID" >&2

source /opt/conda/etc/profile.d/conda.sh

echo "!!! !!! ENVIRONMENT PRE ACTIVATION !!! !!!"
echo "~~~ Pip ~~~"
conda run pip list -v
echo "~~~ Variables ~~~"
env

conda activate py_3.12
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

echo "!!! !!! ENVIRONMENT POST ACTIVATION !!! !!!"
echo "~~~ Pip ~~~"
conda run pip list -v
echo "~~~ Variables ~~~"
env

/bin/uv run python $@
