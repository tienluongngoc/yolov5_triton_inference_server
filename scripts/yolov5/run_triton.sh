docker run \
--gpus all \
--rm \
-p9000:8000 -p9001:8001 -p9002:8002 \
-v $(pwd)/triton_deploy/models:/models \
-v $(pwd)/triton_deploy/plugins:/plugins \
--env LD_PRELOAD=/plugins/libmyplugins.so \
nvcr.io/nvidia/tritonserver:21.09-py3 tritonserver \
--model-repository=/models \
--strict-model-config=false \
--log-verbose 1