docker run \
--gpus all \
--rm \
-p9000:8000 -p9001:8001 -p9002:8002 \
-v /data/tienln/workspace/triton/yolov5_triton_inference_server/triton_deploy/models:/models \
-v /data/tienln/workspace/triton/yolov5_triton_inference_server/triton_deploy/plugins:/plugins \
--env LD_PRELOAD=/plugins/libmyplugins.so \
nvcr.io/nvidia/tritonserver:20.11-py3 tritonserver \
--model-repository=/models \
--strict-model-config=false \
# --log-verbose 1