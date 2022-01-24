# YOLOV5 Triton Inferece Server Using Tensorrt

First of all, I would like to thank [wang-xinyu](https://github.com/wang-xinyu/tensorrtx) and [ultralytics](https://github.com/ultralytics/yolov5). My repo was heavily based on both of two repoes.  
This repo implemented YOLOV5 based on Tensorrt engine and Triton Inference Server

# How to run
## Dowload docker image to create engine file
```
docker pull tienln/tensorrt:8.0.3_opencv 
docker pull tienln/ubuntu:18.04_conda
docker pull nvcr.io/nvidia/tritonserver:21.09-py3
```
## Clone code base from git
Open new terminal 
```
cd yourworkingdirectoryhere  
git clone -b v5.0 https://github.com/ultralytics/yolov5.git
git clone -b yolov5-v5.0 https://github.com/wang-xinyu/tensorrtx.git

```
## Create *.wts file  
Open new terminal 
```
cd yourworkingdirectoryhere  
cp tensorrtx/yolov5/gen_wts.py yolov5  
cd yolov5  
docker run -it --rm --gpus all -v $PWD:/yolov5 tienln/ubuntu:18.04_conda /bin/bash  
cd /yolov5  
conda activate yolov5  
python gen_wts.py -w yolov5s.pt -o yolov5s.wts
```
## Create engine (TRT *.engine) engine file  
Open new terminal  
```
cd yourworkingdirectoryhere 
cp yolov5/yolov5s.wts tensorrtx/yolov5
cd tensorrtx/yolov5  
docker run -it --rm --gpus all -v $PWD:/yolov5 tienln/tensorrt:8.0.3_opencv /bin/bash   
cd /yolov5
mkdir build  
cd build   
cmake ..  
make -j16  
./yolov5 -s ../yolov5s.wts ../yolov5s.engine s  
```
## Create Triton Inference Server  
Open new terminal
```
cd yourworkingdirectoryhere  
mkdir -p triton_deploy/models/yolov5/1/  
mkdir triton_deploy/plugins  
cp tensorrtx/yolov5/yolov5s.engine triton_deploy/models/yolov5/1/model.plan  
cp tensorrtx/yolov5/build/libmyplugins.so triton_deploy/plugins/libmyplugins.so  
```


Run Triron
```
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
```

## Run in client  

Open new terminal
```
cd yourworkingdirectoryhere   
cd clients/yolov5
docker run -it --rm --gpus all --network host -v $PWD:/client tienln/ubuntu:18.04_conda /bin/bash  
conda activate yolov5  
pip install tritonclient  
cd /client
python client.py -o data/dog_result.jpg image data/dog.jpg  
```