# cd yourworkingdirectoryhere  
cp tensorrtx/yolov5/gen_wts.py yolov5  
cd yolov5  
docker run -it --rm --gpus all -v $PWD:/yolov5 tienln/ubuntu:18.04_conda /bin/bash  
cd /yolov5  
conda activate yolov5  
python gen_wts.py -w yolov5s.pt -o yolov5s.wts