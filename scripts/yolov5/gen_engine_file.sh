# cd yourworkingdirectoryhere 
cp yolov5/yolov5s.wts tensorrtx/yolov5
cd tensorrtx/yolov5  
docker run -it --rm --gpus all -v $PWD:/yolov5 tienln/tensorrt:8.0.3_opencv /bin/bash   
cd /yolov5
mkdir build  
cd build   
cmake ..  
make -j16  
./yolov5 -s ../yolov5s.wts ../yolov5s.engine s  