# cd yourworkingdirectoryhere  
mkdir -p triton_deploy/models/yolov5/1/  
mkdir -p triton_deploy/plugins  
cp tensorrtx/yolov5/yolov5s.engine triton_deploy/models/yolov5/1/model.plan  
cp tensorrtx/yolov5/build/libmyplugins.so triton_deploy/plugins/libmyplugins.so  