# cd yourworkingdirectoryhere   
cd clients/yolov5
docker run -it --rm --gpus all --network host -v $PWD:/client tienln/ubuntu:18.04_conda /bin/bash  
conda activate yolov5  
pip install tritonclient  
cd /client
python client.py -o data/dog_result.jpg image data/dog.jpg  