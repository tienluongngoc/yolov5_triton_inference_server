FROM nvcr.io/nvidia/tensorrt:21.09-py3

WORKDIR /root
RUN apt-get update && apt-get upgrade -yy

# install cmake 3.20.0
RUN wget https://github.com/Kitware/CMake/releases/download/v3.20.0/cmake-3.20.0.tar.gz
RUN tar -zxvf cmake-3.20.0.tar.gz
WORKDIR /root/cmake-3.20.0
RUN ./bootstrap && make -j16 && make install
WORKDIR /root
RUN rm -rf cmake-3.20.0 cmake-3.20.0.tar.gz

# install opencv
RUN apt-add-repository "deb http://security.ubuntu.com/ubuntu xenial-security main" && apt-get update
RUN apt-get install libjpeg-dev libpng-dev libtiff5-dev libjasper1 libjasper-dev libdc1394-22-dev \
	libeigen3-dev libtheora-dev libvorbis-dev libxvidcore-dev libx264-dev sphinx-common \
	libtbb-dev yasm libfaac-dev libopencore-amrnb-dev libopencore-amrwb-dev libopenexr-dev \
	libgstreamer-plugins-base1.0-dev libavutil-dev libavfilter-dev libavresample-dev -y
RUN git config --global http.postBuffer 1048576000
RUN git clone https://github.com/Itseez/opencv.git && git clone https://github.com/Itseez/opencv_contrib.git
WORKDIR /root/opencv
RUN mkdir release 
WORKDIR /root/opencv/release
RUN cmake -D BUILD_TIFF=OFF -D WITH_CUDA=OFF -D ENABLE_AVX=OFF -D WITH_OPENGL=OFF \
	-D WITH_OPENCL=OFF -D WITH_IPP=OFF -D WITH_TBB=OFF -D BUILD_TBB=OFF -D WITH_EIGEN=OFF \
	-D WITH_V4L=OFF -D WITH_VTK=OFF -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF \
	-D OPENCV_GENERATE_PKGCONFIG=OFF -D CMAKE_BUILD_TYPE=RELEASE \
	-D CMAKE_INSTALL_PREFIX=/usr/local \
	-D OPENCV_EXTRA_MODULES_PATH=/root/opencv_contrib/modules /root/opencv/
RUN make -j16 && make install
RUN cp /usr/local/lib/pkgconfig/opencv4.pc  /usr/lib/x86_64-linux-gnu/pkgconfig/opencv.pc
WORKDIR /root
RUN rm -rf opencv opencv_contrib 