# Todo
* Convert 3D-OutDet PyTorch model to C++ (done)
* Data processing in CPP
* Train / Eval in CPP 
* Weight Conversion in CPP 
* TensorRT + CPP 
* ROS Integration 
* Docker Integration 


## How to Build 
``` 
cd build 
cmake -DCMAKE_PREFIX_PATH=/var/local/home/aburai/outdet_cpp/libtorch -D CMAKE_CUDA_COMPILER=$(which nvcc) ..
make

```