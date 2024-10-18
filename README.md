# Todo
* Convert 3D-OutDet PyTorch model to C++ (done)
* Data processing in CPP (in a week)
* Train / Eval in CPP (in few hours)
* Weight Conversion in CPP (in few hours)
* TensorRT + CPP (in 2 days)
* ROS Integration (in 2 days)
* Docker Integration (in 1 days)


## How to Build 
``` 
mkdir build 
cd build 
cmake -DCMAKE_PREFIX_PATH=/var/local/home/aburai/outdet_cpp/libtorch -D CMAKE_CUDA_COMPILER=$(which nvcc) ..
make

```