# Todo
* Convert 3D-OutDet PyTorch model to C++ (done)
* Data processing in CPP (on hold: using brute force solution)
* Eval in CPP (done)
* ONNX model (done)
* Weight Conversion in CPP (done)
* TensorRT + CPP: Model conversion (done)
* TensorRT + CPP: inference (done)
* ROS Integration (in 2 days)
* Docker Integration (in 1 days)

## TensorRT Engine from ONNX (image from Netron app)
![ONNX model from outdet](outdet.onnx.png "Onnx Model")


## How to Build 
``` 
mkdir build 
cd build 
cmake -DCMAKE_PREFIX_PATH=/var/local/home/aburai/outdet_cpp/libtorch -D CMAKE_CUDA_COMPILER=$(which nvcc) ..
make

```

## Docker
``` 
docker run --rm -it --entrypoint "/bin/bash" --mount src=/var/local/home/aburai/outdet_cpp,target=/outdet,type=bind roadview
```
