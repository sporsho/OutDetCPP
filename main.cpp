#include <torch/torch.h>
#include <iostream>
#include "src/outdet.h"

int main()
{
    torch::Device device(torch::kCPU);
    torch::Tensor tensor = torch::zeros({2, 2});
    std::cout << tensor << std::endl;


    if (torch::cuda::is_available()) {
      std::cout << "CUDA is available! " << std::endl;
      device = torch::kCUDA;
    }

    torch::Tensor test_gpu_tensor = tensor.to(device);

    std::cout << test_gpu_tensor << std::endl;

    OutDet conv(2, 1, 3, 4,  32);
    conv->to(device);
    auto points = torch::randn({10, 4});
    auto dist = torch::randn({10, 3});
    auto ind = torch::randint(0, 10, {10, 3}).to(torch::kLong);

    auto out = conv->forward(points.to(device), dist.to(device),  ind.to(device));
    std::cout << out << std::endl;
    torch::save(conv, "tmp.pt");
//    auto data_loader = torch::data::make_data_loader(
//            torch::data::datasets::MNIST("./data").map(
//                    torch::data::transforms::Stack<>()
//                    ), 64
//            );
//    for (size_t epoch = 0; epoch < 10; epoch++){
//        size_t batch_index = 0;
//        for (auto& batch : *data_loader){
//            auto data = batch.data.to(device);
//            torch::Tensor pred = net.forward(data);
//            std::cout << pred.size(0) << " x " << pred.size(1) << std::endl;
//            break;
//        }
//    }
    return 0;
}
