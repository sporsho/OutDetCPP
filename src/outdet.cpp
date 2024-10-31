//
// Created by aburai on 2024-10-17.
//
#include<torch/torch.h>
#include "outdet.h"
void print_shape(torch::Tensor tensor, int dim){
    for (int i=0; i< dim; i++){
        std::cout << tensor.size(i) << " ";
    }
    std::cout << std::endl;
}
NHConvImpl::NHConvImpl(int kernel_size, int in_channels, int out_channels, bool bias, bool init) : ks(kernel_size),
in_channels(in_channels), out_channels(out_channels), bias(bias), init(init) {

    if (bias == true){
        b = torch::randn({out_channels});
        register_parameter("b", b);
    }
    if (init == true){
        in_channels = in_channels + 1;
        dw = torch::ones(kernel_size);
        register_parameter("dw", dw);
    }
    weight = torch::randn({in_channels, out_channels, kernel_size});
    torch::nn::init::xavier_normal_(weight);
    register_parameter("weight", weight);
}

torch::Tensor NHConvImpl::forward(torch::Tensor points, torch::Tensor dist) {
    torch::Tensor sdata;
    if (init == true){
//        std::cout << dw << std::endl;
        auto tmp = dw * dist;
        tmp = tmp.unsqueeze(-1);
        sdata = torch::cat({points, tmp}, 2);
        }
    else{
        sdata = points;
    }
    if (in_channels == 1){
        sdata = sdata.unsqueeze(-1);
    }
    auto out = torch::einsum("ijk,klj->il", {sdata, weight});
    if (bias == true){
//        std::cout << b << std::endl;
        return out + b;
    }
    else{
        return out;
    }
}
NHConvBlockImpl::NHConvBlockImpl(int kernel_size, int in_channels, int out_channels, bool init) : ks(kernel_size), in_channels(in_channels), out_channels(out_channels) {
    // declare the member var and modules
    conv1 = NHConv(kernel_size, in_channels, out_channels, true, init);
    register_module("conv1", conv1);
    bn1 = torch::nn::BatchNorm1d(out_channels);
    register_module("bn1", bn1);
    act1 = torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2));
    register_module("act1", act1);

    conv2 = NHConv(kernel_size, out_channels, out_channels, false, false);
    register_module("conv2", conv2);
    bn2 = torch::nn::BatchNorm1d(out_channels);
    register_module("bn2", bn2);
    act2 = torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2));
    register_module("act2", act2);


    if (in_channels != out_channels){
        downsample = torch::nn::Conv1d(torch::nn::Conv1dOptions(in_channels, out_channels, 1).stride(1).padding(0).bias(false));
        register_module("downsample", downsample);
    }
}

torch::Tensor NHConvBlockImpl::forward(torch::Tensor data, torch::Tensor ind, torch::Tensor dist) {
    auto inp = data.index({ind});
    auto out = conv1(inp, dist);
    out = act1(bn1(out));
    out = out.index({ind});
    out = bn2(conv2(out, dist));
    torch::Tensor ds;
    if (in_channels != out_channels){
        ds = downsample(data.unsqueeze(-1));
    }
    else{
        ds = data;
    }
    out  = out + ds.squeeze();
    out = act2(out);
    return out;
}

OutDetImpl::OutDetImpl(int num_classes, int depth, int kernel_size, int in_channels, int out_channels) : num_classes(num_classes),
depth(depth), kernel_size(kernel_size), in_channels(in_channels), out_channels(out_channels){
    tree_kernel = int(kernel_size * kernel_size);
    conv1 = NHConvBlock(tree_kernel, in_channels, out_channels, true);
    register_module("conv1", conv1);
    fc = torch::nn::Linear(torch::nn::LinearOptions(out_channels, num_classes).bias(true));
    register_module("fc", fc);
    drop = torch::nn::Dropout(0.5);
    register_module("drop", drop);
}

torch::Tensor OutDetImpl::forward(torch::Tensor points, torch::Tensor dist, torch::Tensor indices){

    auto out = conv1(points, indices, dist);
    out  = drop(out);
    out = fc(out);
    return out;
}