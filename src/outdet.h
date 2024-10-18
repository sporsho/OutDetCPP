//
// Created by aburai on 2024-10-17.
//
#ifndef OUTDET_CPP_OUTDET_H
#define OUTDET_CPP_OUTDET_H
#include<torch/torch.h>
struct NHConvImpl : torch::nn::Module{
    // declaration of member variables and functions
    int ks, in_channels, out_channels;
    bool bias, init;
    torch::Tensor weight, b, dw;
    NHConvImpl(int kernel_size, int in_channels, int out_channels, bool bias, bool init);
    torch::Tensor forward(torch::Tensor points, torch::Tensor dist);
};
TORCH_MODULE(NHConv);
struct NHConvBlockImpl : torch::nn::Module{
    // declaration of member variables and functions
    int ks, in_channels, out_channels;
    NHConv conv1 = nullptr;
    NHConv conv2 = nullptr;
    torch::nn::BatchNorm1d bn1 = nullptr;
    torch::nn::BatchNorm1d bn2 = nullptr;
    torch::nn::LeakyReLU act1 = nullptr;
    torch::nn::LeakyReLU act2 = nullptr;
    torch::nn::Conv1d downsample = nullptr;
    NHConvBlockImpl(int kernel_size, int in_channels, int out_channels, bool init);
    torch::Tensor forward(torch::Tensor data, torch::Tensor ind, torch::Tensor dist);
};
TORCH_MODULE(NHConvBlock);

struct OutDetImpl : torch::nn::Module{
    // declaration of member variables and modules
    int num_classes, depth, kernel_size, in_channels, out_channels, tree_kernel;
    NHConvBlock conv1 = nullptr;
    torch::nn::Linear fc = nullptr;
    torch::nn::Dropout drop = nullptr;
    OutDetImpl(int num_classes, int depth, int kernel_size, int in_channels, int out_channels);
    torch::Tensor forward(torch::Tensor points, torch::Tensor dist, torch::Tensor indices);
};
TORCH_MODULE(OutDet);
#endif //OUTDET_CPP_OUTDET_H
