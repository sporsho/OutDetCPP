#include <torch/torch.h>
#include <iostream>
#include "src/outdet.h"
#include<string>
#include<iomanip>
#include<vector>
#include<fstream>
#include<tuple>
#include<chrono>
#include<cuda_runtime.h>
#include<helper_functions.h>
#include<helper_cuda.h>
#include<dlfcn.h>
#include<stdlib.h>
#include "src/knncuda.h"
//#include <rmm/cuda_stream_view.hpp>
//#include <cuml/common/logger.hpp>
//#include<cuml/neighbors/knn.hpp>
using namespace torch::indexing;

int main(int argc, char **argv)
{
    try{
        // set device
        auto device = torch::kCUDA;
        // single point cloud
        const char* fname =  "/var/local/home/aburai/DATA/WADS2/sequences/11/velodyne/039498.bin";
        std::ifstream fin(fname, std::ios::binary);
        fin.seekg(0, std::ios::end);
        const size_t num_elements = fin.tellg() / sizeof(float);
        fin.seekg(0, std::ios::beg);
        float data[num_elements];
        fin.read(reinterpret_cast<char*>(&data[0]), num_elements * sizeof(float));
        // initial sizes
        long a_init = long(num_elements / 4);
        long b = 4;
        auto pt_tensor = torch::from_blob(data, {a_init, 4}, torch::TensorOptions().dtype(torch::kFloat));
        torch::Tensor pt_xyz = pt_tensor.index({Slice(), Slice(None, 3)});

        auto o_dist = torch::pow(pt_xyz, 2).sum(1);
        std::cout << o_dist.sizes() << std::endl;
        auto selected_ind = torch::where(o_dist < 10 * 10)[0];
//        auto o_dist = pt
//        std::cout << selected_ind.sizes() << std::endl;
        auto selected_data = pt_tensor.index({selected_ind});
        // sizes to be used
        float *selected_data_arr = selected_data.data_ptr<float>();
        long num_feat = 3;
        int a = 30000;  // the code breaks at 40k points
        // data holder for points
        float points[a * num_feat];
        // copy points
        for (int i =0; i < a; i++){
            for (int j =0; j < 3; j ++){
                points[num_feat * i + j] = selected_data_arr[b * i + j];
            }
        }

//        float cat_arr[9][2] = {{0,0}, {1,1}, {2,2}, {3,3}, {4,4}, {5,5}, {6,6}, {7,7}, {8,8}};
        float *ref = (float*)malloc(a * num_feat * sizeof(float ));
        int k = 9;
        memcpy(ref, &points, sizeof(float) * a * num_feat);


        float *query = (float *)malloc(sizeof(float) * a * num_feat);
        memcpy(query, &points, sizeof(float) * a * num_feat);

        float *knn_dist = (float*)malloc(a *a *  sizeof(float ));

        std::cout << a_init << std::endl;

        kNN_dist(ref, a, query, a, num_feat, knn_dist);
//        print_array(knn_dist, a, a, a);
        auto options = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor dist_tensor = torch::from_blob(knn_dist, {a, a}, options);
        dist_tensor = dist_tensor.to(device);
        auto [dist, ind] = dist_tensor.topk(9, 1, false, true);
        std::cout << dist.sizes()<< " " << ind.sizes() << std::endl;
        torch::Tensor inp = pt_tensor.index({Slice(None, a)});

//        std::cout << inp << std::endl;
        // define model
        OutDet model(2, 1, 3, 4, 32);
        model->to(device);
        auto out = model->forward(inp.to(device), dist.to(device), ind.to(device));
//        std::cout << out << std::endl;


    }
    catch (const std::exception &ex){
        std::cerr << ex.what() << std::endl;
    }

    return 0;
}
