//
// Created by aburai on 2024-10-22.
//
#include<stdio.h>
#include<assert.h>
#include<cuda_runtime.h>
#include<iostream>
#include<cmath>
#include "knncuda.h"
#define BLOCK_DIM 32
__device__ void acquire_lock(int *mutex){
    while (atomicCAS(mutex, 0, 1) != 0);

}
__device__ void release_lock(int *mutex){
    atomicExch(mutex, 0);
}
void print_array(float *arr, int a, int b, int pitch){
    for (int i = 0; i < a; i++){
        for (int j = 0; j< b; j++){
             printf("%f ", arr[i * pitch + j]);
        }
        printf("\n");
    }
}
__global__ void print_device_array(float *arr, int a, int b, size_t pitch){
    for (int i = 0; i < a; i++){
        float * row = (float*)((char*)arr + i * pitch);
        for (int j = 0; j< b; j++){
            printf("%f ", row[j]);
        }
        printf("\n");
    }
}
__global__ void test_cuda(int val){
    printf("[%d, %d]:\t\tValue is:%d\n", blockIdx.y * gridDim.x + blockIdx.x,
           threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
           threadIdx.x,
           val);
}

__global__ void compute_distances(float *ref, int ref_width, int ref_pitch,
                                  float *query, int query_width, int query_pitch,
                                  int num_feat, float *dist, size_t dist_pitch){
    int q_index = blockIdx.x * gridDim.x + threadIdx.x;
        int r_index = blockIdx.y * gridDim.y + threadIdx.y;
        if (q_index < query_width && r_index < ref_width){

            float *row_ref = (float*)((char*)ref + r_index * ref_pitch);
            float *row_q = (float*)((char*)query + q_index * query_pitch);
            float *row_dist = (float*)((char*)dist + q_index * dist_pitch);
            float ssd = 0.0f;
            for (int i =0; i < num_feat; i++){
                float diff = row_ref[i] - row_q[i];
                ssd += diff * diff;
            }
//            printf("%d %d %f\n", r_index, q_index, ssd);
            row_dist[r_index] = ssd;
        }
    }
void call_test(){
    dim3 dimGrid(2, 2);
    dim3 dimBlock(2, 2, 2);
    test_cuda<<<dimGrid, dimBlock>>>(10);
    cudaDeviceSynchronize();
}
bool kNN_dist(const float *points, int num_points, const float *query, int num_query, int dim,
         float *knn_dist){
//    printf("Given hp: dim: %d, k: %d, num q: %d, num data: %d\n", dim, k, num_query, num_points);
//    print_array(knn_dist, 2,k,k);
    cudaError_t err0, err1, err2, err3, err4;
    int nb_devices;
    err0 = cudaGetDeviceCount(&nb_devices);
    if (err0 != cudaSuccess || nb_devices == 0){
        std::cout << "Cuda device not found" << std::endl;
        return false;
    }
    err0 = cudaSetDevice(0);
    if (err0 != cudaSuccess){
        std::cout << "Cannot select cuda device" << std::endl;
        return false;
    }
    // memory allocation
    float * ref_dev = NULL;
    float * query_dev = NULL;
    float * dist_dev = NULL;
    int * index_dev = NULL;
    size_t ref_pitch;
    size_t query_pitch;
    size_t dist_pitch;
    size_t index_pitch;
    size_t holder_pitch;
    err0 = cudaMallocPitch((void **)&ref_dev, &ref_pitch, dim * sizeof(float), num_points);
    err1 = cudaMallocPitch((void **)&query_dev, &query_pitch, dim * sizeof(float), num_query);
    err2 = cudaMallocPitch((void **)&dist_dev, &dist_pitch, num_points * sizeof(float), num_query);
//    err3 = cudaMallocPitch((void **)&index_dev, &index_pitch, num_points * sizeof(int), num_query);
    if (err0 != cudaSuccess ){

        std::cout << "Cannot allocate gpu memory refdev" << std::endl;
        cudaFree(ref_dev);
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev);
        return false;
    }
    if ( err1 != cudaSuccess ){

        std::cout << "Cannot allocate gpu memory query dev" << std::endl;
        cudaFree(ref_dev);
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev);
        return false;
    }
    if ( err2 != cudaSuccess ){

        std::cout << "Cannot allocate gpu memory dist dev" << std::endl;
        cudaFree(ref_dev);
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev);
        return false;
    }
//
    err0 = cudaMemcpy2D(ref_dev,   ref_pitch,   points,   dim * sizeof(float ),   dim * sizeof(float ),   num_points, cudaMemcpyHostToDevice);
    err1 = cudaMemcpy2D(query_dev,   query_pitch, query,   dim * sizeof(float),   dim * sizeof(float ),   num_query, cudaMemcpyHostToDevice);
//    err2 = cudaMemcpy2D(dist_dev, dist_pitch, knn_dist, num_points * sizeof(float ), num_points * sizeof(float ), num_query, cudaMemcpyHostToDevice);
    if (err0 != cudaSuccess || err1 != cudaSuccess ){
        std::cout << "Cannot copy gpu memory" << err0 << " "<< err1<< " " <<std::endl;
        cudaFree(ref_dev);
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev);
        return false;
    }
 // calculate distance
    dim3 block0(BLOCK_DIM, BLOCK_DIM, 1);
    int gridx = int(ceil((float)num_query / BLOCK_DIM));
    int gridy = int(ceil((float)num_points / BLOCK_DIM));
    dim3 grid0(gridx, gridy, 1);
//    std::cout << gridx << " " << gridy << std::endl;
//    print_device_array<<<1,1>>>(dist_dev, 9, 9, dist_pitch);
    compute_distances<<<grid0, block0>>>(ref_dev, num_points, (int)ref_pitch, query_dev, num_query, (int)query_pitch, dim, dist_dev, dist_pitch);
    cudaDeviceSynchronize();


//    print_device_array<<<1,1>>>(dist_dev, 2, 5, dist_pitch);
    err1 = cudaMemcpy2D(knn_dist,   num_points * sizeof(float), dist_dev,   dist_pitch,   num_points * sizeof(float ),   num_query, cudaMemcpyDeviceToHost);
    if (err1 != cudaSuccess){
        std::cout << "Cannot copy dist to cpu" << err1 << std::endl;
        return false;
    }
//    print_array(knn_dist, 2, 5, num_points);
    return true;
}
