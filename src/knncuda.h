//
// Created by aburai on 2024-10-22.
//

#ifndef OUTDET_KNNCUDA_H
#define OUTDET_KNNCUDA_H
void print_array(float *arr, int a, int b, int pitch);
__global__ void print_device_arr(float *arr, int a, int b);
__global__ void test_cuda(int val);
__global__ void compute_distances(float *ref, int ref_width, int ref_pitch,
                                  float *query, int query_width, int query_pitch,
                                  int num_feat, float *dist, size_t dist_pitch);
void call_test();
bool kNN_dist(const float *points, int num_points, const float *query, int num_query, int dim,
         float *knn_dist);
#endif //OUTDET_KNNCUDA_H
