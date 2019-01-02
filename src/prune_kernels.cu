#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"


extern "C" {
#include "cuda.h"
#include "prune.h"
}

__device__ float prune_ele_kernel(float x){return (x > -DP_EPSILON && x < DP_EPSILON) ? .00f : x;} //element wise prune TL190102

__global__ void prune_kernel(float *output, int size_c, int size_w, int size_h)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= size_c * size_h * size_w) return;
    // int zero_n, zero_c;
    output[i] = prune_ele_kernel(output[i]);
    // if (index < size_c * size_h * size_w)
    // {
    //     for (int k = 0; k < size_c; k++)
    //     {// per channle
    //         // check if all element < epsilon
    //         for (int i = 0; i < size_h * size_w; i++)
    //             if ((output[k * size_h * size_w + i]) <= DP_EPSILON && (output[k * size_h * size_w + i]) >= -DP_EPSILON)
    //                 zero_n++;
    //         // if so, clean this channle
    //         if (zero_n == size_h * size_w)
    //         {
    //             for (int i = 0; i < size_h * size_w; i++)
    //             {
    //                 output[i] = 0.f;
    //             }
    //             zero_c++;
    //             // l.prune[size_c] = 1;
    //         }
    //     }
    // }
    // another way to make this function...
    // int c = index % size_c;
    // index /= size_c;
    // int w = index % size_w;
    // index /= size_w;
    // int h = index;
}

extern "C" void prune_channel_gpu(float *output, int size_c, int size_w, int size_h)
{
    int num = size_c * size_h * size_w;
    prune_kernel<<<cuda_gridsize(num), BLOCK>>>(output, size_c, size_w, size_h);
    check_error(cudaPeekAtLastError());
}
