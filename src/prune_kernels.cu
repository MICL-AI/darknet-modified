#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"


extern "C" {
#include "cuda.h"
#include "prune.h"
#include <math.h>
}

__device__ float prune_ele_kernel(float x){return (fabs(x) <= DP_EPSILON) ? .00f : x;} //element wise prune TL190102
__global__ void prune_kernel(float *output, int size_c, int size_w, int size_h)
{
    int k = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (k >= size_c) return;
    long int fmap_total_reduce_gpu = 0,fmap_total_load_gpu = 0;    
    int zero_n = 0;
    // per channle, check if all < DP_EPSILON
    for (int i = 0; i < size_h * size_w; i++)
        if (fabs(output[k * size_h * size_w + i]) <= DP_EPSILON)
            zero_n++;
    // if so, clean this channle
    if (zero_n == size_h * size_w)
    {
        // cudaMemset(&output[k * size_h * size_w], 0x0, size_h * size_w * sizeof(float));  // __host__ cudaMenset can not be called in __global__
        fmap_total_reduce_gpu += size_c * size_w * size_h;
        for (int i = 0; i < size_h * size_w; i++)
        output[k * size_h * size_w + i] = 0.0f;
    }
    fmap_total_load_gpu += size_c * size_h * size_w;
    zero_n = 0;
    
}

extern "C" void prune_channel_gpu(float *output, int size_c, int size_w, int size_h)
{
    prune_kernel<<<cuda_gridsize(size_c), BLOCK>>>(output, size_c, size_w, size_h);
    check_error(cudaPeekAtLastError());
}
