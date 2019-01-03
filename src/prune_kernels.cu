#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"


extern "C" {
#include "cuda.h"
#include "prune.h"
#include <math.h>
}

__device__ float prune_ele_kernel(float x){return (fabs(x) <= DP_EPSILON) ? .00f : x;} //element wise prune TL190102
//TODO: FIXME:
__global__ void prune_kernel(float *output, int size_c, int size_w, int size_h)
{
    int zero_n = 0, zero_c = 0, zero_sum = 0;
    for (int k = 0; k < size_c; k++)
    {   // per channle, check if all < DP_EPSILON
        for (int i = 0; i < size_h * size_w; i++)
            if (fabs(output[k * size_h * size_w + i]) <= DP_EPSILON)
                zero_n++;
        // if so, clean this channle
        if (zero_n == size_h * size_w)
        {
            zero_c++;
            for (int i = 0; i < size_h * size_w; i++)
            output[k * size_h * size_w + i] = 0.0f;
        }
        zero_n = 0;
    }

}

extern "C" void prune_channel_gpu(float *output, int size_c, int size_w, int size_h)
{
    prune_kernel<<<1, 1>>>(output, size_c, size_w, size_h);
    check_error(cudaPeekAtLastError());
}
