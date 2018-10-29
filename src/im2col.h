#ifndef IM2COL_H
#define IM2COL_H
#include "FP16.h"
void im2col_cpu(float* data_im,
        int channels, int height, int width,
        int ksize, int stride, int pad, float* data_col);
void im2col_cpu16(FLT* data_im,
        int channels, int height, int width,
        int ksize, int stride, int pad, FLT* data_col);
#ifdef GPU

void im2col_gpu(float *im,
         int channels, int height, int width,
         int ksize, int stride, int pad,float *data_col);

#endif
#endif
