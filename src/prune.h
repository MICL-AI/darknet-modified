#ifndef __PRUNE_H__
#define __PRUNE_H__

#include "darknet.h"

// #ifdef PRUNE

#ifndef DP_EPSILON
#define DP_EPSILON 0.00f
#endif //EPSILON

void sparsity_stastic(const char *locate, const float *data, const int channel, const int width, const int height);
void prune_channel(float *output, const int channel, const int width, const int height);
void print_channel(float *output, const int channel, const int width, const int height);
// #endif //PRUNE

// #ifdef GPU
    // #include "cuda.h"
void prune_channel_gpu(float *output, const int size_c, const int size_w, const int size_h);
// #endif //GPU

#endif //__PRUNE_H__