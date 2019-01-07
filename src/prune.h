#ifndef __PRUNE_H__
#define __PRUNE_H__

#include "darknet.h"

#ifdef PRUNE

#ifndef DP_EPSILON
#define DP_EPSILON 0.00f
#endif //EPSILON

void prune_channel(float *output, int channel, int width, int height);
#endif //PRUNE

#ifdef GPU
// #include "cuda.h"
void prune_channel_gpu(float *output, int size_c, int size_w, int size_h);
#endif //GPU

#endif //__PRUNE_H__