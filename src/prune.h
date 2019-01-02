#ifndef __PRUNE_H__
#define __PRUNE_H__
#ifdef PRUNE

#define DP_EPSILON 0.00f

void prune_channel(float *output, int channel, int width, int height);
#endif //PRUNE

#ifdef CUDA
#include "cuda.h"
#endif //CUDA

#endif //__PRUNE_H__