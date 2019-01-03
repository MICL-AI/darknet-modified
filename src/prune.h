#ifndef __PRUNE_H__
#define __PRUNE_H__

#include "darknet.h"

#ifdef PRUNE
#ifdef P0 //could be optimized here with Makefile
#define DP_EPSILON 0.00f
#endif
#ifdef P1
#define DP_EPSILON 0.10f
#endif
#ifdef P2
#define DP_EPSILON 0.20f
#endif
#ifdef P3
#define DP_EPSILON 0.30f
#endif
#ifdef P4
#define DP_EPSILON 0.40f
#endif
#ifdef P5
#define DP_EPSILON 0.50f
#endif
void prune_channel(float *output, int channel, int width, int height);
#endif //PRUNE

#ifdef GPU
// #include "cuda.h"
void prune_channel_gpu(float *output, int size_c, int size_w, int size_h);
#endif //GPU

#endif //__PRUNE_H__