#ifndef SOFTMAX_LAYER_H
#define SOFTMAX_LAYER_H
#include "layer.h"
#include "network.h"

typedef layer softmax_layer;
typedef layer16 softmax_layer16;
softmax_layer16 make_softmax_layer16(int batch, int inputs, int groups);
softmax_layer16 make_softmax_layerCJ16(int batch, int inputs, int groups );

void forward_softmax_layer16(const softmax_layer16 l, network16 net);
void forward_softmax_layerCJ16(const softmax_layer16 l, network16 net);

void softmax_array(float *input, int n, float temp, float *output);
softmax_layer make_softmax_layer(int batch, int inputs, int groups);
void forward_softmax_layer(const softmax_layer l, network net);
void backward_softmax_layer(const softmax_layer l, network net);

#ifdef GPU
void pull_softmax_layer_output(const softmax_layer l);
void forward_softmax_layer_gpu(const softmax_layer l, network net);
void backward_softmax_layer_gpu(const softmax_layer l, network net);
#endif

#endif
