#ifndef DROPOUT_LAYER_H
#define DROPOUT_LAYER_H

#include "layer.h"
#include "network.h"

typedef layer dropout_layer;
typedef layer16 dropout_layer16;

dropout_layer make_dropout_layer(int batch, int inputs, float probability);
dropout_layer16 make_dropout_layer16(int batch, int inputs, float probability);

void forward_dropout_layer(dropout_layer l, network net);
void forward_dropout_layer16(dropout_layer16 l, network16 net);
void backward_dropout_layer(dropout_layer l, network net);
void resize_dropout_layer(dropout_layer *l, int inputs);

#ifdef GPU
void forward_dropout_layer_gpu(dropout_layer l, network net);
void backward_dropout_layer_gpu(dropout_layer l, network net);

#endif
#endif
