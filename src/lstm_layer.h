#ifndef LSTM_LAYER_H
#define LSTM_LAYER_H

#include "activations.h"
#include "layer.h"
#include "network.h"
#define USET

layer make_lstm_layer(int batch, int inputs, int outputs, int steps, int batch_normalize, int adam);
layer16 make_lstm_layer16(int batch, int inputs, int outputs, int steps, int batch_normalize, int adam);
layer16 make_lstm_layerCJ16(int batch, int inputs, int outputs, int steps, int batch_normalize, int adam ,int flag_vec);

void forward_lstm_layerCJ16(layer16 l, network16 net);
void forward_lstm_layer16(layer16 l, network16 net);
void forward_lstm_layer(layer l, network net);
 
void update_lstm_layer(layer l, update_args a);

#ifdef GPU
void forward_lstm_layer_gpu(layer l, network net);
void backward_lstm_layer_gpu(layer l, network net);
void update_lstm_layer_gpu(layer l, update_args a); 

#endif
#endif
