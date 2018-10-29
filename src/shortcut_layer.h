#ifndef SHORTCUT_LAYER_H
#define SHORTCUT_LAYER_H

#include "layer.h"
#include "network.h"

layer16 make_shortcut_layer16(int batch, int index, int w, int h, int c, int w2, int h2, int c2);
void forward_shortcut_layer16(const layer16 l, network16 net);

layer make_shortcut_layer(int batch, int index, int w, int h, int c, int w2, int h2, int c2);
void forward_shortcut_layer(const layer l, network net);


void backward_shortcut_layer(const layer l, network net);

#ifdef GPU
void forward_shortcut_layer_gpu(const layer l, network net);
void backward_shortcut_layer_gpu(const layer l, network net);
#endif

#endif
