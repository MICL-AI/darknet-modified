#ifndef AVGPOOL_LAYER_H
#define AVGPOOL_LAYER_H

#include "image.h"
#include "cuda.h"
#include "layer.h"
#include "network.h"

typedef layer avgpool_layer;
typedef layer16 avgpool_layer16;

image get_avgpool_image(avgpool_layer l);
avgpool_layer make_avgpool_layer(int batch, int w, int h, int c);
avgpool_layer16 make_avgpool_layer16(int batch, int w, int h, int c);


void resize_avgpool_layer(avgpool_layer *l, int w, int h);

void forward_avgpool_layer(const avgpool_layer l, network net);
void forward_avgpool_layer16(const avgpool_layer16 l, network16 net);

void backward_avgpool_layer(const avgpool_layer l, network net);




#ifdef GPU
void forward_avgpool_layer_gpu(avgpool_layer l, network net);
void backward_avgpool_layer_gpu(avgpool_layer l, network net);
#endif

#endif

