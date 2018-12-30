#include "avgpool_layer.h"
#include "cuda.h"
#include <stdio.h>
avgpool_layer make_avgpool_layer(int batch, int w, int h, int c)
{
    fprintf(stderr, "avg                     %4d x%4d x%4d   ->  %4d\n",  w, h, c, c);
    avgpool_layer l = {0};
    l.type = AVGPOOL;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.out_w = 1;
    l.out_h = 1;
    l.out_c = c;
    l.outputs = l.out_c;
    l.inputs = h*w*c;
    int output_size = l.outputs * batch;
    l.output =  calloc(output_size, sizeof(float));
    l.delta =   calloc(output_size, sizeof(float));
    l.forward = forward_avgpool_layer;
    l.backward = backward_avgpool_layer;
    #ifdef GPU
    l.forward_gpu = forward_avgpool_layer_gpu;
    l.backward_gpu = backward_avgpool_layer_gpu;
    l.output_gpu  = cuda_make_array(l.output, output_size);
    l.delta_gpu   = cuda_make_array(l.delta, output_size);
    #endif
    return l;
}
avgpool_layer16 make_avgpool_layer16(int batch, int w, int h, int c)
{
    fprintf(stderr, "avg                     %4d x%4d x%4d   ->  %4d\n",  w, h, c, c);    
 
    avgpool_layer16 l = {0};
    l.type = AVGPOOL;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.out_w = 1;
    l.out_h = 1;
    l.out_c = c;
    l.outputs = l.out_c;
    l.inputs = h*w*c;
    int output_size = l.outputs * batch;
    l.output =  calloc(output_size, sizeof(FLT));
    l.delta =  (FLT*) calloc(output_size, sizeof(FLT));//liuj added 20180713
    l.forward = forward_avgpool_layer16;
    //l.backward = backward_avgpool_layer;
    return l;
}

void resize_avgpool_layer(avgpool_layer *l, int w, int h)
{
    l->w = w;
    l->h = h;
    l->inputs = h*w*l->c;
}
void forward_avgpool_layer(const avgpool_layer l, network net)
{
    int b,i,k;

    for(b = 0; b < l.batch; ++b){ // slide in batch
        for(k = 0; k < l.c; ++k){ // slide in channel
            int out_index = k + b*l.c;
            l.output[out_index] = 0;
            for(i = 0; i < l.h*l.w; ++i){ // calc pooled ele
                int in_index = i + l.h*l.w*(k + b*l.c);
                l.output[out_index] += net.input[in_index];
            }
            l.output[out_index] /= l.h*l.w;
        }
    }
#ifdef PRUNE_ALL
    int zero_n = 0, zero_c = 0;
#pragma omp parallel for
    for (int k = 0; k < l.outputs * l.batch; k++)
    {
        zero_c = 0;
        if (fabs(l.output[k]) <= DP_EPSILON)
        {
            zero_c++;
            l.output[k] = 0.00f;
        }
    }
    // printf("Apoo layer, total parm: %d, saved param: %d\n", l.outputs * l.batch, zero_c);
    // printf("In summary, total load = %ld, saved = %ld\n", total_load_param += l.outputs * l.batch, total_saved_param += zero_c);
#endif
}
void forward_avgpool_layer16(const avgpool_layer16 l, network16 net)
{
printf("forward_avgpool_layer16 flag_vec = %d\n", net.flag_vec);

    int b,i,k;
    float X=0;
    int index=0;
    FLT divisor = 1.0/l.w/l.h;
   
    for(b = 0; b < l.batch; ++b){
        for(k = 0; k < l.c; ++k){
            int out_index = k + b*l.c;
            l.output[out_index] = 0;
            for(i = 0; i < l.h*l.w; ++i){
                int in_index = i + l.h*l.w*(k + b*l.c);
                l.output[out_index] += net.input[in_index] * divisor; //printf("!!!!!!!!! net.input[%d]:%f\n",in_index,net.input[in_index]);
            }
            X = l.output[out_index];
            //X /= l.h*l.w;
            l.output[out_index] = (FLT) X;  index = out_index;
        }printf("!!!!X:%f,output[%d]:%f\n",X,index,l.output[index]);
    }
}

void backward_avgpool_layer(const avgpool_layer l, network net)
{
    int b,i,k;

    for(b = 0; b < l.batch; ++b){
        for(k = 0; k < l.c; ++k){
            int out_index = k + b*l.c;
            for(i = 0; i < l.h*l.w; ++i){
                int in_index = i + l.h*l.w*(k + b*l.c);
                net.delta[in_index] += l.delta[out_index] / (l.h*l.w);
            }
        }
    }
}

