#include "shortcut_layer.h"
#include "cuda.h"
#include "blas.h"
#include "activations.h"

#include <stdio.h>
#include <assert.h>

layer make_shortcut_layer(int batch, int index, int w, int h, int c, int w2, int h2, int c2)
{
    fprintf(stderr, "Shortcut Layer: %d\n", index);
    layer l = {0};
    l.type = SHORTCUT;
    l.batch = batch;
    l.w = w2;
    l.h = h2;
    l.c = c2;
    l.out_w = w;
    l.out_h = h;
    l.out_c = c;
    l.outputs = w * h * c;
    l.inputs = l.outputs;

    l.index = index;

    l.delta = calloc(l.outputs * batch, sizeof(float));
    l.output = calloc(l.outputs * batch, sizeof(float));
    ;

    l.forward = forward_shortcut_layer;
    l.backward = backward_shortcut_layer;
#ifdef GPU
    l.forward_gpu = forward_shortcut_layer_gpu;
    l.backward_gpu = backward_shortcut_layer_gpu;

    l.delta_gpu = cuda_make_array(l.delta, l.outputs * batch);
    l.output_gpu = cuda_make_array(l.output, l.outputs * batch);
#endif
    return l;
}

layer16 make_shortcut_layer16(int batch, int index, int w, int h, int c, int w2, int h2, int c2)
{
    fprintf(stderr, "Shortcut Layer16: %d\n", index);
    layer16 l = {0};
    l.type = SHORTCUT;
    l.batch = batch;
    l.w = w2;
    l.h = h2;
    l.c = c2;
    l.out_w = w;
    l.out_h = h;
    l.out_c = c;
    l.outputs = w * h * c;
    l.inputs = l.outputs;

    l.index = index;

    l.delta = (FLT *)calloc(l.outputs * batch, sizeof(FLT)); //liuj added 20180713
    l.output = calloc(l.outputs * batch, sizeof(FLT));

    l.forward = forward_shortcut_layer16;

    fprintf(stderr, "delta: %f\n", l.delta[0]);
    return l;
}

void forward_shortcut_layer16(const layer16 l, network16 net)
{
    printf(" forward_shortcut_layer16 flag_vec = %d\n", net.flag_vec);

    copy_cpu16(l.outputs * l.batch, net.input, 1, l.output, 1);
    printf("foreward_shortcut:net.input[0]:%f,net.input[1]%f,net.input[2]:%f\n", net.input[0], net.input[1], net.input[2]);
    shortcut_cpu16(l.batch, l.w, l.h, l.c, net.layers[l.index].output, l.out_w, l.out_h, l.out_c, l.output); //printf("activate_array16!!!\n");
    activate_array16(l.output, l.outputs * l.batch, l.activation);
}

void forward_shortcut_layer(const layer l, network net)
{
    copy_cpu(l.outputs * l.batch, net.input, 1, l.output, 1);
    shortcut_cpu(l.batch, l.w, l.h, l.c, net.layers[l.index].output, l.out_w, l.out_h, l.out_c, l.output);
    activate_array(l.output, l.outputs * l.batch, l.activation);
#ifdef PRUNE
    int zero_n = 0, zero_c = 0;
#pragma omp parallel for
    for (int k = 0; k < l.outputs * l.batch; k++)
    {
        zero_c = 0;
        if (fabs(l.output[k]) <= dp_epsilon)
        {
            zero_c++;
            l.output[k] = 0.00f;
        }
    }
    // printf("Shrt layer, total parm: %d, saved param: %d\n", l.outputs * l.batch, zero_c);
    // printf("In summary, total load = %d, saved = %d\n", total_load_param += l.outputs * l.batch, total_saved_param += zero_c);
#endif
}

void backward_shortcut_layer(const layer l, network net)
{
    gradient_array(l.output, l.outputs * l.batch, l.activation, l.delta);
    axpy_cpu(l.outputs * l.batch, 1, l.delta, 1, net.delta, 1);
    shortcut_cpu(l.batch, l.out_w, l.out_h, l.out_c, l.delta, l.w, l.h, l.c, net.layers[l.index].delta);
}

#ifdef GPU
void forward_shortcut_layer_gpu(const layer l, network net)
{
    copy_gpu(l.outputs * l.batch, net.input_gpu, 1, l.output_gpu, 1);
    shortcut_gpu(l.batch, l.w, l.h, l.c, net.layers[l.index].output_gpu, l.out_w, l.out_h, l.out_c, l.output_gpu);
    activate_array_gpu(l.output_gpu, l.outputs * l.batch, l.activation);
}

void backward_shortcut_layer_gpu(const layer l, network net)
{
    gradient_array_gpu(l.output_gpu, l.outputs * l.batch, l.activation, l.delta_gpu);
    axpy_gpu(l.outputs * l.batch, 1, l.delta_gpu, 1, net.delta_gpu, 1);
    shortcut_gpu(l.batch, l.out_w, l.out_h, l.out_c, l.delta_gpu, l.w, l.h, l.c, net.layers[l.index].delta_gpu);
}
#endif
