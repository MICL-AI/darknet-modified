
#include <connected_layer.h>
//#include "activations_vec.h"

#include "convolutional_layer.h"
#include "batchnorm_layer.h"
#include "utils.h"
#include "cuda.h"
#include "blas.h"
#include "gemm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//#include <vec.h>
#include "time.h"

extern void transpose_matrix16(FLT *a, int rows, int cols);

int g_nebug = 0;

layer make_connected_layer(int batch, int inputs, int outputs, ACTIVATION activation, int batch_normalize, int adam)
{
    int i;
    layer l = {0};
    l.learning_rate_scale = 1;
    l.type = CONNECTED;

    l.inputs = inputs;
    l.outputs = outputs;
    l.batch = batch;
    printf("batch:%d\n", l.batch);
    l.batch_normalize = batch_normalize;
    l.h = 1;
    l.w = 1;
    l.c = inputs;
    l.out_h = 1;
    l.out_w = 1;
    l.out_c = outputs;
    printf("\n connected inputs:%d,outputs:%d\n", inputs, outputs);

    l.output = calloc(batch * outputs, sizeof(float));
    l.delta = calloc(batch * outputs, sizeof(float));

    l.weight_updates = calloc(inputs * outputs, sizeof(float));
    l.bias_updates = calloc(outputs, sizeof(float));

    l.weights = calloc(outputs * inputs, sizeof(float));
    printf("\nin make_connected_layer after calloc %f mem l.weights[33]=%f\n", (float)(outputs * inputs * sizeof(float)), l.weights[33]);
    l.biases = calloc(outputs, sizeof(float));

    l.forward = forward_connected_layer;
    l.backward = backward_connected_layer;
    l.update = update_connected_layer;

    //float scale = 1./sqrt(inputs);
    float scale = sqrt(2. / inputs);
    for (i = 0; i < outputs * inputs; ++i)
    {
        l.weights[i] = scale * rand_uniform(-1, 1);
    }

    for (i = 0; i < outputs; ++i)
    {
        l.biases[i] = 0;
    }

    if (adam)
    {
        l.m = calloc(l.inputs * l.outputs, sizeof(float));
        l.v = calloc(l.inputs * l.outputs, sizeof(float));
        l.bias_m = calloc(l.outputs, sizeof(float));
        l.scale_m = calloc(l.outputs, sizeof(float));
        l.bias_v = calloc(l.outputs, sizeof(float));
        l.scale_v = calloc(l.outputs, sizeof(float));
    }
    if (batch_normalize)
    {
        l.scales = calloc(outputs, sizeof(float));
        l.scale_updates = calloc(outputs, sizeof(float));
        for (i = 0; i < outputs; ++i)
        {
            l.scales[i] = 1;
        }

        l.mean = calloc(outputs, sizeof(float));
        l.mean_delta = calloc(outputs, sizeof(float));
        l.variance = calloc(outputs, sizeof(float));
        l.variance_delta = calloc(outputs, sizeof(float));

        l.rolling_mean = calloc(outputs, sizeof(float));
        l.rolling_variance = calloc(outputs, sizeof(float));

        l.x = calloc(batch * outputs, sizeof(float));
        l.x_norm = calloc(batch * outputs, sizeof(float));
    }

#ifdef GPU
    l.forward_gpu = forward_connected_layer_gpu;
    l.backward_gpu = backward_connected_layer_gpu;
    l.update_gpu = update_connected_layer_gpu;

    l.weights_gpu = cuda_make_array(l.weights, outputs * inputs);
    l.biases_gpu = cuda_make_array(l.biases, outputs);

    l.weight_updates_gpu = cuda_make_array(l.weight_updates, outputs * inputs);
    l.bias_updates_gpu = cuda_make_array(l.bias_updates, outputs);

    l.output_gpu = cuda_make_array(l.output, outputs * batch);
    l.delta_gpu = cuda_make_array(l.delta, outputs * batch);
    if (adam)
    {
        l.m_gpu = cuda_make_array(0, inputs * outputs);
        l.v_gpu = cuda_make_array(0, inputs * outputs);
        l.bias_m_gpu = cuda_make_array(0, outputs);
        l.bias_v_gpu = cuda_make_array(0, outputs);
        l.scale_m_gpu = cuda_make_array(0, outputs);
        l.scale_v_gpu = cuda_make_array(0, outputs);
    }

    if (batch_normalize)
    {
        l.mean_gpu = cuda_make_array(l.mean, outputs);
        l.variance_gpu = cuda_make_array(l.variance, outputs);

        l.rolling_mean_gpu = cuda_make_array(l.mean, outputs);
        l.rolling_variance_gpu = cuda_make_array(l.variance, outputs);

        l.mean_delta_gpu = cuda_make_array(l.mean, outputs);
        l.variance_delta_gpu = cuda_make_array(l.variance, outputs);

        l.scales_gpu = cuda_make_array(l.scales, outputs);
        l.scale_updates_gpu = cuda_make_array(l.scale_updates, outputs);

        l.x_gpu = cuda_make_array(l.output, l.batch * outputs);
        l.x_norm_gpu = cuda_make_array(l.output, l.batch * outputs);
#ifdef CUDNN
        cudnnCreateTensorDescriptor(&l.normTensorDesc);
        cudnnCreateTensorDescriptor(&l.dstTensorDesc);
        cudnnSetTensor4dDescriptor(l.dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l.batch, l.out_c, l.out_h, l.out_w);
        cudnnSetTensor4dDescriptor(l.normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l.out_c, 1, 1);
#endif
    }
#endif
    l.activation = activation;
    fprintf(stderr, "connected                            %4d  ->  %4d\n", inputs, outputs);
    return l;
}

layer16 make_connected_layer16(int batch, int inputs, int outputs, ACTIVATION activation, int batch_normalize, int adam)
{
    int i;
    layer16 l = {0};
    l.learning_rate_scale = 1;
    l.type = CONNECTED;

    l.inputs = inputs;
    l.outputs = outputs;
    l.batch = batch;
    l.batch_normalize = batch_normalize;
    l.h = 1;
    l.w = 1;
    l.c = inputs;
    l.out_h = 1;
    l.out_w = 1;
    l.out_c = outputs;

    l.output = calloc(batch * outputs, sizeof(FLT));
    l.delta = calloc(batch * outputs, sizeof(FLT));
    l.weights = calloc(outputs * inputs, sizeof(FLT));
    printf("\nin make_connected_layer16 after calloc %f mem l.weights[33]=%f\n", (float)(outputs * inputs * sizeof(FLT)), l.weights[33]);
    l.biases = calloc(outputs, sizeof(FLT));

    l.forward = forward_connected_layer16;

    FLT scale = sqrt(2. / inputs);
    for (i = 0; i < outputs * inputs; ++i)
    {
        l.weights[i] = scale * rand_uniform(-1, 1);
    }

    for (i = 0; i < outputs; ++i)
    {
        l.biases[i] = 0;
    }

    if (batch_normalize)
    {
        l.scales = calloc(outputs, sizeof(FLT));
        for (i = 0; i < outputs; ++i)
        {
            l.scales[i] = 1;
        }
        l.rolling_mean = calloc(outputs, sizeof(FLT));
        l.rolling_variance = calloc(outputs, sizeof(FLT));
        l.x = calloc(batch * outputs, sizeof(FLT));
    }

    l.activation = activation;
    fprintf(stderr, "connected                            %4d  ->  %4d\n", inputs, outputs);
    return l;
}

void update_connected_layer(layer l, update_args a)
{
    float learning_rate = a.learning_rate * l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;
    axpy_cpu(l.outputs, learning_rate / batch, l.bias_updates, 1, l.biases, 1);
    scal_cpu(l.outputs, momentum, l.bias_updates, 1);

    if (l.batch_normalize)
    {
        axpy_cpu(l.outputs, learning_rate / batch, l.scale_updates, 1, l.scales, 1);
        scal_cpu(l.outputs, momentum, l.scale_updates, 1);
    }

    axpy_cpu(l.inputs * l.outputs, -decay * batch, l.weights, 1, l.weight_updates, 1);
    axpy_cpu(l.inputs * l.outputs, learning_rate / batch, l.weight_updates, 1, l.weights, 1);
    scal_cpu(l.inputs * l.outputs, momentum, l.weight_updates, 1);
}
void saveresf(char *filename, float *data, int len)
{
    FILE *fp = fopen(filename, "w+");
    int num;
    for (num = 0; num < len; num++)
    {
        fprintf(fp, "%f ", data[num]);
    }
    fclose(fp);
}
void saveres(char *filename, FLT *data, int len)
{
    FILE *fp = fopen(filename, "w+");
    int num;
    for (num = 0; num < len; num++)
    {
        fprintf(fp, "%f ", (float)data[num]);
    }
    fclose(fp);
}
int g_conn = 0, g_bias = 0, g_batchnorm = 0;
char filename[64];

void forward_connected_layer(layer l, network net)
{
    printf("batch:%d,inputs:%d,outputs:%d\n", l.batch, l.inputs, l.outputs);
    if (0)
    {
        printf("file write\n");
        FILE *fp = fopen("connectedinput.dat", "wb");
        if (!fp)
            file_error(filename);
        fwrite(net.input, sizeof(float), l.batch * l.inputs, fp);
        fclose(fp);
        FILE *fpw = fopen("connectedweights.weights", "wb");
        int major = 0;
        int minor = 2;
        int revision = 0;
        *(net.seen) = 1;
        fwrite(&major, sizeof(int), 1, fpw);
        fwrite(&minor, sizeof(int), 1, fpw);
        fwrite(&revision, sizeof(int), 1, fpw);
        fwrite(&(net.seen), sizeof(size_t), 1, fpw);

        fwrite(l.biases, sizeof(float), l.outputs, fpw);
        fwrite(l.weights, sizeof(float), l.inputs * l.outputs, fpw);
        fclose(fpw);
        exit(0);
    }

    fill_cpu(l.outputs * l.batch, 0, l.output, 1);
    int m = l.batch;
    int k = l.inputs;
    int n = l.outputs;
    float *a = net.input;
    float *b = l.weights;
    float *c = l.output;
    printf("NOTE:in forward_connected_layer weights b[33]=%f, b[34]=%f\n", b[33], b[34]);
    
    printf("\nNOTE:in forward_connected_layer16 weights l.biases[33]=%f, l.biases[34]=%f\n", l.biases[33], l.biases[34]);
    if (net.flag_vec == 0)
    {
        printf("in forward_connected_layer gemmtype:%d,transpose flg:%d\n",net.gemm_type,l.transpose);
        if (l.transpose == 1)
            gemm(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
        else
            gemm(0, 1, m, n, k, 1, a, k, b, k, 1, c, n);
        if (l.batch_normalize)
            forward_batchnorm_layer(l, net);
        else
            add_bias16(l.output, l.biases, l.batch, l.outputs, 1);
        activate_array16(l.output, l.outputs * l.batch, l.activation);
    }

    else
    {
        /*gemm16vec(0,1,m,n,k,1,a,k,b,k,1,c,n);
	    
	    if(l.batch_normalize){
		forward_batchnorm_layer16(l, net); 
	    } else {   		 
		mul_or_add_cpu16vec(l.outputs ,l.biases , l.output ,1);//add
	    }
	    activate_array16vec(l.output, l.outputs*l.batch, l.activation);*/
    }

    if (l.batch_normalize)
    {
        forward_batchnorm_layer(l, net);
    }
    else
    {
        add_bias(l.output, l.biases, l.batch, l.outputs, 1);
    }

    activate_array(l.output, l.outputs * l.batch, l.activation);

    g_conn++;
}
clock_t aa, bb;
void forward_connected_layer16(layer16 l, network16 net)
{
    fill_cpu16(l.outputs * l.batch, 0, l.output, 1);
    int m = l.batch;
    int k = l.inputs;
    int n = l.outputs;
    FLT *a = net.input;
    FLT *b = l.weights;
    FLT *c = l.output;
    printf("\nNOTE:in forward_connected_layer16 weights b[33]=%f, b[34]=%f\n", b[33], b[34]);
    printf("\nNOTE:in forward_connected_layer16 weights l.biases[33]=%f, l.biases[34]=%f\n", l.biases[33], l.biases[34]);
    if (net.flag_vec == 0)
    {
        printf("NOTE:in forward_connected_layer16 gemmtype:%d,transpose flg:%d\n",net.gemm_type,l.transpose);
        if (l.transpose == 1)
        {
            //printf("transpose\n");
            //transpose_matrix16(l.weights,n,k);
            if (net.gemm_type == 5)
                gemm16m5(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
            else
                gemm16(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
        }
        else
        {
            if (net.gemm_type == 5)
                gemm16m5(0, 1, m, n, k, 1, a, k, b, k, 1, c, n);
            else
                gemm16(0, 1, m, n, k, 1, a, k, b, k, 1, c, n);
        }
        if (l.batch_normalize)
        {
            forward_batchnorm_layer16(l, net);
        }
        else
        {
            add_bias16(l.output, l.biases, l.batch, l.outputs, 1);
        }

        activate_array16(l.output, l.outputs * l.batch, l.activation);
    }

    else
    {
        /*gemm16vec(0,1,m,n,k,1,a,k,b,k,1,c,n);
	    
	    if(l.batch_normalize){
		forward_batchnorm_layer16(l, net); 
	    } else {   		 
		mul_or_add_cpu16vec(l.outputs ,l.biases , l.output ,1);//add
	    }
	    activate_array16vec(l.output, l.outputs*l.batch, l.activation);*/
    }
}

void backward_connected_layer(layer l, network net)
{
    gradient_array(l.output, l.outputs * l.batch, l.activation, l.delta);

    if (l.batch_normalize)
    {
        backward_batchnorm_layer(l, net);
    }
    else
    {
        backward_bias(l.bias_updates, l.delta, l.batch, l.outputs, 1);
    }

    int m = l.outputs;
    int k = l.batch;
    int n = l.inputs;
    float *a = l.delta;
    float *b = net.input;
    float *c = l.weight_updates;
    gemm(1, 0, m, n, k, 1, a, m, b, n, 1, c, n);

    m = l.batch;
    k = l.outputs;
    n = l.inputs;

    a = l.delta;
    b = l.weights;
    c = net.delta;

    if (c)
        gemm(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
}

void denormalize_connected_layer(layer l)
{
    int i, j;
    for (i = 0; i < l.outputs; ++i)
    {
        float scale = l.scales[i] / sqrt(l.rolling_variance[i] + .000001);
        for (j = 0; j < l.inputs; ++j)
        {
            l.weights[i * l.inputs + j] *= scale;
        }
        l.biases[i] -= l.rolling_mean[i] * scale;
        l.scales[i] = 1;
        l.rolling_mean[i] = 0;
        l.rolling_variance[i] = 1;
    }
}

void statistics_connected_layer(layer l)
{
    if (l.batch_normalize)
    {
        printf("Scales ");
        print_statistics(l.scales, l.outputs);
        /*
           printf("Rolling Mean ");
           print_statistics(l.rolling_mean, l.outputs);
           printf("Rolling Variance ");
           print_statistics(l.rolling_variance, l.outputs);
         */
    }
    printf("Biases ");
    print_statistics(l.biases, l.outputs);
    printf("Weights ");
    print_statistics(l.weights, l.outputs);
}

#ifdef GPU

void pull_connected_layer(layer l)
{
    cuda_pull_array(l.weights_gpu, l.weights, l.inputs * l.outputs);
    cuda_pull_array(l.biases_gpu, l.biases, l.outputs);
    cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.inputs * l.outputs);
    cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.outputs);
    if (l.batch_normalize)
    {
        cuda_pull_array(l.scales_gpu, l.scales, l.outputs);
        cuda_pull_array(l.rolling_mean_gpu, l.rolling_mean, l.outputs);
        cuda_pull_array(l.rolling_variance_gpu, l.rolling_variance, l.outputs);
    }
}

void push_connected_layer(layer l)
{
    cuda_push_array(l.weights_gpu, l.weights, l.inputs * l.outputs);
    cuda_push_array(l.biases_gpu, l.biases, l.outputs);
    cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.inputs * l.outputs);
    cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.outputs);
    if (l.batch_normalize)
    {
        cuda_push_array(l.scales_gpu, l.scales, l.outputs);
        cuda_push_array(l.rolling_mean_gpu, l.rolling_mean, l.outputs);
        cuda_push_array(l.rolling_variance_gpu, l.rolling_variance, l.outputs);
    }
}

void update_connected_layer_gpu(layer l, update_args a)
{
    float learning_rate = a.learning_rate * l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;
    if (a.adam)
    {
        adam_update_gpu(l.weights_gpu, l.weight_updates_gpu, l.m_gpu, l.v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.inputs * l.outputs, batch, a.t);
        adam_update_gpu(l.biases_gpu, l.bias_updates_gpu, l.bias_m_gpu, l.bias_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.outputs, batch, a.t);
        if (l.scales_gpu)
        {
            adam_update_gpu(l.scales_gpu, l.scale_updates_gpu, l.scale_m_gpu, l.scale_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.outputs, batch, a.t);
        }
    }
    else
    {
        axpy_gpu(l.outputs, learning_rate / batch, l.bias_updates_gpu, 1, l.biases_gpu, 1);
        scal_gpu(l.outputs, momentum, l.bias_updates_gpu, 1);

        if (l.batch_normalize)
        {
            axpy_gpu(l.outputs, learning_rate / batch, l.scale_updates_gpu, 1, l.scales_gpu, 1);
            scal_gpu(l.outputs, momentum, l.scale_updates_gpu, 1);
        }

        axpy_gpu(l.inputs * l.outputs, -decay * batch, l.weights_gpu, 1, l.weight_updates_gpu, 1);
        axpy_gpu(l.inputs * l.outputs, learning_rate / batch, l.weight_updates_gpu, 1, l.weights_gpu, 1);
        scal_gpu(l.inputs * l.outputs, momentum, l.weight_updates_gpu, 1);
    }
}

void forward_connected_layer_gpu(layer l, network net)
{
    fill_gpu(l.outputs * l.batch, 0, l.output_gpu, 1);

    int m = l.batch;
    int k = l.inputs;
    int n = l.outputs;
    float *a = net.input_gpu;
    float *b = l.weights_gpu;
    float *c = l.output_gpu;
    gemm_gpu(0, 1, m, n, k, 1, a, k, b, k, 1, c, n);

    if (l.batch_normalize)
    {
        forward_batchnorm_layer_gpu(l, net);
    }
    else
    {
        add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.outputs, 1);
    }
    activate_array_gpu(l.output_gpu, l.outputs * l.batch, l.activation);
}

void backward_connected_layer_gpu(layer l, network net)
{
    constrain_gpu(l.outputs * l.batch, 1, l.delta_gpu, 1);
    gradient_array_gpu(l.output_gpu, l.outputs * l.batch, l.activation, l.delta_gpu);
    if (l.batch_normalize)
    {
        backward_batchnorm_layer_gpu(l, net);
    }
    else
    {
        backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.outputs, 1);
    }

    int m = l.outputs;
    int k = l.batch;
    int n = l.inputs;
    float *a = l.delta_gpu;
    float *b = net.input_gpu;
    float *c = l.weight_updates_gpu;
    gemm_gpu(1, 0, m, n, k, 1, a, m, b, n, 1, c, n);

    m = l.batch;
    k = l.outputs;
    n = l.inputs;

    a = l.delta_gpu;
    b = l.weights_gpu;
    c = net.delta_gpu;

    if (c)
        gemm_gpu(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
}
#endif
