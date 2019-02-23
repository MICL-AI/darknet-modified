#include "convolutional_layer.h"
#include "utils.h"
#include "batchnorm_layer.h"
#include "im2col.h"
#include "col2im.h"
#include "blas.h"
#include "gemm.h"
#include <stdio.h>
#include <time.h>
//#include "vec.h"

#ifdef AI2
#include "xnor_layer.h"
#endif

void swap_binary(convolutional_layer *l)
{
    float *swap = l->weights;
    l->weights = l->binary_weights;
    l->binary_weights = swap;

#ifdef GPU
    swap = l->weights_gpu;
    l->weights_gpu = l->binary_weights_gpu;
    l->binary_weights_gpu = swap;
#endif
}

void binarize_weights(float *weights, int n, int size, float *binary)
{
    int i, f;
    for (f = 0; f < n; ++f)
    {
        float mean = 0;
        for (i = 0; i < size; ++i)
        {
            mean += fabs(weights[f * size + i]);
        }
        mean = mean / size;
        for (i = 0; i < size; ++i)
        {
            binary[f * size + i] = (weights[f * size + i] > 0) ? mean : -mean;
        }
    }
}

void binarize_cpu(float *input, int n, float *binary)
{
    int i;
    for (i = 0; i < n; ++i)
    {
        binary[i] = (input[i] > 0) ? 1 : -1;
    }
}

void binarize_input(float *input, int n, int size, float *binary)
{
    int i, s;
    for (s = 0; s < size; ++s)
    {
        float mean = 0;
        for (i = 0; i < n; ++i)
        {
            mean += fabs(input[i * size + s]);
        }
        mean = mean / n;
        for (i = 0; i < n; ++i)
        {
            binary[i * size + s] = (input[i * size + s] > 0) ? mean : -mean;
        }
    }
}
int convolutional_out_height(convolutional_layer l)
{
    return (l.h + 2 * l.pad - l.size) / l.stride + 1;
}
int convolutional_out_height16(convolutional_layer16 l)
{
    return (l.h + 2 * l.pad - l.size) / l.stride + 1;
}

int convolutional_out_width(convolutional_layer l)
{
    return (l.w + 2 * l.pad - l.size) / l.stride + 1;
}
int convolutional_out_width16(convolutional_layer16 l)
{
    return (l.w + 2 * l.pad - l.size) / l.stride + 1;
}
image get_convolutional_image(convolutional_layer l)
{
    return float_to_image(l.out_w, l.out_h, l.out_c, l.output);
}

image get_convolutional_delta(convolutional_layer l)
{
    return float_to_image(l.out_w, l.out_h, l.out_c, l.delta);
}

static size_t get_workspace_size16(layer16 l)
{
    return (size_t)l.out_h * l.out_w * l.size * l.size * l.c / l.groups * sizeof(FLT);
}

static size_t get_workspace_size(layer l)
{
#ifdef CUDNN
    if (gpu_index >= 0)
    {
        size_t most = 0;
        size_t s = 0;
        cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle(),
                                                l.srcTensorDesc,
                                                l.weightDesc,
                                                l.convDesc,
                                                l.dstTensorDesc,
                                                l.fw_algo,
                                                &s);
        if (s > most)
            most = s;
        cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn_handle(),
                                                       l.srcTensorDesc,
                                                       l.ddstTensorDesc,
                                                       l.convDesc,
                                                       l.dweightDesc,
                                                       l.bf_algo,
                                                       &s);
        if (s > most)
            most = s;
        cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle(),
                                                     l.weightDesc,
                                                     l.ddstTensorDesc,
                                                     l.convDesc,
                                                     l.dsrcTensorDesc,
                                                     l.bd_algo,
                                                     &s);
        if (s > most)
            most = s;
        return most;
    }
#endif
    return (size_t)l.out_h * l.out_w * l.size * l.size * l.c / l.groups * sizeof(float);
}

#ifdef GPU
#ifdef CUDNN
void cudnn_convolutional_setup(layer *l)
{
    cudnnSetTensor4dDescriptor(l->dsrcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->c, l->h, l->w);
    cudnnSetTensor4dDescriptor(l->ddstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->out_c, l->out_h, l->out_w);

    cudnnSetTensor4dDescriptor(l->srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->c, l->h, l->w);
    cudnnSetTensor4dDescriptor(l->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->out_c, l->out_h, l->out_w);
    cudnnSetTensor4dDescriptor(l->normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l->out_c, 1, 1);

    cudnnSetFilter4dDescriptor(l->dweightDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, l->n, l->c / l->groups, l->size, l->size);
    cudnnSetFilter4dDescriptor(l->weightDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, l->n, l->c / l->groups, l->size, l->size);
#if CUDNN_MAJOR >= 6
    cudnnSetConvolution2dDescriptor(l->convDesc, l->pad, l->pad, l->stride, l->stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
#else
    cudnnSetConvolution2dDescriptor(l->convDesc, l->pad, l->pad, l->stride, l->stride, 1, 1, CUDNN_CROSS_CORRELATION);
#endif

#if CUDNN_MAJOR >= 7
    cudnnSetConvolutionGroupCount(l->convDesc, l->groups);
#else
    if (l->groups > 1)
    {
        error("CUDNN < 7 doesn't support groups, please upgrade!");
    }
#endif

    cudnnGetConvolutionForwardAlgorithm(cudnn_handle(),
                                        l->srcTensorDesc,
                                        l->weightDesc,
                                        l->convDesc,
                                        l->dstTensorDesc,
                                        CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
                                        4000000000,
                                        &l->fw_algo);
    cudnnGetConvolutionBackwardDataAlgorithm(cudnn_handle(),
                                             l->weightDesc,
                                             l->ddstTensorDesc,
                                             l->convDesc,
                                             l->dsrcTensorDesc,
                                             CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
                                             4000000000,
                                             &l->bd_algo);
    cudnnGetConvolutionBackwardFilterAlgorithm(cudnn_handle(),
                                               l->srcTensorDesc,
                                               l->ddstTensorDesc,
                                               l->convDesc,
                                               l->dweightDesc,
                                               CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
                                               4000000000,
                                               &l->bf_algo);
}
#endif
#endif

convolutional_layer16 make_convolutional_layer16(int batch, int h, int w, int c, int n, int groups, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam)
{
    int i;
    convolutional_layer16 l = {0};
    l.type = CONVOLUTIONAL;

    l.groups = groups;
    l.h = h;
    l.w = w;
    l.c = c;
    l.n = n;
    l.binary = binary;
    l.xnor = xnor;
    l.batch = batch;
    l.stride = stride;
    l.size = size;
    l.pad = padding;
    l.batch_normalize = batch_normalize;

    l.weights = (FLT *)calloc(c / groups * n * size * size, sizeof(FLT));
    if (!l.weights)
    {
        printf("in make_convolutional_layer16 weights calloc failed\n");
        exit(1);
    }
    l.biases = (FLT *)calloc(n, sizeof(FLT));
    l.bn_bias = (FLT *)calloc(n, sizeof(FLT));
    l.nweights = c / groups * n * size * size;

    FLT scale = sqrt(2. / (size * size * c / l.groups));
    for (i = 0; i < l.nweights; ++i)
        l.weights[i] = scale * rand_normal();
    int out_w = convolutional_out_width16(l);
    int out_h = convolutional_out_height16(l);
    l.out_h = out_h;
    l.out_w = out_w;
    l.out_c = n;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = l.w * l.h * l.c;

    l.output = (FLT *)calloc(l.batch * l.outputs, sizeof(FLT));
    if (!l.output)
    {
        printf("in make_convolutional_layer16 output calloc failed\n");
        exit(1);
    }

    //TODO: find out if delta calloc failed caused connected weights err by jliu 181030
    l.delta = calloc(l.batch * l.outputs, sizeof(FLT)); //liuj
    if (!l.delta)
    {
        printf("in make_convolutional_layer16 delta calloc failed\n");
        exit(1);
    }
    l.forward = forward_convolutional_layer16;

    if (batch_normalize)
    {
        l.scales = (FLT *)calloc(n, sizeof(FLT));
        l.bn_scale = (FLT *)calloc(n, sizeof(FLT));
        for (i = 0; i < n; ++i)
        {
            l.scales[i] = 1;
            l.bn_scale[i] = 1;
        }

        l.rolling_mean = (FLT *)calloc(n, sizeof(FLT));
        l.rolling_variance = (FLT *)calloc(n, sizeof(FLT));
        l.rolling_varianceMultiscales = (FLT *)calloc(n, sizeof(FLT));
        l.x = (FLT *)calloc(l.batch * l.outputs, sizeof(FLT));
        l.x_norm = (FLT *)calloc(l.batch * l.outputs, sizeof(FLT));
    }
    l.workspace_size = get_workspace_size16(l);
    l.activation = activation;

    fprintf(stderr, "conv  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", n, size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c);
    //    for(i=0;i<l.batch*l.outputs;i++)
    //     l.delta[0] = 0;
    fprintf(stderr, "conv  l.batch*l.outputs:%d,delta:delta[0]:%f,delta[%d*%d]:%f\n", l.batch * l.outputs, (float)l.delta[0], l.batch, l.outputs, (float)l.delta[l.batch * l.outputs - 1]);
    //fprintf(stderr,"sizeof(FLT):%ld, sizeof(float):%ld\n",sizeof(FLT),sizeof(float));
    //fprintf(stderr, "conv l.delta:%lf\n",*(double *)l.delta);
    return l;
}

convolutional_layer make_convolutional_layer(int batch, int h, int w, int c, int n, int groups, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam)
{
    int i;
    convolutional_layer l = {0};
    l.type = CONVOLUTIONAL;

    l.groups = groups;
    l.h = h;
    l.w = w;
    l.c = c;
    l.n = n;
    l.binary = binary;
    l.xnor = xnor;
    l.batch = batch;
    l.stride = stride;
    l.size = size;
    l.pad = padding;
    l.batch_normalize = batch_normalize;
    l.prune = calloc(l.out_c, sizeof(bool));

    l.weights = calloc(c / groups * n * size * size, sizeof(float));
    if (!l.weights)
    {
        printf("no enough memory for output\n");
        exit(1);
    }
    l.weight_updates = calloc(c / groups * n * size * size, sizeof(float));

    l.biases = calloc(n, sizeof(float));
    l.bias_updates = calloc(n, sizeof(float));

    l.nweights = c / groups * n * size * size; //printf("numweights:%d\n",l.weights);
    l.nbiases = n;

    // float scale = 1./sqrt(size*size*c);
    float scale = sqrt(2. / (size * size * c / l.groups));
    //scale = .02;
    //for(i = 0; i < c*n*size*size; ++i) l.weights[i] = scale*rand_uniform(-1, 1);
    for (i = 0; i < l.nweights; ++i)
        l.weights[i] = scale * rand_normal();
    int out_w = convolutional_out_width(l);
    int out_h = convolutional_out_height(l);
    l.out_h = out_h;
    l.out_w = out_w;
    l.out_c = n;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = l.w * l.h * l.c;

    l.output = calloc(l.batch * l.outputs, sizeof(float));
    if (!l.output)
    {
        printf("no enough memory for output\n");
        exit(1);
    }
    l.delta = calloc(l.batch * l.outputs, sizeof(float));

    l.forward = forward_convolutional_layer;
    l.backward = backward_convolutional_layer;
    l.update = update_convolutional_layer;
    if (binary)
    {
        l.binary_weights = calloc(l.nweights, sizeof(float));
        l.cweights = calloc(l.nweights, sizeof(char));
        l.scales = calloc(n, sizeof(float));
    }
    if (xnor)
    {
        l.binary_weights = calloc(l.nweights, sizeof(float));
        l.binary_input = calloc(l.inputs * l.batch, sizeof(float));
    }

    if (batch_normalize)
    {
        l.scales = calloc(n, sizeof(float));
        l.scale_updates = calloc(n, sizeof(float));
        for (i = 0; i < n; ++i)
        {
            l.scales[i] = 1;
        }

        l.mean = calloc(n, sizeof(float));
        l.variance = calloc(n, sizeof(float));

        l.mean_delta = calloc(n, sizeof(float));
        l.variance_delta = calloc(n, sizeof(float));

        l.rolling_mean = calloc(n, sizeof(float));
        l.rolling_variance = calloc(n, sizeof(float));
        l.x = calloc(l.batch * l.outputs, sizeof(float));
        l.x_norm = calloc(l.batch * l.outputs, sizeof(float));
    }
    if (adam)
    {
        l.m = calloc(l.nweights, sizeof(float));
        l.v = calloc(l.nweights, sizeof(float));
        l.bias_m = calloc(n, sizeof(float));
        l.scale_m = calloc(n, sizeof(float));
        l.bias_v = calloc(n, sizeof(float));
        l.scale_v = calloc(n, sizeof(float));
    }

#ifdef GPU
    l.forward_gpu = forward_convolutional_layer_gpu;
    l.backward_gpu = backward_convolutional_layer_gpu;
    l.update_gpu = update_convolutional_layer_gpu;

    if (gpu_index >= 0)
    {
        if (adam)
        {
            l.m_gpu = cuda_make_array(l.m, l.nweights);
            l.v_gpu = cuda_make_array(l.v, l.nweights);
            l.bias_m_gpu = cuda_make_array(l.bias_m, n);
            l.bias_v_gpu = cuda_make_array(l.bias_v, n);
            l.scale_m_gpu = cuda_make_array(l.scale_m, n);
            l.scale_v_gpu = cuda_make_array(l.scale_v, n);
        }

        l.weights_gpu = cuda_make_array(l.weights, l.nweights);
        l.weight_updates_gpu = cuda_make_array(l.weight_updates, l.nweights);

        l.biases_gpu = cuda_make_array(l.biases, n);
        l.bias_updates_gpu = cuda_make_array(l.bias_updates, n);

        l.delta_gpu = cuda_make_array(l.delta, l.batch * out_h * out_w * n);
        l.output_gpu = cuda_make_array(l.output, l.batch * out_h * out_w * n);

        if (binary)
        {
            l.binary_weights_gpu = cuda_make_array(l.weights, l.nweights);
        }
        if (xnor)
        {
            l.binary_weights_gpu = cuda_make_array(l.weights, l.nweights);
            l.binary_input_gpu = cuda_make_array(0, l.inputs * l.batch);
        }

        if (batch_normalize)
        {
            l.mean_gpu = cuda_make_array(l.mean, n);
            l.variance_gpu = cuda_make_array(l.variance, n);

            l.rolling_mean_gpu = cuda_make_array(l.mean, n);
            l.rolling_variance_gpu = cuda_make_array(l.variance, n);

            l.mean_delta_gpu = cuda_make_array(l.mean, n);
            l.variance_delta_gpu = cuda_make_array(l.variance, n);

            l.scales_gpu = cuda_make_array(l.scales, n);
            l.scale_updates_gpu = cuda_make_array(l.scale_updates, n);

            l.x_gpu = cuda_make_array(l.output, l.batch * out_h * out_w * n);
            l.x_norm_gpu = cuda_make_array(l.output, l.batch * out_h * out_w * n);
        }
#ifdef CUDNN
        cudnnCreateTensorDescriptor(&l.normTensorDesc);
        cudnnCreateTensorDescriptor(&l.srcTensorDesc);
        cudnnCreateTensorDescriptor(&l.dstTensorDesc);
        cudnnCreateFilterDescriptor(&l.weightDesc);
        cudnnCreateTensorDescriptor(&l.dsrcTensorDesc);
        cudnnCreateTensorDescriptor(&l.ddstTensorDesc);
        cudnnCreateFilterDescriptor(&l.dweightDesc);
        cudnnCreateConvolutionDescriptor(&l.convDesc);
        cudnn_convolutional_setup(&l);
#endif
    }
#endif
    l.workspace_size = get_workspace_size(l);
    l.activation = activation;

    fprintf(stderr, "conv  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", n, size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c);

    return l;
}

void denormalize_convolutional_layer(convolutional_layer l)
{
    int i, j;
    for (i = 0; i < l.n; ++i)
    {
        float scale = l.scales[i] / sqrt(l.rolling_variance[i] + .00001);
        for (j = 0; j < l.c / l.groups * l.size * l.size; ++j)
        {
            l.weights[i * l.c / l.groups * l.size * l.size + j] *= scale;
        }
        l.biases[i] -= l.rolling_mean[i] * scale;
        l.scales[i] = 1;
        l.rolling_mean[i] = 0;
        l.rolling_variance[i] = 1;
    }
}

/*
void test_convolutional_layer()
{
    convolutional_layer l = make_convolutional_layer(1, 5, 5, 3, 2, 5, 2, 1, LEAKY, 1, 0, 0, 0);
    l.batch_normalize = 1;
    float data[] = {1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3};
    //net.input = data;
    //forward_convolutional_layer(l);
}
*/

void resize_convolutional_layer(convolutional_layer *l, int w, int h)
{
    l->w = w;
    l->h = h;
    int out_w = convolutional_out_width(*l);
    int out_h = convolutional_out_height(*l);

    l->out_w = out_w;
    l->out_h = out_h;

    l->outputs = l->out_h * l->out_w * l->out_c;
    l->inputs = l->w * l->h * l->c;

    l->output = realloc(l->output, l->batch * l->outputs * sizeof(float));
    l->delta = realloc(l->delta, l->batch * l->outputs * sizeof(float));
    if (l->batch_normalize)
    {
        l->x = realloc(l->x, l->batch * l->outputs * sizeof(float));
        l->x_norm = realloc(l->x_norm, l->batch * l->outputs * sizeof(float));
    }

#ifdef GPU
    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);

    l->delta_gpu = cuda_make_array(l->delta, l->batch * l->outputs);
    l->output_gpu = cuda_make_array(l->output, l->batch * l->outputs);

    if (l->batch_normalize)
    {
        cuda_free(l->x_gpu);
        cuda_free(l->x_norm_gpu);

        l->x_gpu = cuda_make_array(l->output, l->batch * l->outputs);
        l->x_norm_gpu = cuda_make_array(l->output, l->batch * l->outputs);
    }
#ifdef CUDNN
    cudnn_convolutional_setup(l);
#endif
#endif
    l->workspace_size = get_workspace_size(*l);
}
void add_bias16(FLT *output, FLT *biases, int batch, int n, int size)
{
    int i, j, b;
    for (b = 0; b < batch; ++b)
    {
        for (i = 0; i < n; ++i)
        {
            for (j = 0; j < size; ++j)
            {
                output[(b * n + i) * size + j] += biases[i];
            }
        }
    }
}
void add_bias(float *output, float *biases, int batch, int n, int size)
{
    int i, j, b;
    for (b = 0; b < batch; ++b)
    {
        for (i = 0; i < n; ++i)
        {
            for (j = 0; j < size; ++j)
            {
                output[(b * n + i) * size + j] += biases[i];
                //printf("output[%d]:%f",(b*n + i)*size + j,output[(b*n + i)*size + j]);
            } //printf("output[%d]:%f",(b*n + i)*size + j,output[(b*n + i)*size + j]);printf("\n");
        }
    }
}

void scale_bias16vec(FLT *output, FLT *scales, int batch, int n, int size)
{ //ddd
    int i, j, b;
    for (b = 0; b < batch; ++b)
    {
        for (i = 0; i < n; ++i)
        {
            for (j = 0; j < size; ++j)
            {
                output[(b * n + i) * size + j] *= scales[i];
            }
        }
    }
}

void scale_bias16(FLT *output, FLT *scales, int batch, int n, int size)
{
    int i, j, b;
    for (b = 0; b < batch; ++b)
    {
        for (i = 0; i < n; ++i)
        {
            for (j = 0; j < size; ++j)
            {
                output[(b * n + i) * size + j] *= scales[i];
            }
        }
    }
}
void scale_bias(float *output, float *scales, int batch, int n, int size)
{
    int i, j, b;
    for (b = 0; b < batch; ++b)
    {
        for (i = 0; i < n; ++i)
        {
            for (j = 0; j < size; ++j)
            {
                output[(b * n + i) * size + j] *= scales[i];
            }
        }
    }
}

void backward_bias(float *bias_updates, float *delta, int batch, int n, int size)
{
    int i, b;
    for (b = 0; b < batch; ++b)
    {
        for (i = 0; i < n; ++i)
        {
            bias_updates[i] += sum_array(delta + size * (i + b * n), size);
        }
    }
}

void saveresfconv(char *filename, float *data, int len)
{
    FILE *fp = fopen(filename, "w+");
    int num;
    for (num = 0; num < len; num++)
    {
        fprintf(fp, "%f ", data[num]);
    }
    fclose(fp);
}
void saveresconv(char *filename, FLT *data, int len)
{
    FILE *fp = fopen(filename, "w+");
    int num;
    for (num = 0; num < len; num++)
    {
        fprintf(fp, "%f ", (float)data[num]);
    }
    fclose(fp);
}

//liuj0825 combain
void normalize_scales_biases_cpu16(FLT *output, FLT *mean, FLT *biases, int batch, int filters, int spatial, FLT *varianceMultiscales)
{
    int b, f, i;
    for (b = 0; b < batch; ++b)
    {
        for (f = 0; f < filters; ++f)
        {
            for (i = 0; i < spatial; ++i)
            {
                int index = b * filters * spatial + f * spatial + i;
                output[index] = (output[index] - mean[f]) * varianceMultiscales[f] + biases[f];
                //output[index] = (output[index] - mean[f])*variance[f];
                //output[index] *= scales[f];
                //output[index] += biases[f];
            }
        }
    }
}

void norm_scale_bias(FLT *output, FLT *bn_scale, FLT *bn_bias, int batch, int filters, int spatial)
{

    int b, f, i;
    for (b = 0; b < batch; ++b)
        for (f = 0; f < filters; ++f)
            for (i = 0; i < spatial; ++i)
            {
                int index = b + filters * spatial + f * spatial + i;
                output[index] = output[index] * bn_scale[f] + bn_bias[f];
            }
}

void forward_convolutional_layer16(convolutional_layer16 l, network16 net)
{
    int i, j; //printf("forward_convolutional_layer16\n");

    //printf("net.input[0]:%f,net.input[1]:%f,net.input[2]:%f\n",(float)net.input[0],(float)net.input[1],(float)net.input[2]);
    if (net.flag_vec)
    { //vec
        /*fill_cpu16vec(l.outputs*l.batch, 0, l.output);

	    int m = l.n/l.groups;
	    int k = l.size*l.size*l.c/l.groups;
	    int n = l.out_w*l.out_h;

	    for(i = 0; i < l.batch; ++i){
		    for(j = 0; j < l.groups; ++j){
			    FLT *a = l.weights + j*l.nweights/l.groups;
			    FLT *b = net.workspace;
			    FLT *c = l.output + (i*l.groups + j)*n*m;
			    im2col_cpu16(net.input + (i*l.groups + j)*l.c/l.groups*l.h*l.w,
					    l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
			    gemm16(0,0,m,n,k,1,a,k,b,n,1,c,n);				
		    }
	    }

	    if(l.batch_normalize){
		    forward_batchnorm_layer16(l, net);
	    } else {
		    add_bias16(l.output, l.biases, l.batch, l.n, l.out_h*l.out_w);
	    }
		
	    activate_array16(l.output, l.outputs*l.batch, l.activation);
*/
        ;
    }
    else
    { //scalar
        fill_cpu16(l.outputs * l.batch, 0, l.output, 1);

        int m = l.n / l.groups;
        int k = l.size * l.size * l.c / l.groups;
        int n = l.out_w * l.out_h;
        for (i = 0; i < l.batch; ++i)
        {
            for (j = 0; j < l.groups; ++j)
            {
                FLT *a = l.weights + j * l.nweights / l.groups;
                FLT *b = net.workspace;
                FLT *c = l.output + (i * l.groups + j) * n * m;
                im2col_cpu16(net.input + (i * l.groups + j) * l.c / l.groups * l.h * l.w,
                             l.c / l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
                if (net.gemm_type == 2)
                    gemm16m2(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
                else if (net.gemm_type == 5)
                    gemm16m5(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
                else
                    gemm16(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
            }
        }

        if (l.batch_normalize)
        {
            //		    normalize_scales_biases_cpu16(l.output, l.rolling_mean, l.biases, l.batch, l.out_c, l.out_h*l.out_w,l.rolling_varianceMultiscales);//liuj20180825
            // norm_scale_bias(l.output, l.bn_scale,  l.bn_bias,  l.batch, l.n, n);
            forward_batchnorm_layer16(l, net);
        }
        else
        {
            add_bias16(l.output, l.biases, l.batch, l.n, l.out_h * l.out_w);
        }

        activate_array16(l.output, l.outputs * l.batch, l.activation);
    }
}
void forward_convolutional_layer(convolutional_layer l, network net)
{
    int i, j;

    fill_cpu(l.outputs * l.batch, 0, l.output, 1);

    if (l.xnor)
    {
        binarize_weights(l.weights, l.n, l.c / l.groups * l.size * l.size, l.binary_weights);
        swap_binary(&l);
        binarize_cpu(net.input, l.c * l.h * l.w * l.batch, l.binary_input);
        net.input = l.binary_input;
    }

    int m = l.n / l.groups;
    int k = l.size * l.size * l.c / l.groups;
    int n = l.out_w * l.out_h;
    for (i = 0; i < l.batch; ++i)
    {
        for (j = 0; j < l.groups; ++j)
        {
            float *a = l.weights + j * l.nweights / l.groups;
            float *b = net.workspace;
            float *c = l.output + (i * l.groups + j) * n * m; // printf("outnum:%d\n",n*m);
            im2col_cpu(net.input + (i * l.groups + j) * l.c / l.groups * l.h * l.w,
                       l.c / l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
            gemm(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
        }
    }
    //printf("!!!out gemm\n");
    if (l.batch_normalize)
    { //printf("normailize\n");
        forward_batchnorm_layer(l, net);
    }
    else
    { //printf("add bias\n");
        add_bias(l.output, l.biases, l.batch, l.n, l.out_h * l.out_w);
    }
    // printf("call activate\n");
    activate_array(l.output, l.outputs * l.batch, l.activation);
    
    /*TL 181203 adding for dynamic pruning test.*/ //Pruning
#ifdef PRUNE
    //print_channel(l.output, l.out_c, l.out_h, l.out_w);
    puts("*********pruning*********");
    prune_channel(l.output, l.out_c, l.out_h, l.out_w);
#endif
    if (l.binary || l.xnor)
        swap_binary(&l);
}

void backward_convolutional_layer(convolutional_layer l, network net)
{
    int i, j;
    int m = l.n / l.groups;
    int n = l.size * l.size * l.c / l.groups;
    int k = l.out_w * l.out_h;

    gradient_array(l.output, l.outputs * l.batch, l.activation, l.delta);

    if (l.batch_normalize)
    {
        backward_batchnorm_layer(l, net);
    }
    else
    {
        backward_bias(l.bias_updates, l.delta, l.batch, l.n, k);
    }

    for (i = 0; i < l.batch; ++i)
    {
        for (j = 0; j < l.groups; ++j)
        {
            float *a = l.delta + (i * l.groups + j) * m * k;
            float *b = net.workspace;
            float *c = l.weight_updates + j * l.nweights / l.groups;

            float *im = net.input + (i * l.groups + j) * l.c / l.groups * l.h * l.w;

            im2col_cpu(im, l.c / l.groups, l.h, l.w,
                       l.size, l.stride, l.pad, b);
            gemm(0, 1, m, n, k, 1, a, k, b, k, 1, c, n);

            if (net.delta)
            {
                a = l.weights + j * l.nweights / l.groups;
                b = l.delta + (i * l.groups + j) * m * k;
                c = net.workspace;

                gemm(1, 0, n, k, m, 1, a, n, b, k, 0, c, k);

                col2im_cpu(net.workspace, l.c / l.groups, l.h, l.w, l.size, l.stride,
                           l.pad, net.delta + (i * l.groups + j) * l.c / l.groups * l.h * l.w);
            }
        }
    }
}

void update_convolutional_layer(convolutional_layer l, update_args a)
{
    float learning_rate = a.learning_rate * l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;

    axpy_cpu(l.n, learning_rate / batch, l.bias_updates, 1, l.biases, 1);
    scal_cpu(l.n, momentum, l.bias_updates, 1);

    if (l.scales)
    {
        axpy_cpu(l.n, learning_rate / batch, l.scale_updates, 1, l.scales, 1);
        scal_cpu(l.n, momentum, l.scale_updates, 1);
    }

    axpy_cpu(l.nweights, -decay * batch, l.weights, 1, l.weight_updates, 1);
    axpy_cpu(l.nweights, learning_rate / batch, l.weight_updates, 1, l.weights, 1);
    scal_cpu(l.nweights, momentum, l.weight_updates, 1);
}

image get_convolutional_weight(convolutional_layer l, int i)
{
    int h = l.size;
    int w = l.size;
    int c = l.c / l.groups;
    return float_to_image(w, h, c, l.weights + i * h * w * c);
}

void rgbgr_weights(convolutional_layer l)
{
    int i;
    for (i = 0; i < l.n; ++i)
    {
        image im = get_convolutional_weight(l, i);
        if (im.c == 3)
        {
            rgbgr_image(im);
        }
    }
}

void rescale_weights(convolutional_layer l, float scale, float trans)
{
    int i;
    for (i = 0; i < l.n; ++i)
    {
        image im = get_convolutional_weight(l, i);
        if (im.c == 3)
        {
            scale_image(im, scale);
            float sum = sum_array(im.data, im.w * im.h * im.c);
            l.biases[i] += sum * trans;
        }
    }
}

image *get_weights(convolutional_layer l)
{
    image *weights = calloc(l.n, sizeof(image));
    int i;
    for (i = 0; i < l.n; ++i)
    {
        weights[i] = copy_image(get_convolutional_weight(l, i));
        normalize_image(weights[i]);
        /*
           char buff[256];
           sprintf(buff, "filter%d", i);
           save_image(weights[i], buff);
         */
    }
    //error("hey");
    return weights;
}

image *visualize_convolutional_layer(convolutional_layer l, char *window, image *prev_weights)
{
    image *single_weights = get_weights(l);
    show_images(single_weights, l.n, window);

    image delta = get_convolutional_image(l);
    image dc = collapse_image_layers(delta, 1);
    char buff[256];
    sprintf(buff, "%s: Output", window);
    //show_image(dc, buff);
    //save_image(dc, buff);
    free_image(dc);
    return single_weights;
}
