#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include "activation_layer.h"
#include "activations.h"
#include "avgpool_layer.h"
#include "batchnorm_layer.h"
#include "blas.h"
#include "connected_layer.h"
#include "deconvolutional_layer.h"
#include "convolutional_layer.h"
#include "cost_layer.h"
#include "crnn_layer.h"
#include "crop_layer.h"
#include "detection_layer.h"
#include "dropout_layer.h"
#include "gru_layer.h"
#include "list.h"
#include "local_layer.h"
#include "maxpool_layer.h"
#include "normalization_layer.h"
#include "option_list.h"
#include "parser.h"
#include "region_layer.h"
#include "reorg_layer.h"
#include "rnn_layer.h"
#include "route_layer.h"
#include "shortcut_layer.h"
#include "softmax_layer.h"
#include "lstm_layer.h"
#include "utils.h"
//TL added 181109
#include "sparse.h"

typedef struct
{
    char *type;
    list *options;
} section;

list *read_cfg(char *filename);

//FILE *fc = fopen("loadweigts.txt", "w");

LAYER_TYPE string_to_layer_type(char *type)
{

    if (strcmp(type, "[shortcut]") == 0)
        return SHORTCUT;
    if (strcmp(type, "[crop]") == 0)
        return CROP;
    if (strcmp(type, "[cost]") == 0)
        return COST;
    if (strcmp(type, "[detection]") == 0)
        return DETECTION;
    if (strcmp(type, "[region]") == 0)
        return REGION;
    if (strcmp(type, "[local]") == 0)
        return LOCAL;
    if (strcmp(type, "[conv]") == 0 || strcmp(type, "[convolutional]") == 0)
        return CONVOLUTIONAL;
    if (strcmp(type, "[deconv]") == 0 || strcmp(type, "[deconvolutional]") == 0)
        return DECONVOLUTIONAL;
    if (strcmp(type, "[activation]") == 0)
        return ACTIVE;
    if (strcmp(type, "[net]") == 0 || strcmp(type, "[network]") == 0)
        return NETWORK;
    if (strcmp(type, "[crnn]") == 0)
        return CRNN;
    if (strcmp(type, "[gru]") == 0)
        return GRU;
    if (strcmp(type, "[lstm]") == 0)
        return LSTM;
    if (strcmp(type, "[lstmcj]") == 0)
        return LSTMCJ;

    if (strcmp(type, "[rnn]") == 0)
        return RNN;
    if (strcmp(type, "[conn]") == 0 || strcmp(type, "[connected]") == 0)
        return CONNECTED;
    if (strcmp(type, "[max]") == 0 || strcmp(type, "[maxpool]") == 0)
        return MAXPOOL;
    if (strcmp(type, "[reorg]") == 0)
        return REORG;
    if (strcmp(type, "[avg]") == 0 || strcmp(type, "[avgpool]") == 0)
        return AVGPOOL;
    if (strcmp(type, "[dropout]") == 0)
        return DROPOUT;
    if (strcmp(type, "[lrn]") == 0 || strcmp(type, "[normalization]") == 0)
        return NORMALIZATION;
    if (strcmp(type, "[batchnorm]") == 0)
        return BATCHNORM;
    if (strcmp(type, "[soft]") == 0 || strcmp(type, "[softmax]") == 0)
        return SOFTMAX;
    if (strcmp(type, "[softcj]") == 0 || strcmp(type, "[softmaxcj]") == 0)
        return SOFTMAXCJ;
    if (strcmp(type, "[route]") == 0)
        return ROUTE;
    return BLANK;
}

void free_section(section *s)
{
    free(s->type);
    node *n = s->options->front;
    while (n)
    {
        kvp *pair = (kvp *)n->val;
        free(pair->key);
        free(pair);
        node *next = n->next;
        free(n);
        n = next;
    }
    free(s->options);
    free(s);
}

void parse_data(char *data, float *a, int n)
{
    int i;
    if (!data)
        return;
    char *curr = data;
    char *next = data;
    int done = 0;
    for (i = 0; i < n && !done; ++i)
    {
        while (*++next != '\0' && *next != ',')
            ;
        if (*next == '\0')
            done = 1;
        *next = '\0';
        sscanf(curr, "%g", &a[i]);
        curr = next + 1;
    }
}

typedef struct size_params
{
    int batch;
    int inputs;
    int h;
    int w;
    int c;
    int index;
    int flag_vec;
    int time_steps;
    network *net;
} size_params;
typedef struct size_params16
{
    int batch;
    int inputs;
    int h;
    int w;
    int c;
    int index;
    int time_steps;
    int flag_vec;
    network16 *net;
} size_params16;
local_layer parse_local(list *options, size_params params)
{
    int n = option_find_int(options, "filters", 1);
    int size = option_find_int(options, "size", 1);
    int stride = option_find_int(options, "stride", 1);
    int pad = option_find_int(options, "pad", 0);
    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);

    int batch, h, w, c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch = params.batch;
    if (!(h && w && c))
        error("Layer before local layer must output image.");

    local_layer layer = make_local_layer(batch, h, w, c, n, size, stride, pad, activation);

    return layer;
}

layer parse_deconvolutional(list *options, size_params params)
{
    int n = option_find_int(options, "filters", 1);
    int size = option_find_int(options, "size", 1);
    int stride = option_find_int(options, "stride", 1);

    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);

    int batch, h, w, c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch = params.batch;
    if (!(h && w && c))
        error("Layer before deconvolutional layer must output image.");
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);
    int pad = option_find_int_quiet(options, "pad", 0);
    int padding = option_find_int_quiet(options, "padding", 0);
    if (pad)
        padding = size / 2;

    layer l = make_deconvolutional_layer(batch, h, w, c, n, size, stride, padding, activation, batch_normalize, params.net->adam);

    return l;
}

convolutional_layer parse_convolutional(list *options, size_params params)
{
    int n = option_find_int(options, "filters", 1);
    int size = option_find_int(options, "size", 1);
    int stride = option_find_int(options, "stride", 1);
    int pad = option_find_int_quiet(options, "pad", 0);
    int padding = option_find_int_quiet(options, "padding", 0);
    int groups = option_find_int_quiet(options, "groups", 1);
    if (pad)
        padding = size / 2;

    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);

    int batch, h, w, c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch = params.batch;
    if (!(h && w && c))
        error("Layer before convolutional layer must output image.");
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);
    int binary = option_find_int_quiet(options, "binary", 0);
    int xnor = option_find_int_quiet(options, "xnor", 0);

    convolutional_layer layer = make_convolutional_layer(batch, h, w, c, n, groups, size, stride, padding, activation, batch_normalize, binary, xnor, params.net->adam);
    layer.flipped = option_find_int_quiet(options, "flipped", 0);
    layer.dot = option_find_float_quiet(options, "dot", 0);

    return layer;
}
convolutional_layer16 parse_convolutional16(list *options, size_params16 params)
{
    int n = option_find_int(options, "filters", 1);
    int size = option_find_int(options, "size", 1);
    int stride = option_find_int(options, "stride", 1);
    int pad = option_find_int_quiet(options, "pad", 0);
    int padding = option_find_int_quiet(options, "padding", 0);
    int groups = option_find_int_quiet(options, "groups", 1);
    if (pad)
        padding = size / 2;

    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);

    int batch, h, w, c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch = params.batch;
    if (!(h && w && c))
        error("Layer before convolutional layer must output image.");
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);
    int binary = option_find_int_quiet(options, "binary", 0);
    int xnor = option_find_int_quiet(options, "xnor", 0);

    convolutional_layer16 layer = make_convolutional_layer16(batch, h, w, c, n, groups, size, stride, padding, activation, batch_normalize, binary, xnor, params.net->adam);
    layer.flipped = option_find_int_quiet(options, "flipped", 0);
    layer.dot = option_find_float_quiet(options, "dot", 0);

    return layer;
}
layer parse_crnn(list *options, size_params params)
{
    int output_filters = option_find_int(options, "output_filters", 1);
    int hidden_filters = option_find_int(options, "hidden_filters", 1);
    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);

    layer l = make_crnn_layer(params.batch, params.w, params.h, params.c, hidden_filters, output_filters, params.time_steps, activation, batch_normalize);

    l.shortcut = option_find_int_quiet(options, "shortcut", 0);

    return l;
}

layer parse_rnn(list *options, size_params params)
{
    int output = option_find_int(options, "output", 1);
    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);
    printf("aaa=%d\n", batch_normalize);
    layer l = make_rnn_layer(params.batch, params.inputs, output, params.time_steps, activation, batch_normalize, params.net->adam);
    printf("bbb=%d\n", batch_normalize);

    l.shortcut = option_find_int_quiet(options, "shortcut", 0);

    return l;
}
layer16 parse_rnn16(list *options, size_params16 params)
{
    int output = option_find_int(options, "output", 1);
    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);

    layer16 l = make_rnn_layer16(params.batch, params.inputs, output, params.time_steps, activation, batch_normalize, params.net->adam);

    l.shortcut = option_find_int_quiet(options, "shortcut", 0);

    return l;
}
layer parse_gru(list *options, size_params params)
{
    int output = option_find_int(options, "output", 1);
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);

    layer l = make_gru_layer(params.batch, params.inputs, output, params.time_steps, batch_normalize, params.net->adam);
    l.tanh = option_find_int_quiet(options, "tanh", 0);

    return l;
}
layer16 parse_lstm16(list *options, size_params16 params)
{

    int output = option_find_int(options, "output", 1);
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);

    layer16 l = make_lstm_layer16(params.batch, params.inputs, output, params.time_steps, batch_normalize, params.net->adam);

    return l;
}
layer16 parse_lstmCJ16(list *options, size_params16 params)
{
    int input = option_find_int(options, "input", 1);
    int output = option_find_int(options, "output", 1);
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);
    layer16 l = make_lstm_layerCJ16(params.batch, input, output, params.time_steps, batch_normalize, params.net->adam, params.flag_vec);

    return l;
}
layer parse_lstm(list *options, size_params params)
{
    int output = option_find_int(options, "output", 1);
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);

    layer l = make_lstm_layer(params.batch, params.inputs, output, params.time_steps, batch_normalize, params.net->adam);

    return l;
}

layer parse_connected(list *options, size_params params)
{
    int output = option_find_int(options, "output", 1);
    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);
    int weights_transpose = option_find_int_quiet(options, "transpose", 0);
    // printf("connect transposeflg:%d\n", weights_transpose);

    layer l = make_connected_layer(params.batch, params.inputs, output, activation, batch_normalize, params.net->adam);
    l.transpose = weights_transpose;
    return l;
}
layer16 parse_connected16(list *options, size_params16 params)
{
    int output = option_find_int(options, "output", 1);
    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);
    int weights_transpose = option_find_int_quiet(options, "transpose", 0);
    printf("connect 16transposeflg:%d\n", weights_transpose);

    layer16 l = make_connected_layer16(params.batch, params.inputs, output, activation, batch_normalize, params.net->adam);
    l.transpose = weights_transpose;
    return l;
}
softmax_layer parse_softmax(list *options, size_params params)
{
    int groups = option_find_int_quiet(options, "groups", 1);
    softmax_layer layer = make_softmax_layer(params.batch, params.inputs, groups);
    layer.temperature = option_find_float_quiet(options, "temperature", 1);
    char *tree_file = option_find_str(options, "tree", 0);
    if (tree_file)
        layer.softmax_tree = read_tree(tree_file);

    layer.w = params.w;
    layer.h = params.h;
    layer.c = params.c;

    layer.spatial = option_find_float_quiet(options, "spatial", 0);

    return layer;
}
softmax_layer16 parse_softmaxCJ16(list *options, size_params16 params)
{

    int groups = option_find_int_quiet(options, "groups", 1);
    int inputs = option_find_int_quiet(options, "input", 1);
    softmax_layer16 layer = make_softmax_layerCJ16(params.batch, inputs, groups);
    layer.temperature = option_find_float_quiet(options, "temperature", 1);

    layer.w = params.w;
    layer.h = params.h;
    layer.c = params.c;

    layer.spatial = option_find_float_quiet(options, "spatial", 0);

    return layer;
}
softmax_layer16 parse_softmax16(list *options, size_params16 params)
{

    int groups = option_find_int_quiet(options, "groups", 1);
    softmax_layer16 layer = make_softmax_layer16(params.batch, params.inputs, groups);
    layer.temperature = option_find_float_quiet(options, "temperature", 1);
    printf("##parse temperature:%f,%f\n", (double)layer.temperature, layer.temperature);

    layer.w = params.w;
    layer.h = params.h;
    layer.c = params.c;

    layer.spatial = option_find_float_quiet(options, "spatial", 0);

    return layer;
}
layer parse_region(list *options, size_params params)
{
    int coords = option_find_int(options, "coords", 4);
    int classes = option_find_int(options, "classes", 20);
    int num = option_find_int(options, "num", 1);

    layer l = make_region_layer(params.batch, params.w, params.h, num, classes, coords);
    assert(l.outputs == params.inputs);

    l.log = option_find_int_quiet(options, "log", 0);
    l.sqrt = option_find_int_quiet(options, "sqrt", 0);

    l.softmax = option_find_int(options, "softmax", 0);
    l.background = option_find_int_quiet(options, "background", 0);
    l.max_boxes = option_find_int_quiet(options, "max", 30);
    l.jitter = option_find_float(options, "jitter", .2);
    l.rescore = option_find_int_quiet(options, "rescore", 0);

    l.thresh = option_find_float(options, "thresh", .5);
    l.classfix = option_find_int_quiet(options, "classfix", 0);
    l.absolute = option_find_int_quiet(options, "absolute", 0);
    l.random = option_find_int_quiet(options, "random", 0);

    l.coord_scale = option_find_float(options, "coord_scale", 1);
    l.object_scale = option_find_float(options, "object_scale", 1);
    l.noobject_scale = option_find_float(options, "noobject_scale", 1);
    l.mask_scale = option_find_float(options, "mask_scale", 1);
    l.class_scale = option_find_float(options, "class_scale", 1);
    l.bias_match = option_find_int_quiet(options, "bias_match", 0);

    char *tree_file = option_find_str(options, "tree", 0);
    if (tree_file)
        l.softmax_tree = read_tree(tree_file);
    char *map_file = option_find_str(options, "map", 0);
    if (map_file)
        l.map = read_map(map_file);

    char *a = option_find_str(options, "anchors", 0);
    if (a)
    {
        int len = strlen(a);
        int n = 1;
        int i;
        for (i = 0; i < len; ++i)
        {
            if (a[i] == ',')
                ++n;
        }
        for (i = 0; i < n; ++i)
        {
            float bias = atof(a);
            l.biases[i] = bias;
            a = strchr(a, ',') + 1;
        }
    }
    return l;
}
detection_layer parse_detection(list *options, size_params params)
{
    int coords = option_find_int(options, "coords", 1);
    int classes = option_find_int(options, "classes", 1);
    int rescore = option_find_int(options, "rescore", 0);
    int num = option_find_int(options, "num", 1);
    int side = option_find_int(options, "side", 7);
    detection_layer layer = make_detection_layer(params.batch, params.inputs, num, side, classes, coords, rescore);

    layer.softmax = option_find_int(options, "softmax", 0);
    layer.sqrt = option_find_int(options, "sqrt", 0);

    layer.max_boxes = option_find_int_quiet(options, "max", 30);
    layer.coord_scale = option_find_float(options, "coord_scale", 1);
    layer.forced = option_find_int(options, "forced", 0);
    layer.object_scale = option_find_float(options, "object_scale", 1);
    layer.noobject_scale = option_find_float(options, "noobject_scale", 1);
    layer.class_scale = option_find_float(options, "class_scale", 1);
    layer.jitter = option_find_float(options, "jitter", .2);
    layer.random = option_find_int_quiet(options, "random", 0);
    layer.reorg = option_find_int_quiet(options, "reorg", 0);
    return layer;
}
cost_layer16 parse_cost16(list *options, size_params16 params)
{
    char *type_s = option_find_str(options, "type", "sse");
    COST_TYPE type = get_cost_type(type_s);

    cost_layer16 layer = make_cost_layer16(params.batch, params.inputs, type, 1);

    layer.ratio = 0;
    layer.noobject_scale = 1;
    layer.thresh = 0;
    return layer;
}

cost_layer parse_cost(list *options, size_params params)
{
    char *type_s = option_find_str(options, "type", "sse");
    COST_TYPE type = get_cost_type(type_s);
    float scale = option_find_float_quiet(options, "scale", 1);
    cost_layer layer = make_cost_layer(params.batch, params.inputs, type, scale);
    layer.ratio = option_find_float_quiet(options, "ratio", 0);
    layer.noobject_scale = option_find_float_quiet(options, "noobj", 1);
    layer.thresh = option_find_float_quiet(options, "thresh", 0);
    return layer;
}

crop_layer parse_crop(list *options, size_params params)
{
    int crop_height = option_find_int(options, "crop_height", 1);
    int crop_width = option_find_int(options, "crop_width", 1);
    int flip = option_find_int(options, "flip", 0);
    float angle = option_find_float(options, "angle", 0);
    float saturation = option_find_float(options, "saturation", 1);
    float exposure = option_find_float(options, "exposure", 1);

    int batch, h, w, c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch = params.batch;
    if (!(h && w && c))
        error("Layer before crop layer must output image.");

    int noadjust = option_find_int_quiet(options, "noadjust", 0);

    crop_layer l = make_crop_layer(batch, h, w, c, crop_height, crop_width, flip, angle, saturation, exposure);
    l.shift = option_find_float(options, "shift", 0);
    l.noadjust = noadjust;
    return l;
}

crop_layer16 parse_crop16(list *options, size_params16 params)
{
    int crop_height = option_find_int(options, "crop_height", 1);
    int crop_width = option_find_int(options, "crop_width", 1);
    int flip = option_find_int(options, "flip", 0);
    float angle = option_find_float(options, "angle", 0);
    float saturation = option_find_float(options, "saturation", 1);
    float exposure = option_find_float(options, "exposure", 1);

    int batch, h, w, c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch = params.batch;
    if (!(h && w && c))
        error("Layer before crop layer must output image.");

    int noadjust = option_find_int_quiet(options, "noadjust", 0);

    crop_layer16 l = make_crop_layer16(batch, h, w, c, crop_height, crop_width, flip, angle, saturation, exposure);
    l.shift = option_find_float(options, "shift", 0);
    l.noadjust = noadjust;
    return l;
}

layer parse_reorg(list *options, size_params params)
{
    int stride = option_find_int(options, "stride", 1);
    int reverse = option_find_int_quiet(options, "reverse", 0);
    int flatten = option_find_int_quiet(options, "flatten", 0);
    int extra = option_find_int_quiet(options, "extra", 0);

    int batch, h, w, c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch = params.batch;
    if (!(h && w && c))
        error("Layer before reorg layer must output image.");

    layer layer = make_reorg_layer(batch, w, h, c, stride, reverse, flatten, extra);
    return layer;
}
maxpool_layer16 parse_maxpool16(list *options, size_params16 params)
{
    int stride = option_find_int(options, "stride", 1);
    int size = option_find_int(options, "size", stride);
    int padding = option_find_int_quiet(options, "padding", (size - 1) / 2);

    int batch, h, w, c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch = params.batch;
    if (!(h && w && c))
        error("Layer before maxpool layer must output image.");

    maxpool_layer16 layer = make_maxpool_layer16(batch, h, w, c, size, stride, padding);
    return layer;
}
maxpool_layer parse_maxpool(list *options, size_params params)
{
    int stride = option_find_int(options, "stride", 1);
    int size = option_find_int(options, "size", stride);
    int padding = option_find_int_quiet(options, "padding", (size - 1) / 2);

    int batch, h, w, c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch = params.batch;
    if (!(h && w && c))
        error("Layer before maxpool layer must output image.");

    maxpool_layer layer = make_maxpool_layer(batch, h, w, c, size, stride, padding);
    return layer;
}
avgpool_layer16 parse_avgpool16(list *options, size_params16 params)
{
    int batch, w, h, c;
    w = params.w;
    h = params.h;
    c = params.c;
    batch = params.batch;
    if (!(h && w && c))
        error("Layer before avgpool layer must output image.");

    avgpool_layer16 layer = make_avgpool_layer16(batch, w, h, c);
    return layer;
}
avgpool_layer parse_avgpool(list *options, size_params params)
{
    int batch, w, h, c;
    w = params.w;
    h = params.h;
    c = params.c;
    batch = params.batch;
    if (!(h && w && c))
        error("Layer before avgpool layer must output image.");

    avgpool_layer layer = make_avgpool_layer(batch, w, h, c);
    return layer;
}

dropout_layer16 parse_dropout16(list *options, size_params16 params)
{
    float probability = option_find_float(options, "probability", .5);
    dropout_layer16 layer = make_dropout_layer16(params.batch, params.inputs, probability);
    layer.out_w = params.w;
    layer.out_h = params.h;
    layer.out_c = params.c;
    return layer;
}

dropout_layer parse_dropout(list *options, size_params params)
{
    float probability = option_find_float(options, "probability", .5);
    dropout_layer layer = make_dropout_layer(params.batch, params.inputs, probability);
    layer.out_w = params.w;
    layer.out_h = params.h;
    layer.out_c = params.c;
    return layer;
}

layer parse_normalization(list *options, size_params params)
{
    float alpha = option_find_float(options, "alpha", .0001);
    float beta = option_find_float(options, "beta", .75);
    float kappa = option_find_float(options, "kappa", 1);
    int size = option_find_int(options, "size", 5);
    layer l = make_normalization_layer(params.batch, params.w, params.h, params.c, size, alpha, beta, kappa);
    return l;
}

layer parse_batchnorm(list *options, size_params params)
{
    layer l = make_batchnorm_layer(params.batch, params.w, params.h, params.c);
    return l;
}
layer16 parse_shortcut16(list *options, size_params16 params, network16 *net)
{
    char *l = option_find(options, "from");
    int index = atoi(l);
    if (index < 0)
        index = params.index + index;

    int batch = params.batch;
    layer16 from = net->layers[index];

    layer16 s = make_shortcut_layer16(batch, index, params.w, params.h, params.c, from.out_w, from.out_h, from.out_c);

    char *activation_s = option_find_str(options, "activation", "linear");
    ACTIVATION activation = get_activation(activation_s);
    s.activation = activation;
    return s;
}

layer parse_shortcut(list *options, size_params params, network *net)
{
    char *l = option_find(options, "from");
    int index = atoi(l);
    if (index < 0)
        index = params.index + index;

    int batch = params.batch;
    layer from = net->layers[index];

    layer s = make_shortcut_layer(batch, index, params.w, params.h, params.c, from.out_w, from.out_h, from.out_c);

    char *activation_s = option_find_str(options, "activation", "linear");
    ACTIVATION activation = get_activation(activation_s);
    s.activation = activation;
    return s;
}

layer parse_activation(list *options, size_params params)
{
    char *activation_s = option_find_str(options, "activation", "linear");
    ACTIVATION activation = get_activation(activation_s);

    layer l = make_activation_layer(params.batch, params.inputs, activation);

    l.out_h = params.h;
    l.out_w = params.w;
    l.out_c = params.c;
    l.h = params.h;
    l.w = params.w;
    l.c = params.c;

    return l;
}

route_layer parse_route(list *options, size_params params, network *net)
{
    char *l = option_find(options, "layers");
    int len = strlen(l);
    if (!l)
        error("Route Layer must specify input layers");
    int n = 1;
    int i;
    for (i = 0; i < len; ++i)
    {
        if (l[i] == ',')
            ++n;
    }

    int *layers = calloc(n, sizeof(int));
    int *sizes = calloc(n, sizeof(int));
    for (i = 0; i < n; ++i)
    {
        int index = atoi(l);
        l = strchr(l, ',') + 1;
        if (index < 0)
            index = params.index + index;
        layers[i] = index;
        sizes[i] = net->layers[index].outputs;
    }
    int batch = params.batch;

    route_layer layer = make_route_layer(batch, n, layers, sizes);

    convolutional_layer first = net->layers[layers[0]];
    layer.out_w = first.out_w;
    layer.out_h = first.out_h;
    layer.out_c = first.out_c;
    for (i = 1; i < n; ++i)
    {
        int index = layers[i];
        convolutional_layer next = net->layers[index];
        if (next.out_w == first.out_w && next.out_h == first.out_h)
        {
            layer.out_c += next.out_c;
        }
        else
        {
            layer.out_h = layer.out_w = layer.out_c = 0;
        }
    }

    return layer;
}

route_layer16 parse_route16(list *options, size_params16 params, network16 *net)
{
    char *l = option_find(options, "layers");
    int len = strlen(l);
    if (!l)
        error("Route Layer must specify input layers");
    int n = 1;
    int i;
    for (i = 0; i < len; ++i)
    {
        if (l[i] == ',')
            ++n;
    }

    int *layers = calloc(n, sizeof(int));
    int *sizes = calloc(n, sizeof(int));
    for (i = 0; i < n; ++i)
    {
        int index = atoi(l);
        l = strchr(l, ',') + 1;
        if (index < 0)
            index = params.index + index;
        layers[i] = index;
        sizes[i] = net->layers[index].outputs;
    }
    int batch = params.batch;

    route_layer16 layer = make_route_layer16(batch, n, layers, sizes);

    convolutional_layer16 first = net->layers[layers[0]];
    layer.out_w = first.out_w;
    layer.out_h = first.out_h;
    layer.out_c = first.out_c;
    for (i = 1; i < n; ++i)
    {
        int index = layers[i];
        convolutional_layer16 next = net->layers[index];
        if (next.out_w == first.out_w && next.out_h == first.out_h)
        {
            layer.out_c += next.out_c;
        }
        else
        {
            layer.out_h = layer.out_w = layer.out_c = 0;
        }
    }

    return layer;
}

learning_rate_policy get_policy(char *s)
{
    if (strcmp(s, "random") == 0)
        return RANDOM;
    if (strcmp(s, "poly") == 0)
        return POLY;
    if (strcmp(s, "constant") == 0)
        return CONSTANT;
    if (strcmp(s, "step") == 0)
        return STEP;
    if (strcmp(s, "exp") == 0)
        return EXP;
    if (strcmp(s, "sigmoid") == 0)
        return SIG;
    if (strcmp(s, "steps") == 0)
        return STEPS;
    fprintf(stderr, "Couldn't find policy %s, going with constant\n", s);
    return CONSTANT;
}
void parse_net_options16(list *options, network16 *net)
{
    net->batch = option_find_int(options, "batch", 1);
    net->time_steps = option_find_int_quiet(options, "time_steps", 1);

    net->adam = option_find_int_quiet(options, "adam", 0);

    net->h = option_find_int_quiet(options, "height", 0);
    net->w = option_find_int_quiet(options, "width", 0);
    net->c = option_find_int_quiet(options, "channels", 0);
    net->inputs = option_find_int_quiet(options, "inputs", net->h * net->w * net->c);

    net->flag_vec = option_find_int_quiet(options, "flag_vec", 0);
    net->transpose = option_find_int_quiet(options, "transpose", 0); //printf("transposeflg:%d\n",net->transpose);
    if (!net->inputs && !(net->h && net->w && net->c))
        error("No input parameters supplied");
}
void parse_net_options(list *options, network *net)
{
    //TL: fixing transpose in alexnet
    net->flag_vec = option_find_int_quiet(options, "flag_vec", 0);
    net->transpose = option_find_int_quiet(options, "transpose", 0);

    net->batch = option_find_int(options, "batch", 1);
    net->learning_rate = option_find_float(options, "learning_rate", .001);
    net->momentum = option_find_float(options, "momentum", .9);
    net->decay = option_find_float(options, "decay", .0001);
    int subdivs = option_find_int(options, "subdivisions", 1);
    net->time_steps = option_find_int_quiet(options, "time_steps", 1);
    net->notruth = option_find_int_quiet(options, "notruth", 0);
    net->batch /= subdivs;
    net->batch *= net->time_steps;
    net->subdivisions = subdivs;
    net->random = option_find_int_quiet(options, "random", 0);
    net->transpose = option_find_int_quiet(options, "transpose", 0);

    net->adam = option_find_int_quiet(options, "adam", 0);
    if (net->adam)
    {
        net->B1 = option_find_float(options, "B1", .9);
        net->B2 = option_find_float(options, "B2", .999);
        net->eps = option_find_float(options, "eps", .0000001);
    }

    net->h = option_find_int_quiet(options, "height", 0);
    net->w = option_find_int_quiet(options, "width", 0);
    net->c = option_find_int_quiet(options, "channels", 0);
    net->inputs = option_find_int_quiet(options, "inputs", net->h * net->w * net->c);
    net->max_crop = option_find_int_quiet(options, "max_crop", net->w * 2);
    net->min_crop = option_find_int_quiet(options, "min_crop", net->w);
    net->max_ratio = option_find_float_quiet(options, "max_ratio", (float)net->max_crop / net->w);
    net->min_ratio = option_find_float_quiet(options, "min_ratio", (float)net->min_crop / net->w);
    net->center = option_find_int_quiet(options, "center", 0);

    net->angle = option_find_float_quiet(options, "angle", 0);
    net->aspect = option_find_float_quiet(options, "aspect", 1);
    net->saturation = option_find_float_quiet(options, "saturation", 1);
    net->exposure = option_find_float_quiet(options, "exposure", 1);
    net->hue = option_find_float_quiet(options, "hue", 0);

    if (!net->inputs && !(net->h && net->w && net->c))
        error("No input parameters supplied");

    char *policy_s = option_find_str(options, "policy", "constant");
    net->policy = get_policy(policy_s);
    net->burn_in = option_find_int_quiet(options, "burn_in", 0);
    net->power = option_find_float_quiet(options, "power", 4);
    if (net->policy == STEP)
    {
        net->step = option_find_int(options, "step", 1);
        net->scale = option_find_float(options, "scale", 1);
    }
    else if (net->policy == STEPS)
    {
        char *l = option_find(options, "steps");
        char *p = option_find(options, "scales");
        if (!l || !p)
            error("STEPS policy must have steps and scales in cfg file");

        int len = strlen(l);
        int n = 1;
        int i;
        for (i = 0; i < len; ++i)
        {
            if (l[i] == ',')
                ++n;
        }
        int *steps = calloc(n, sizeof(int));
        float *scales = calloc(n, sizeof(float));
        for (i = 0; i < n; ++i)
        {
            int step = atoi(l);
            float scale = atof(p);
            l = strchr(l, ',') + 1;
            p = strchr(p, ',') + 1;
            steps[i] = step;
            scales[i] = scale;
        }
        net->scales = scales;
        net->steps = steps;
        net->num_steps = n;
    }
    else if (net->policy == EXP)
    {
        net->gamma = option_find_float(options, "gamma", 1);
    }
    else if (net->policy == SIG)
    {
        net->gamma = option_find_float(options, "gamma", 1);
        net->step = option_find_int(options, "step", 1);
    }
    else if (net->policy == POLY || net->policy == RANDOM)
    {
    }
    net->max_batches = option_find_int(options, "max_batches", 0);
}

int is_network(section *s)
{
    return (strcmp(s->type, "[net]") == 0 || strcmp(s->type, "[network]") == 0);
}

network *parse_network_cfg(char *filename)
{
    list *sections = read_cfg(filename);
    node *n = sections->front;
    if (!n)
        error("Config file has no sections");
    network *net = make_network(sections->size - 1);
    net->gpu_index = gpu_index;
    size_params params;

    section *s = (section *)n->val;
    list *options = s->options;
    if (!is_network(s))
        error("First section must be [net] or [network]");
    parse_net_options(options, net);

    params.h = net->h;
    params.w = net->w;
    params.c = net->c;
    params.inputs = net->inputs;
    params.batch = net->batch;
    params.time_steps = net->time_steps;
    params.net = net;

    size_t workspace_size = 0;
    n = n->next;
    int count = 0;
    free_section(s);
    fprintf(stderr, "layer     filters    size              input                output\n");
    while (n)
    {
        params.index = count;
        fprintf(stderr, "%5d ", count);
        s = (section *)n->val;
        options = s->options;
        layer l = {0};
        LAYER_TYPE lt = string_to_layer_type(s->type);
        if (lt == CONVOLUTIONAL)
        {
            l = parse_convolutional(options, params);
        }
        else if (lt == DECONVOLUTIONAL)
        {
            l = parse_deconvolutional(options, params);
        }
        else if (lt == LOCAL)
        {
            l = parse_local(options, params);
        }
        else if (lt == ACTIVE)
        {
            l = parse_activation(options, params);
        }
        else if (lt == RNN)
        {
            l = parse_rnn(options, params);
        }
        else if (lt == GRU)
        {
            l = parse_gru(options, params);
        }
        else if (lt == LSTM)
        {
            l = parse_lstm(options, params);
        }
        else if (lt == CRNN)
        {
            l = parse_crnn(options, params);
        }
        else if (lt == CONNECTED)
        {
            l = parse_connected(options, params);
        }
        else if (lt == CROP)
        {
            l = parse_crop(options, params);
        }
        else if (lt == COST)
        {
            l = parse_cost(options, params);
        }
        else if (lt == REGION)
        {
            l = parse_region(options, params);
        }
        else if (lt == DETECTION)
        {
            l = parse_detection(options, params);
        }
        else if (lt == SOFTMAX)
        {
            l = parse_softmax(options, params);
            net->hierarchy = l.softmax_tree;
        }
        else if (lt == NORMALIZATION)
        {
            l = parse_normalization(options, params);
        }
        else if (lt == BATCHNORM)
        {
            l = parse_batchnorm(options, params);
        }
        else if (lt == MAXPOOL)
        {
            l = parse_maxpool(options, params);
        }
        else if (lt == REORG)
        {
            l = parse_reorg(options, params);
        }
        else if (lt == AVGPOOL)
        {
            l = parse_avgpool(options, params);
        }
        else if (lt == ROUTE)
        {
            l = parse_route(options, params, net);
        }
        else if (lt == SHORTCUT)
        {
            l = parse_shortcut(options, params, net);
        }
        else if (lt == DROPOUT)
        {
            l = parse_dropout(options, params);
            l.output = net->layers[count - 1].output;
            l.delta = net->layers[count - 1].delta;
#ifdef GPU
            l.output_gpu = net->layers[count - 1].output_gpu;
            l.delta_gpu = net->layers[count - 1].delta_gpu;
#endif
        }
        else
        {
            fprintf(stderr, "Type not recognized: %s\n", s->type);
        }
        l.truth = option_find_int_quiet(options, "truth", 0);
        l.onlyforward = option_find_int_quiet(options, "onlyforward", 0);
        l.stopbackward = option_find_int_quiet(options, "stopbackward", 0);
        l.dontload = option_find_int_quiet(options, "dontload", 0);
        l.dontloadscales = option_find_int_quiet(options, "dontloadscales", 0);
        l.learning_rate_scale = option_find_float_quiet(options, "learning_rate", 1);
        l.smooth = option_find_float_quiet(options, "smooth", 0);
        option_unused(options);
        net->layers[count] = l;
        if (l.workspace_size > workspace_size)
            workspace_size = l.workspace_size;
        free_section(s);
        n = n->next;
        ++count;
        if (n)
        {
            params.h = l.out_h;
            params.w = l.out_w;
            params.c = l.out_c;
            params.inputs = l.outputs;
        }
    }
    free_list(sections);
    layer out = get_network_output_layer(net);
    net->outputs = out.outputs;
    net->truths = out.outputs;
    if (net->layers[net->n - 1].truths)
        net->truths = net->layers[net->n - 1].truths;
    net->output = out.output;
    net->input = calloc(net->inputs * net->batch, sizeof(float));
    net->truth = calloc(net->truths * net->batch, sizeof(float));
#ifdef GPU
    net->output_gpu = out.output_gpu;
    net->input_gpu = cuda_make_array(net->input, net->inputs * net->batch);
    net->truth_gpu = cuda_make_array(net->truth, net->truths * net->batch);
#endif
    if (workspace_size)
    {
        //printf("%ld\n", workspace_size);
#ifdef GPU
        if (gpu_index >= 0)
        {
            net->workspace = cuda_make_array(0, (workspace_size - 1) / sizeof(float) + 1);
        }
        else
        {
            net->workspace = calloc(1, workspace_size);
        }
#else
        net->workspace = calloc(1, workspace_size);
#endif
    }
    return net;
}

network16 *parse_network_cfg_CJ16(char *filename)
{
    list *sections = read_cfg(filename);
    node *n = sections->front;
    if (!n)
        error("Config file has no sections");
    network16 *net = make_network16(sections->size - 1);
    size_params16 params;

    section *s = (section *)n->val;
    list *options = s->options;
    if (!is_network(s))
        error("First section must be [net] or [network]");
    parse_net_options16(options, net);

    params.h = net->h;
    params.w = net->w;
    params.c = net->c;
    params.inputs = net->inputs;

    params.batch = net->batch;
    params.time_steps = net->time_steps;
    params.net = net;

    params.flag_vec = net->flag_vec;
    size_t workspace_size = 0;
    if (params.flag_vec == 0)
        printf("calculate type    : scalar\n");
    else
        printf("calculate type    : vector\n");

    n = n->next;
    int count = 0;
    free_section(s);
    fprintf(stderr, "layer     filters    size              input                output\n");
    while (n)
    {
        params.index = count;
        fprintf(stderr, "%5d ", count);
        s = (section *)n->val;
        options = s->options;
        layer16 l = {0};
        LAYER_TYPE lt = string_to_layer_type(s->type);

        if (lt == RNN)
        {
            l = parse_rnn16(options, params);
        }
        else if (lt == CONNECTED)
        {
            l = parse_connected16(options, params);
        }
        else if (lt == SOFTMAX)
        {
            l = parse_softmax16(options, params);
        }
        else if (lt == SOFTMAXCJ)
        {
            l = parse_softmaxCJ16(options, params);
        }
        else if (lt == COST)
        {
            l = parse_cost16(options, params);
        }
        else if (lt == LSTM)
        {
            l = parse_lstm16(options, params);
        }
        else if (lt == LSTMCJ)
        {
            l = parse_lstmCJ16(options, params);
        }
        else
        {
            fprintf(stderr, "Type not recognized: %s\n", s->type);
        }
        l.truth = option_find_int_quiet(options, "truth", 0);
        l.onlyforward = option_find_int_quiet(options, "onlyforward", 0);
        l.stopbackward = option_find_int_quiet(options, "stopbackward", 0);
        l.dontload = option_find_int_quiet(options, "dontload", 0);
        l.dontloadscales = option_find_int_quiet(options, "dontloadscales", 0);
        l.learning_rate_scale = option_find_float_quiet(options, "learning_rate", 1);
        l.smooth = option_find_float_quiet(options, "smooth", 0);
        option_unused(options);
        net->layers[count] = l;
        if (l.workspace_size > workspace_size)
            workspace_size = l.workspace_size;
        free_section(s);
        n = n->next;
        ++count;
        if (n)
        {
            params.h = l.out_h;
            params.w = l.out_w;
            params.c = l.out_c;
            params.inputs = l.outputs;
        }
    }
    free_list(sections);
    layer16 out = get_network_output_layer16(net);
    net->outputs = out.outputs;
    net->truths = out.outputs;
    net->output = out.output;

    net->h_cpu = out.h_cpu;

    net->input = calloc(net->inputs * net->batch, sizeof(FLT));
    net->truth = calloc(net->truths * net->batch, sizeof(FLT));

    if (workspace_size)
    {
        //printf("%ld\n", workspace_size);
        net->workspace = calloc(1, workspace_size);
    }
    return net;
}

network16 *parse_network_cfg16(char *filename)
{
    list *sections = read_cfg(filename);
    node *n = sections->front;
    if (!n)
        error("Config file has no sections");
    network16 *net = make_network16(sections->size - 1);
    size_params16 params;

    section *s = (section *)n->val;
    list *options = s->options;
    if (!is_network(s))
        error("First section must be [net] or [network]");
    parse_net_options16(options, net);

    params.h = net->h;
    params.w = net->w;
    params.c = net->c;
    params.inputs = net->inputs;
    params.batch = net->batch;
    params.time_steps = net->time_steps;
    params.net = net;

    size_t workspace_size = 0;
    n = n->next;
    int count = 0;
    free_section(s);
    fprintf(stderr, "layer     filters    size              input                output\n");
    while (n)
    {
        params.index = count;
        fprintf(stderr, "%5d ", count);
        s = (section *)n->val;
        options = s->options;
        layer16 l = {0};
        LAYER_TYPE lt = string_to_layer_type(s->type);

        if (lt == RNN)
        {
            l = parse_rnn16(options, params);
        }
        else if (lt == CONVOLUTIONAL)
        {
            l = parse_convolutional16(options, params);
        }
        else if (lt == CONNECTED)
        {
            l = parse_connected16(options, params);
        }
        else if (lt == SOFTMAX)
        {
            l = parse_softmax16(options, params);
        }
        else if (lt == SOFTMAXCJ)
        {
            l = parse_softmaxCJ16(options, params);
        }
        else if (lt == COST)
        {
            l = parse_cost16(options, params);
        }
        else if (lt == LSTM)
        {
            l = parse_lstm16(options, params);
        }
        else if (lt == LSTMCJ)
        {
            l = parse_lstmCJ16(options, params);
        }
        else if (lt == MAXPOOL)
        {
            l = parse_maxpool16(options, params);
        }
        else if (lt == AVGPOOL)
        {
            l = parse_avgpool16(options, params);
        }
        else if (lt == ROUTE)
        {
            l = parse_route16(options, params, net);
        }
        else if (lt == SHORTCUT)
        {
            l = parse_shortcut16(options, params, net);
        }
        else if (lt == DROPOUT)
        {
            l = parse_dropout16(options, params);
            l.output = net->layers[count - 1].output;
            l.delta = net->layers[count - 1].delta;
        }
        else if (lt == CROP)
        {
            l = parse_crop16(options, params);
        }
        else
        {
            fprintf(stderr, "Type not recognized: %s\n", s->type);
        }
        l.truth = option_find_int_quiet(options, "truth", 0);
        l.onlyforward = option_find_int_quiet(options, "onlyforward", 0);
        l.stopbackward = option_find_int_quiet(options, "stopbackward", 0);
        l.dontload = option_find_int_quiet(options, "dontload", 0);
        l.dontloadscales = option_find_int_quiet(options, "dontloadscales", 0);
        l.learning_rate_scale = option_find_float_quiet(options, "learning_rate", 1);
        l.smooth = option_find_float_quiet(options, "smooth", 0);
        option_unused(options);
        net->layers[count] = l;
        if (l.workspace_size > workspace_size)
            workspace_size = l.workspace_size;
        free_section(s);
        n = n->next;
        ++count;
        if (n)
        {
            params.h = l.out_h;
            params.w = l.out_w;
            params.c = l.out_c;
            params.inputs = l.outputs;
        }
    }
    free_list(sections);
    layer16 out = get_network_output_layer16(net);
    net->outputs = out.outputs;
    net->truths = out.outputs;
    net->output = out.output;

    net->input = calloc(net->inputs * net->batch, sizeof(FLT));
    net->truth = calloc(net->truths * net->batch, sizeof(FLT));
    if (workspace_size)
    {
        //printf("%ld\n", workspace_size);
        net->workspace = calloc(1, workspace_size);
    }

    //for(int i=0;i< net->n;i++){
    //printf("net.layer[%d].delta:%lf\n",i,net->layers[i].delta);}

    return net;
}

list *read_cfg(char *filename)
{
    FILE *file = fopen(filename, "r");
    printf("%s\n", filename);
    if (file == 0)
        file_error(filename);
    char *line;
    int nu = 0;
    list *options = make_list();
    section *current = 0;
    while ((line = fgetl(file)) != 0)
    {
        ++nu;
        strip(line);
        switch (line[0])
        {
        case '[':
            current = malloc(sizeof(section));
            list_insert(options, current);
            current->options = make_list();
            current->type = line;
            break;
        case '\0':
        case '#':
        case ';':
            free(line);
            break;
        default:
            if (!read_option(line, current->options))
            {
                fprintf(stderr, "Config file error line %d, could parse: %s\n", nu, line);
                free(line);
            }
            break;
        }
    }
    fclose(file);
    printf("readdd cfg end\n");
    return options;
}

void save_convolutional_weights_binary(layer l, FILE *fp)
{
#ifdef GPU
    if (gpu_index >= 0)
    {
        pull_convolutional_layer(l);
    }
#endif
    binarize_weights(l.weights, l.n, l.c * l.size * l.size, l.binary_weights);
    int size = l.c * l.size * l.size;
    int i, j, k;
    fwrite(l.biases, sizeof(float), l.n, fp);
    if (l.batch_normalize)
    {
        fwrite(l.scales, sizeof(float), l.n, fp);
        fwrite(l.rolling_mean, sizeof(float), l.n, fp);
        fwrite(l.rolling_variance, sizeof(float), l.n, fp);
    }
    for (i = 0; i < l.n; ++i)
    {
        float mean = l.binary_weights[i * size];
        if (mean < 0)
            mean = -mean;
        fwrite(&mean, sizeof(float), 1, fp);
        for (j = 0; j < size / 8; ++j)
        {
            int index = i * size + j * 8;
            unsigned char c = 0;
            for (k = 0; k < 8; ++k)
            {
                if (j * 8 + k >= size)
                    break;
                if (l.binary_weights[index + k] > 0)
                    c = (c | 1 << k);
            }
            fwrite(&c, sizeof(char), 1, fp);
        }
    }
}

void save_convolutional_weights(layer l, FILE *fp)
{
    if (l.binary)
    {
        //save_convolutional_weights_binary(l, fp);
        //return;
    }
#ifdef GPU
    if (gpu_index >= 0)
    {
        pull_convolutional_layer(l);
    }
#endif
    int num = l.nweights;
    fwrite(l.biases, sizeof(float), l.n, fp);
    if (l.batch_normalize)
    {
        fwrite(l.scales, sizeof(float), l.n, fp);
        fwrite(l.rolling_mean, sizeof(float), l.n, fp);
        fwrite(l.rolling_variance, sizeof(float), l.n, fp);
    }
    fwrite(l.weights, sizeof(float), num, fp);
}

void save_batchnorm_weights(layer l, FILE *fp)
{
#ifdef GPU
    if (gpu_index >= 0)
    {
        pull_batchnorm_layer(l);
    }
#endif
    fwrite(l.scales, sizeof(float), l.c, fp);
    fwrite(l.rolling_mean, sizeof(float), l.c, fp);
    fwrite(l.rolling_variance, sizeof(float), l.c, fp);
}

void save_connected_weights(layer l, FILE *fp)
{
#ifdef GPU
    if (gpu_index >= 0)
    {
        pull_connected_layer(l);
    }
#endif
    fwrite(l.biases, sizeof(float), l.outputs, fp);
    fwrite(l.weights, sizeof(float), l.outputs * l.inputs, fp);
    if (l.batch_normalize)
    {
        fwrite(l.scales, sizeof(float), l.outputs, fp);
        fwrite(l.rolling_mean, sizeof(float), l.outputs, fp);
        fwrite(l.rolling_variance, sizeof(float), l.outputs, fp);
    }
}

void save_weights_upto(network *net, char *filename, int cutoff)
{
#ifdef GPU
    if (net->gpu_index >= 0)
    {
        cuda_set_device(net->gpu_index);
    }
#endif
    fprintf(stderr, "Saving weights to %s\n", filename);
    FILE *fp = fopen(filename, "wb");
    if (!fp)
        file_error(filename);

    int major = 0;
    int minor = 2;
    int revision = 0;
    fwrite(&major, sizeof(int), 1, fp);
    fwrite(&minor, sizeof(int), 1, fp);
    fwrite(&revision, sizeof(int), 1, fp);
    fwrite(net->seen, sizeof(size_t), 1, fp);

    int i;
    for (i = 0; i < net->n && i < cutoff; ++i)
    {
        layer l = net->layers[i];
        if (l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL)
        {
            save_convolutional_weights(l, fp);
        }
        if (l.type == CONNECTED)
        {
            save_connected_weights(l, fp);
        }
        if (l.type == BATCHNORM)
        {
            save_batchnorm_weights(l, fp);
        }
        if (l.type == RNN)
        {
            save_connected_weights(*(l.input_layer), fp);
            save_connected_weights(*(l.self_layer), fp);
            save_connected_weights(*(l.output_layer), fp);
        }
        if (l.type == LSTM)
        {
            save_connected_weights(*(l.wi), fp);
            save_connected_weights(*(l.wf), fp);
            save_connected_weights(*(l.wo), fp);
            save_connected_weights(*(l.wg), fp);
            save_connected_weights(*(l.ui), fp);
            save_connected_weights(*(l.uf), fp);
            save_connected_weights(*(l.uo), fp);
            save_connected_weights(*(l.ug), fp);
        }
        if (l.type == GRU)
        {
            if (1)
            {
                save_connected_weights(*(l.wz), fp);
                save_connected_weights(*(l.wr), fp);
                save_connected_weights(*(l.wh), fp);
                save_connected_weights(*(l.uz), fp);
                save_connected_weights(*(l.ur), fp);
                save_connected_weights(*(l.uh), fp);
            }
            else
            {
                save_connected_weights(*(l.reset_layer), fp);
                save_connected_weights(*(l.update_layer), fp);
                save_connected_weights(*(l.state_layer), fp);
            }
        }
        if (l.type == CRNN)
        {
            save_convolutional_weights(*(l.input_layer), fp);
            save_convolutional_weights(*(l.self_layer), fp);
            save_convolutional_weights(*(l.output_layer), fp);
        }
        if (l.type == LOCAL)
        {
#ifdef GPU
            if (gpu_index >= 0)
            {
                pull_local_layer(l);
            }
#endif
            int locations = l.out_w * l.out_h;
            int size = l.size * l.size * l.c * l.n * locations;
            fwrite(l.biases, sizeof(float), l.outputs, fp);
            fwrite(l.weights, sizeof(float), size, fp);
        }
    }
    fclose(fp);
}
void save_weights(network *net, char *filename)
{
    save_weights_upto(net, filename, net->n);
}
void transpose_matrix16(FLT *a, int rows, int cols)
{
    FLT *transpose = calloc(rows * cols, sizeof(FLT));
    int x, y;
    for (x = 0; x < rows; ++x)
    {
        for (y = 0; y < cols; ++y)
        {
            transpose[y * rows + x] = a[x * cols + y];
        }
    }
    memcpy(a, transpose, rows * cols * sizeof(FLT));
    free(transpose);
}
void transpose_matrix(float *a, int rows, int cols)
{
    float *transpose = calloc(rows * cols, sizeof(float));
    int x, y;
    for (x = 0; x < rows; ++x)
    {
        for (y = 0; y < cols; ++y)
        {
            transpose[y * rows + x] = a[x * cols + y];
        }
    }
    memcpy(a, transpose, rows * cols * sizeof(float));
    free(transpose);
}

void load_connected_weights16(layer16 l, FILE *fp, int transpose)
{

    //float f1024[4096],f1024_1024[4096*4096];
    //printf("WHEREAMI:\t[loaded_conn_weights_16]\tsum = %d\n", l.outputs * l.inputs);

    float *param_tmp = (float *)calloc(l.outputs, sizeof(float));
    float *weights_tmp = (float *)calloc(l.outputs * l.inputs, sizeof(float));

    //bias convertion from float32 to des.
    fread(param_tmp, sizeof(float), l.outputs, fp);
    for (int i = 0; i < l.outputs; i++)
    {
        l.biases[i] = (FLT)param_tmp[i];
    }
    //weights convertion from float32 to des.
    fread(weights_tmp, sizeof(float), l.outputs * l.inputs, fp);
    for (int i = 0; i < l.inputs * l.outputs; i++)
    {
        l.weights[i] = (FLT)weights_tmp[i];
        //printf("weights:%f ",l_temp.weights[i]);
    } //printf("\n");
    if (transpose)
    {
        // printf("NOTE:\t[load_connected_weights_16]\tTRANSPOSE \n");
        transpose_matrix16(l.weights, l.inputs, l.outputs);
    }
    if (l.transpose)
    {
        // printf("NOTE:\t[load_connected_weights_16]\tl.transpose = 1\n");
        transpose_matrix16(l.weights, l.outputs, l.inputs);
    }

    if (l.batch_normalize && (!l.dontloadscales))
    {
        // printf("NOTE:\t[load_connected_weights_16]\t(l.batch_normalize && (!l.dontloadscales))\n");
        fread(param_tmp, sizeof(float), l.outputs, fp);
        for (int i = 0; i < l.outputs; i++)
        {
            l.scales[i] = (FLT)param_tmp[i];
        }

        fread(param_tmp, sizeof(float), l.outputs, fp);
        for (int i = 0; i < l.outputs; i++)
        {
            l.rolling_mean[i] = (FLT)param_tmp[i];
        }

        fread(param_tmp, sizeof(float), l.outputs, fp);
        for (int i = 0; i < l.outputs; i++)
        {
            l.rolling_variance[i] = (FLT)param_tmp[i];
        }
        // fread(l_temp.scales, sizeof(float), l_temp.outputs, fp);
        // fread(l_temp.rolling_mean, sizeof(float), l_temp.outputs, fp);
        // fread(l_temp.rolling_variance, sizeof(float), l_temp.outputs, fp);
    }
}

void load_connected_weights(layer l, FILE *fp, int transpose)
{

    fread(l.biases, sizeof(float), l.outputs, fp);

#ifdef CSR
    printf("----CSR in Conn---- outputs * inputs (%d*%d)=%d\n", l.outputs, l.inputs, l.outputs * l.inputs);
    float *wei = calloc(l.outputs * l.inputs, sizeof(float));
    fread(wei, sizeof(float), l.outputs * l.inputs, fp);
    // l.spmt = mat2csr_divide(wei, l.outputs, l.inputs, l.outputs, l.inputs);
    l.spmt = mat2csr_divide(wei, l.outputs, l.inputs, 1024, 256);
    // memcpy(l.weights, wei, sizeof(float) * (l.outputs * l.inputs));
    memcpy(l.weights, csr2mat_comb(l.spmt), sizeof(float) * (l.outputs * l.inputs));
#else
    fread(l.weights, sizeof(float), l.outputs * l.inputs, fp);
#endif

    // printf("connected layer with %d*%d\n", l.outputs, l.inputs);
    // for (int j = 0; j < 3; j++) //row
    // {
    //     for (int k = 0; k < 30; k++) //col
    //         printf("%.1f ", l.weights[j * l.inputs + k]);
    //     printf("\n");
    // }

    //TL 1017 adding convert from float to int, then back to float
    for (int i = 0; i < l.outputs * l.inputs; i++)
    {
        l.con_max = l.wei_max > l.weights[i] ? l.con_max : l.weights[i];
        l.con_min = l.wei_min < l.weights[i] ? l.con_min : l.weights[i];
    }

    if (transpose)
    {
        // printf("NOTE:\t[load_connected_weights]\tTRANSPOSE \n");

        transpose_matrix(l.weights, l.inputs, l.outputs);
    }
    if (l.transpose)
    {
        // printf("NOTE:\t[load_connected_weights]\tl.transpose = 1\n");

        transpose_matrix(l.weights, l.outputs, l.inputs);
    }

    //printf("Biases: %f mean %f variance\n", mean_array(l.biases, l.outputs), variance_array(l.biases, l.outputs));
    //printf("Weights: %f mean %f variance\n", mean_array(l.weights, l.outputs*l.inputs), variance_array(l.weights, l.outputs*l.inputs));
    if (l.batch_normalize && (!l.dontloadscales))
    {
        // printf("NOTE:\t[load_connected_weights]\t(l.batch_normalize && (!l.dontloadscales))\n");
        fread(l.scales, sizeof(float), l.outputs, fp);
        fread(l.rolling_mean, sizeof(float), l.outputs, fp);
        fread(l.rolling_variance, sizeof(float), l.outputs, fp);
        //printf("Scales: %f mean %f variance\n", mean_array(l.scales, l.outputs), variance_array(l.scales, l.outputs));
        //printf("rolling_mean: %f mean %f variance\n", mean_array(l.rolling_mean, l.outputs), variance_array(l.rolling_mean, l.outputs));
        //printf("rolling_variance: %f mean %f variance\n", mean_array(l.rolling_variance, l.outputs), variance_array(l.rolling_variance, l.outputs));
    }
#ifdef GPU
    if (gpu_index >= 0)
    {
        push_connected_layer(l);
    }
#endif
}

void load_batchnorm_weights16(layer16 l, FILE *fp)
{
    fread(l.scales, sizeof(FLT), l.c, fp);
    fread(l.rolling_mean, sizeof(FLT), l.c, fp);
    fread(l.rolling_variance, sizeof(FLT), l.c, fp);
}
void load_batchnorm_weights(layer l, FILE *fp)
{
    fread(l.scales, sizeof(float), l.c, fp);
    fread(l.rolling_mean, sizeof(float), l.c, fp);
    fread(l.rolling_variance, sizeof(float), l.c, fp);
#ifdef GPU
    if (gpu_index >= 0)
    {
        push_batchnorm_layer(l);
    }
#endif
}

void load_convolutional_weights_binary(layer l, FILE *fp)
{
    fread(l.biases, sizeof(float), l.n, fp);
    if (l.batch_normalize && (!l.dontloadscales))
    {
        fread(l.scales, sizeof(float), l.n, fp);
        fread(l.rolling_mean, sizeof(float), l.n, fp);
        fread(l.rolling_variance, sizeof(float), l.n, fp);
    }
    int size = l.c * l.size * l.size;
    int i, j, k;
    for (i = 0; i < l.n; ++i)
    {
        float mean = 0;
        fread(&mean, sizeof(float), 1, fp);
        for (j = 0; j < size / 8; ++j)
        {
            int index = i * size + j * 8;
            unsigned char c = 0;
            fread(&c, sizeof(char), 1, fp);
            for (k = 0; k < 8; ++k)
            {
                if (j * 8 + k >= size)
                    break;
                l.weights[index + k] = (c & 1 << k) ? mean : -mean;
            }
        }
    }
#ifdef GPU
    if (gpu_index >= 0)
    {
        push_convolutional_layer(l);
    }
#endif
}

void load_convolutional_weights16(layer16 l, FILE *fp)
{
    int i;
    if (l.binary)
    {
        //load_convolutional_weights_binary(l, fp);
        //return;
    }
    int num = l.nweights; //printf("nweights:%d\n",num);
    //printf("WHEREAMI:\t[load_convolutional_weights_16]\tload conv biases:%d\n", l.n);
    float *f_tmp = calloc(l.n, sizeof(float));
    if (!f_tmp)
    {
        printf("calloc failed\n");
        exit(1);
    }

    //printf("load convolutional weights16\n");
    fread(f_tmp, sizeof(float), l.n, fp);
    for (i = 0; i < l.n; i++)
    {
        l.biases[i] = (FLT)f_tmp[i];
    }

    if (l.batch_normalize && (!l.dontloadscales))
    {
        fread(f_tmp, sizeof(float), l.n, fp);
        for (i = 0; i < l.n; i++)
        {
            l.scales[i] = (FLT)f_tmp[i];
        }
        for (i = 0; i < l.n; i++)
        {
            l.bn_scale[i] = l.scales[i] / (sqrt(l.rolling_variance[i]) + .000001f);
            l.bn_bias[i] = l.biases[i] - l.scales[i] * l.rolling_mean[i] / (sqrt(l.rolling_variance[i]) + .000001f);
        }

        fread(f_tmp, sizeof(float), l.n, fp);
        for (i = 0; i < l.n; i++)
        {
            l.rolling_mean[i] = (FLT)f_tmp[i];
        }
        fread(f_tmp, sizeof(float), l.n, fp);
        for (i = 0; i < l.n; i++)
        {
            l.rolling_variance[i] = (FLT)f_tmp[i];
            // l.rolling_variance[i] = 1/(sqrt(l.rolling_variance[i]) + .000001f); //liuj2018025
            // l.rolling_varianceMultiscales[i] = l.rolling_variance[i] *l.scales[i];
        }
    }
    printf("loaded_conv_weights_16, sum = %d\n", num);

    float *f_tmp1 = (float *)calloc(num, sizeof(float));
    if (!f_tmp1)
    {
        printf("calloc failed\n");
        exit(1);
    }
    //printf("num:%d\n",num);
    fread(f_tmp1, sizeof(float), num, fp);
    for (i = 0; i < num; i++)
    {
        l.weights[i] = (FLT)f_tmp1[i];
    }
    //if(l.c == 3) scal_cpu(num, 1./256, l.weights, 1);
    if (l.flipped)
    {
        puts("l.flipped\n");
        transpose_matrix16(l.weights, l.c * l.size * l.size, l.n);
    }
}

void load_convolutional_weights(layer l, FILE *fp)
{
    if (l.binary)
    {
        //load_convolutional_weights_binary(l, fp);
        //return;
    }
    int num = l.nweights;
    //printf("WHEREAMI:\t[load_convolutional_weights]\tload conv biases:%d\n", l.n);
    fread(l.biases, sizeof(float), l.n, fp);
    // for (int j = 0; j < l.n; j++)
    //     printf("bias[%d]:%f",j,l.biases[j]);printf("\n");
    // //printf("\n biases[%d]:%f\n", l.n - 1, l.biases[l.n - 1]);

    if (0)
    {
        int i;
        for (i = 0; i < l.n; i++)
        {
            printf("bias[%d]:%d ", i, double_to_fp16(l.biases[i]));
            if (i % 6 == 5)
                printf("\n");
        }
        printf("\n");

        exit(0);
    }

    if (l.batch_normalize && (!l.dontloadscales))
    {
        //printf("batch_normalize\n");
        fread(l.scales, sizeof(float), l.n, fp);
        fread(l.rolling_mean, sizeof(float), l.n, fp);
        fread(l.rolling_variance, sizeof(float), l.n, fp);
        for (int i = 0; i < l.n; i++)
        {
            ;   //l.rolling_variance[i] = 1/(sqrt(l.rolling_variance[i]) + .000001f);  //liuj0825 changed  for normalize_cpu
                //printf("variance[%d]:%f\n",i,l.rolling_variance[i]);  // x[index] = (x[index] - mean[f])*(sqrt(variance[f]
        }
        // for (int i = 0; i < l.n; i++)
        //     if (l.rolling_variance[i] < 0)
        //         printf("VARIANCE[%d]=%f\t", i, l.rolling_variance[i]);

        /* for(int i=0;i<l.n;i++){
          l.bn_scale[i] = l.scales[i] / (sqrt(l.rolling_variance[i] )+ .000001f);
          l.bn_bias[i] = l.biases[i] - l.scales[i] * l.rolling_mean[i] /(sqrt(l.rolling_variance[i]) + .000001f);
         }*/
        if (0)
        {
            int i;
            for (i = 0; i < l.n; ++i)
            {
                printf("%g, ", l.rolling_mean[i]);
            }
            printf("\n");
            for (i = 0; i < l.n; ++i)
            {
                printf("%g, ", l.rolling_variance[i]);
            }
            printf("\n");
        }
        if (0)
        {
            fill_cpu(l.n, 0, l.rolling_mean, 1);
            fill_cpu(l.n, 0, l.rolling_variance, 1);
        }
        if (0)
        {
            int i;
            for (i = 0; i < l.n; ++i)
            {
                printf("%g, ", l.rolling_mean[i]);
            }
            printf("\n");
            for (i = 0; i < l.n; ++i)
            {
                printf("%g, ", l.rolling_variance[i]);
            }
            printf("\n");
        }
    }

#ifdef CSR
    printf("----CSR in Conv---- c/grps * n * size * size (%d/%d*%d*%d^2)=%d\n", l.c, l.groups, l.n, l.size, num);
    float *wei = calloc(num, sizeof(float));
    fread(wei, sizeof(float), num, fp);
    l.spmt = mat2csr_divide(wei, 1, num, 1024, 256);
    memcpy(l.weights, csr2mat_comb(l.spmt), sizeof(float) * num);
#else
    fread(l.weights, sizeof(float), num, fp);
#endif

    sparsity_stastic("----LOAD_CONV_WEI----", l.weights, num, 1, 1);

#ifdef QUANTIZE
    printf("BEFORE, weight[%d]:%f,weight[%d]:%f\n", num - 2, l.weights[num - 2], num - 1, l.weights[num - 1]);
    printf("loaded_conv_weights, sum = %d\n", num);
    for (int i = 0; i < 100; i++)
        printf("cov_w[%d]=%f\t", i, l.weights[i]);
    puts("");
MARK:
TL:
    1024 cutting mantissa for (int i = 0; i < num; i++)
    {
        int base = 12800;
        int mantissa = 23;
        union ui32_f32 {
            float f;
            uint32_t ui;
        } uA;
        //cut the mantissa and exponents
        uA.f = l.weights[i];
        //printf("BEFORE,WEIGHTS[%d]:%f \n", i, l.weights[i]);
        //cut 20 mantissa
        //uA.ui = (uA.ui >> (23 - mantissa)) << (23 - mantissa);
        l.weights[i] = uA.f;
        // printf("AFTER,WEIGHTS[%d]:%f \n", i, l.weights[i]);
        //Find max/min in weights per layer
        l.wei_max = l.wei_max > l.weights[i] ? l.wei_max : l.weights[i];
        l.wei_min = l.wei_min < l.weights[i] ? l.wei_min : l.weights[i];
        // printf("MAX:[%f]\tMIN:[%f]", l.wei_max, l.wei_min);
    }
    printf("MAX:[%f]\tMIN:[%f]\n", l.wei_max, l.wei_min);

    printf("AFTER,weight[%d]:%f,weight[%d]:%f\n", num - 2, l.weights[num - 2], num - 1, l.weights[num - 1]);
#endif
    //if(l.c == 3) scal_cpu(num, 1./256, l.weights, 1);
    if (l.flipped)
    {
        puts("l.flipped\n");
        transpose_matrix(l.weights, l.c * l.size * l.size, l.n);
    }
    //if (l.binary) binarize_weights(l.weights, l.n, l.c*l.size*l.size, l.weights);
#ifdef GPU
    if (gpu_index >= 0)
    {
        push_convolutional_layer(l);
    }
#endif
}

void load_weights_upto(network *net, char *filename, int start, int cutoff)
{
#ifdef GPU
    if (net->gpu_index >= 0)
    {
        cuda_set_device(net->gpu_index);
    }
#endif
    fprintf(stderr, "Loading weights from %s...\n", filename);
    fflush(stdout);
    FILE *fp = fopen(filename, "rb");
    if (!fp)
        file_error(filename);

    int major;
    int minor;
    int revision;
    fread(&major, sizeof(int), 1, fp);
    // printf("\n major:%d,sizeof(int):%d\n", major, sizeof(int));
    fread(&minor, sizeof(int), 1, fp);
    // printf("minor:%d\n", minor);
    fread(&revision, sizeof(int), 1, fp);
    // printf("revision:%d\n", revision);
    if ((major * 10 + minor) >= 2 && major < 1000 && minor < 1000)
    {
        fread(net->seen, sizeof(size_t), 1, fp);
        // printf("seen:%ld,sizeof(size_t):%d\n", net->seen, sizeof(size_t));
    }
    else
    {
        int iseen = 0;
        fread(&iseen, sizeof(int), 1, fp);
        *net->seen = iseen;
        // printf("seen:%d\n", iseen);
    }
    int transpose = (major > 1000) || (minor > 1000);

    int i;
    for (i = start; i < net->n && i < cutoff; ++i)
    {
        layer l = net->layers[i];
        if (l.dontload)
            continue;
        if (l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL)
        {
            load_convolutional_weights(l, fp);
        }
        if (l.type == CONNECTED)
        {
            load_connected_weights(l, fp, transpose);
        }
        if (l.type == BATCHNORM)
        {
            load_batchnorm_weights(l, fp);
        }
        if (l.type == CRNN)
        {
            load_convolutional_weights(*(l.input_layer), fp);
            load_convolutional_weights(*(l.self_layer), fp);
            load_convolutional_weights(*(l.output_layer), fp);
        }
        if (l.type == RNN)
        {
            load_connected_weights(*(l.input_layer), fp, transpose);
            load_connected_weights(*(l.self_layer), fp, transpose);
            load_connected_weights(*(l.output_layer), fp, transpose);
        }
        if (l.type == LSTM)
        {
            load_connected_weights(*(l.wi), fp, transpose);
            load_connected_weights(*(l.wf), fp, transpose);
            load_connected_weights(*(l.wo), fp, transpose);
            load_connected_weights(*(l.wg), fp, transpose);
            load_connected_weights(*(l.ui), fp, transpose);
            load_connected_weights(*(l.uf), fp, transpose);
            load_connected_weights(*(l.uo), fp, transpose);
            load_connected_weights(*(l.ug), fp, transpose);
        }
        if (l.type == GRU)
        {
            if (1)
            {
                load_connected_weights(*(l.wz), fp, transpose);
                load_connected_weights(*(l.wr), fp, transpose);
                load_connected_weights(*(l.wh), fp, transpose);
                load_connected_weights(*(l.uz), fp, transpose);
                load_connected_weights(*(l.ur), fp, transpose);
                load_connected_weights(*(l.uh), fp, transpose);
            }
            else
            {
                load_connected_weights(*(l.reset_layer), fp, transpose);
                load_connected_weights(*(l.update_layer), fp, transpose);
                load_connected_weights(*(l.state_layer), fp, transpose);
            }
        }
        if (l.type == LOCAL)
        {
            int locations = l.out_w * l.out_h;
            int size = l.size * l.size * l.c * l.n * locations;
            fread(l.biases, sizeof(float), l.outputs, fp);
            fread(l.weights, sizeof(float), size, fp);
#ifdef GPU
            if (gpu_index >= 0)
            {
                push_local_layer(l);
            }
#endif
        }
    }
    fprintf(stderr, "Load weiths Done!\n");
    fclose(fp);
}

void load_weights_upto16(network16 *net, char *filename, int start, int cutoff)
{
    fprintf(stderr, "Loading weights from %s...", filename);
    fflush(stdout);
    FILE *fp = fopen(filename, "r");
    if (!fp)
        file_error(filename);
    printf("open success\n");
    int major;
    int minor;
    int revision;
    fread(&major, sizeof(int), 1, fp);
    printf("major:%d,sizeof(int):%d\n", major, sizeof(int));
    fread(&minor, sizeof(int), 1, fp);
    printf("minor:%d\n", minor);
    fread(&revision, sizeof(int), 1, fp);
    printf("revision:%d\n", revision);

    if ((major * 10 + minor) >= 2 && major < 1000 && minor < 1000)
    {
        fread(net->seen, sizeof(size_t), 1, fp);
    }
    else
    {
        int iseen = 0;
        fread(&iseen, sizeof(int), 1, fp);
        *net->seen = iseen;
    }

    int transpose = (major > 1000) || (minor > 1000);

    int i;

    for (i = start; i < net->n && i < cutoff; ++i)
    {
        layer16 l = net->layers[i];
        if (l.dontload)
            continue;
        if (l.type == CONNECTED)
        {
            load_connected_weights16(l, fp, transpose); //printf("load connected success\n");
        }
        if (l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL)
        {
            load_convolutional_weights16(l, fp);
        }
        if (l.type == BATCHNORM)
        {
            load_batchnorm_weights16(l, fp);
        }
        if (l.type == RNN)
        {
            load_connected_weights16(*(l.input_layer), fp, transpose);
            load_connected_weights16(*(l.self_layer), fp, transpose);
            load_connected_weights16(*(l.output_layer), fp, transpose);
        }
        if (l.type == LSTM)
        { //printf("(l.type == LSTM)\n");
            load_connected_weights16(*(l.wi), fp, transpose);
            load_connected_weights16(*(l.wf), fp, transpose);
            load_connected_weights16(*(l.wo), fp, transpose);
            load_connected_weights16(*(l.wg), fp, transpose);
            load_connected_weights16(*(l.ui), fp, transpose);
            load_connected_weights16(*(l.uf), fp, transpose);
            load_connected_weights16(*(l.uo), fp, transpose);
            load_connected_weights16(*(l.ug), fp, transpose);
        }
        if (l.type == LSTMCJ)
        { //printf("(l.type == LSTMCJ)\n");//i f o c y
            load_connected_weights16(*(l.wi), fp, transpose);
            load_connected_weights16(*(l.wf), fp, transpose);
            load_connected_weights16(*(l.wo), fp, transpose);
            load_connected_weights16(*(l.wc), fp, transpose);
            load_connected_weights16(*(l.wy), fp, transpose);
        }
    }
    fprintf(stderr, "Done!\n");
    fclose(fp);
}

void load_weights(network *net, char *filename)
{
    load_weights_upto(net, filename, 0, net->n);
}
void load_weights16(network16 *net, char *filename)
{
    load_weights_upto16(net, filename, 0, net->n);
}