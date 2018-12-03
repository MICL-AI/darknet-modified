#include "activations.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vec.h>

char *get_activation_string(ACTIVATION a)
{
    switch (a)
    {
    case LOGISTIC:
        return "logistic";
    case LOGGY:
        return "loggy";
    case RELU:
        return "relu";
    case ELU:
        return "elu";
    case RELIE:
        return "relie";
    case RAMP:
        return "ramp";
    case LINEAR:
        return "linear";
    case TANH:
        return "tanh";
    case PLSE:
        return "plse";
    case LEAKY:
        return "leaky";
    case STAIR:
        return "stair";
    case HARDTAN:
        return "hardtan";
    case LHTAN:
        return "lhtan";
    default:
        break;
    }
    return "relu";
}

ACTIVATION get_activation(char *s)
{
    if (strcmp(s, "logistic") == 0)
        return LOGISTIC;
    if (strcmp(s, "loggy") == 0)
        return LOGGY;
    if (strcmp(s, "relu") == 0)
        return RELU;
    if (strcmp(s, "elu") == 0)
        return ELU;
    if (strcmp(s, "relie") == 0)
        return RELIE;
    if (strcmp(s, "plse") == 0)
        return PLSE;
    if (strcmp(s, "hardtan") == 0)
        return HARDTAN;
    if (strcmp(s, "lhtan") == 0)
        return LHTAN;
    if (strcmp(s, "linear") == 0)
        return LINEAR;
    if (strcmp(s, "ramp") == 0)
        return RAMP;
    if (strcmp(s, "leaky") == 0)
        return LEAKY;
    if (strcmp(s, "tanh") == 0)
        return TANH;
    if (strcmp(s, "stair") == 0)
        return STAIR;
    fprintf(stderr, "Couldn't find activation function %s, going with ReLU\n", s);
    return RELU;
}

float activate(float x, ACTIVATION a)
{
    switch (a)
    {
    case LINEAR:
        return linear_activate(x);
    case LOGISTIC:
        return logistic_activate(x);
    case LOGGY:
        return loggy_activate(x);
    case RELU:
        return relu_activate(x);
    case ELU:
        return elu_activate(x);
    case RELIE:
        return relie_activate(x);
    case RAMP:
        return ramp_activate(x);
    case LEAKY:
        return leaky_activate(x);
    case TANH:
        return tanh_activate(x);
    case PLSE:
        return plse_activate(x);
    case STAIR:
        return stair_activate(x);
    case HARDTAN:
        return hardtan_activate(x);
    case LHTAN:
        return lhtan_activate(x);
    }
    return 0;
}
FLT activate16(FLT x, ACTIVATION a)
{
    switch (a)
    {
    case LINEAR:
        return linear_activate16(x);
    case LOGISTIC:
        return logistic_activate16(x);
    case LOGGY:
        return loggy_activate16(x);
    case RELU:
        return relu_activate16(x);
    case ELU:
        return elu_activate16(x);
    case RELIE:
        return relie_activate16(x);
    case RAMP:
        return ramp_activate16(x);
    case LEAKY:
        return leaky_activate16(x);
    case TANH:
        return tanh_activate16(x);
    case PLSE:
        return plse_activate16(x);
    case STAIR:
        return stair_activate16(x);
    case HARDTAN:
        return hardtan_activate16(x);
    case LHTAN:
        return lhtan_activate16(x);
    }
    return 0;
}
void activate_array16(FLT *x, const int n, const ACTIVATION a)
{
    int i;
    //printf("activate_array16\n");
    for (i = 0; i < n; ++i)
    {
        x[i] = activate16(x[i], a); //if(isNaN(x[i])) printf("x[%d] is Nan",i);
    }                               //printf("active type:%d,x[0]:%f,%x,x[%d]:%f,x[%d]:%f\n",a,x[0],x[0],n/2,x[n/2],n,x[n-1]);
}
void activate_array(float *x, const int n, const ACTIVATION a)
{

    int i, j;
    int batch, output;
    batch = n / (sizeof(*x) / sizeof(float)); //batch
    output = sizeof(*x) / sizeof(float);
    for (i = 0; i < output * batch; ++i)
    {
        x[i] = activate(x[i], a);
    }
    // printf("output = %d, batch = %d\n", output, batch);
    //printf("out activate_array\n");
}

float gradient(float x, ACTIVATION a)
{
    switch (a)
    {
    case LINEAR:
        return linear_gradient(x);
    case LOGISTIC:
        return logistic_gradient(x);
    case LOGGY:
        return loggy_gradient(x);
    case RELU:
        return relu_gradient(x);
    case ELU:
        return elu_gradient(x);
    case RELIE:
        return relie_gradient(x);
    case RAMP:
        return ramp_gradient(x);
    case LEAKY:
        return leaky_gradient(x);
    case TANH:
        return tanh_gradient(x);
    case PLSE:
        return plse_gradient(x);
    case STAIR:
        return stair_gradient(x);
    case HARDTAN:
        return hardtan_gradient(x);
    case LHTAN:
        return lhtan_gradient(x);
    }
    return 0;
}

void gradient_array(const float *x, const int n, const ACTIVATION a, float *delta)
{
    int i;
    for (i = 0; i < n; ++i)
    {
        delta[i] *= gradient(x[i], a);
    }
}
