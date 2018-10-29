#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H
#include "darknet.h"
#include "cuda.h"
#include "math.h"
void activate_array16(FLT *x, const int n, const ACTIVATION a);
//void activate_array16vec(FLT *x, const int n, const ACTIVATION a);


static inline FLT leaky_activate16(FLT x){return (x>0) ? x : .1*x;}
static inline FLT relu_activate16(FLT x){return x*(x>0);}
static inline FLT linear_activate16(FLT x){return x;}
static inline FLT tanh_activate16(FLT x){
	if(x>=5) x=5;
	return ((float)exp(2*x)-1)/(float)(exp(2*x)+1);
}
static inline FLT logistic_activate16(FLT x){return 1./(float)(1. + exp(-x));}
static inline FLT loggy_activate16(FLT x){return 2./(float)(1. + exp(-x)) - 1;}
static inline FLT elu_activate16(FLT x){return (x >= 0)*x + (x < 0)*(exp(x)-1);}
static inline FLT relie_activate16(FLT x){return (x>0) ? x : .01*x;}
static inline FLT ramp_activate16(FLT x){return x*(x>0)+.1*x;}
static inline FLT plse_activate16(FLT x)
{
    if(x < -4) return .01 * (x + 4);
    if(x > 4)  return .01 * (x - 4) + 1;
    return .125*x + .5;
}
static inline FLT stair_activate16(FLT x)
{
    int n = floor(x);
    if (n%2 == 0) return floor(0.5*x);
    else return (x - n) + floor(0.5*x);
}
static inline FLT hardtan_activate16(FLT x)
{
    if (x < -1) return -1;
    if (x > 1) return 1;
    return x;
}

static inline FLT lhtan_activate16(FLT x)
{
    if(x < 0) return .001*x;
    if(x > 1) return .001*(x-1) + 1;
    return x;
}


ACTIVATION get_activation(char *s); 

char *get_activation_string(ACTIVATION a);
float activate(float x, ACTIVATION a);
float gradient(float x, ACTIVATION a);
void gradient_array(const float *x, const int n, const ACTIVATION a, float *delta);
void activate_array(float *x, const int n, const ACTIVATION a);

#ifdef GPU
void activate_array_gpu(float *x, int n, ACTIVATION a);
void gradient_array_gpu(float *x, int n, ACTIVATION a, float *delta);
#endif

static inline float stair_activate(float x)
{
    int n = floor(x);
    if (n%2 == 0) return floor(x/2.);
    else return (x - n) + floor(x/2.);
}
static inline float hardtan_activate(float x)
{
    if (x < -1) return -1;
    if (x > 1) return 1;
    return x;
}
static inline float linear_activate(float x){return x;}
static inline float logistic_activate(float x){return 1./(1. + exp(-x));}
static inline float loggy_activate(float x){return 2./(1. + exp(-x)) - 1;}
static inline float relu_activate(float x){return x*(x>0);}
static inline float elu_activate(float x){return (x >= 0)*x + (x < 0)*(exp(x)-1);}
static inline float relie_activate(float x){return (x>0) ? x : .01*x;}
static inline float ramp_activate(float x){return x*(x>0)+.1*x;}
static inline float leaky_activate(float x){return (x>0) ? x : .1*x;}
static inline float tanh_activate(float x){return (exp(2*x)-1)/(exp(2*x)+1);}
static inline float plse_activate(float x)
{
    if(x < -4) return .01 * (x + 4);
    if(x > 4)  return .01 * (x - 4) + 1;
    return .125*x + .5;
}

static inline float lhtan_activate(float x)
{
    if(x < 0) return .001*x;
    if(x > 1) return .001*(x-1) + 1;
    return x;
}





static inline float lhtan_gradient(float x)
{
    if(x > 0 && x < 1) return 1;
    return .001;
}

static inline float hardtan_gradient(float x)
{
    if (x > -1 && x < 1) return 1;
    return 0;
}
static inline float linear_gradient(float x){return 1;}
static inline float logistic_gradient(float x){return (1-x)*x;}
static inline float loggy_gradient(float x)
{
    float y = (x+1.)/2.;
    return 2*(1-y)*y;
}
static inline float stair_gradient(float x)
{
    if (floor(x) == x) return 0;
    return 1;
}
static inline float relu_gradient(float x){return (x>0);}
static inline float elu_gradient(float x){return (x >= 0) + (x < 0)*(x + 1);}
static inline float relie_gradient(float x){return (x>0) ? 1 : .01;}
static inline float ramp_gradient(float x){return (x>0)+.1;}
static inline float leaky_gradient(float x){return (x>0) ? 1 : .1;}
static inline float tanh_gradient(float x){return 1-x*x;}
static inline float plse_gradient(float x){return (x < 0 || x > 1) ? .01 : .125;}

#endif

