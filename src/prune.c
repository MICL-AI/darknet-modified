#include "prune.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef PRUNE
long int fmap_total_load, fmap_total_reduce, fmap_total_zero, conn_total, conn_zero;
float conv_reduce_max, conv_reduce_min;
int layer_conv_sum, conv_layer_reduced;

void prune_channel(float *output, int channel, int width, int height)
{
    int zero_n = 0, zero_c = 0, zero_sum = 0;
    for (int k = 0; k < channel; k++)
    { // per channel
        // check if all < DP_EPSILON
        // #pragma omp parallel for
        for (int i = 0; i < height * width; i++)
            if (fabs(output[k * height * width + i]) <= DP_EPSILON)
                zero_n++;
        // if so, clean this channel
        if (zero_n == height * width)
        {
            zero_c++;
            memset(&output[k * height * width], 0x0, height * width * sizeof(float));
            //l.prune[channel] = 1;
        }
        fmap_total_zero += zero_n;
        zero_sum += zero_n;
        zero_n = 0;
    }
    // printf("%d * %d, channel = %d, zero_c = %d\n", width, height, channel, zero_c);
    printf("Conv layer, total parm: %d, saved param: %d, zeros: %ld\n", width * height * channel, width * height * zero_c, zero_sum);
    printf("In summary, total load = %ld, saved = %ld, zeros = %ld, reduced = %.2f\%, fmap sparsity:%.2f\%\n", fmap_total_load += width * height * channel, fmap_total_reduce += width * height * zero_c, fmap_total_zero, ((float)fmap_total_reduce / fmap_total_load) * 100, (float)fmap_total_zero / fmap_total_load * 100);
    layer_conv_sum++;
    conv_layer_reduced += (zero_c > 0);
    conv_reduce_max = (float)zero_c / channel > conv_reduce_max ? (float)zero_c / channel : conv_reduce_max;
    conv_reduce_min = (float)zero_c / channel > 0 && (float)zero_c / channel < conv_reduce_min ? (float)zero_c / channel : conv_reduce_min;
    // printf("%.2f\t%.2f\n", 16 * (float)(width * height * zero_c) / 1000 / 1000, 16 * (float)(width * height * (channel - zero_c)) / 1000 / 1000);
    // printf("%d/%d reduced min = %.2f\%, max = %.2f\%\n", conv_layer_reduced, layer_conv_sum, conv_reduce_min * 100, conv_reduce_max * 100);
    // printf("layer_sparsity:%.2f\n", (float)zero_sum / (width * height * channel)); //get sparsity
    
    /*END TL 181203 adding for dynamic pruning test.*/
}
void print_channel(float *output, int channel, int width, int height)
{
    for (int k = 0; k < channel; k++)
    {
        printf("ch:%d---------\n", k);
        for (int j = 0; j < height; j++)
        {
            for (int i = 0; i < width; i++)
                printf("%d ", fabs(output[k * width * height + j * height + i]) ? 1 : 0); //fabs(floor(output[k * width * height + j * height + i])));
            printf("\n");
        }
    }
}

#endif