#include "prune.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define SPA_NUM 16

// entire network stastic
int fmap_total_load = 0, fmap_total_reduce = 0, fmap_total_zero[SPA_NUM] = {0}, conn_total = 0, conn_zero = 0;
float fmap_sparsity[SPA_NUM] = {0};
float SPA_TH[SPA_NUM] = {30, 20, 10, 5, 4, 3, 2, 1, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0};

float conv_reduce_max, conv_reduce_min;
int layer_conv_sum, conv_layer_reduced;

float fval;

void sparsity_stastic(const char* locate, const float *data, const int channel, const int width, const int height)
{
    puts(locate);
    
    int i, j, k, l;
    int map_zero[SPA_NUM] = {0}, empty_map[SPA_NUM] = {0}, channel_zero[SPA_NUM] = {0}, layer_zero[SPA_NUM] = {0};
    int layer_load = 0, map_load = 0, channel_load = 0;
    float layer_sparsity[SPA_NUM] = {0}, channel_sparsity[SPA_NUM] = {0};

    for (k = 0; k < channel; k++)
    { //channel level

        map_load = 0;
        for (l = 0; l < SPA_NUM; l++)
            map_zero[l] = 0;

        for (j = 0; j < height * width; j++)
        {                              //map level
            map_load = width * height; // get all element number in single map
            fval = fabs(data[k * width * height + j]);
            for (l = 0; l < SPA_NUM; l++)
                map_zero[l] += (fval <= SPA_TH[l]);
        }

        channel_load += map_load; // get all element number in single channel
        // get zero element number in single channel -- with differnet level
        for (l = 0; l < SPA_NUM; l++)
        {
            if (map_zero[l] == map_load)
                empty_map[l]++; // count empty map number
            channel_zero[l] += map_zero[l];
        }
    }

    layer_load += channel_load;    // all element number loaded in one layer.
    fmap_total_load += layer_load; // all element number loaded from the first layer.
    // calc layer sparsity -- with different level
    for (l = 0; l < SPA_NUM; l++)
    {
        layer_zero[l] += channel_zero[l];
        layer_sparsity[l] = (float)layer_zero[l] / layer_load;
        fmap_total_zero[l] += layer_zero[l];
        fmap_sparsity[l] = (float)fmap_total_zero[l] / fmap_total_load;

        printf("SPA_LEVEL %d[%.1f]\tThis layer: sparsity %d/%d = %.2f\%\t\t\t Total fmap: sparsity %d/%d = %.2f\%\n",
               l, SPA_TH[l], layer_zero[l], layer_load, layer_sparsity[l] * 100, fmap_total_zero[l], fmap_total_load, fmap_sparsity[l] * 100);
    }
}

void prune_channel(float *output, const int channel, const int width, const int height)
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
        // fmap_total_zero += zero_n;
        zero_sum += zero_n;
        zero_n = 0;
    }
    // printf("%d * %d, channel = %d, zero_c = %d\n", width, height, channel, zero_c);
    fmap_total_load += width * height * channel;
    fmap_total_reduce += width * height * zero_c;
    layer_conv_sum++;
    conv_layer_reduced += (zero_c > 0);
    conv_reduce_max = (float)zero_c / channel > conv_reduce_max ? (float)zero_c / channel : conv_reduce_max;
    conv_reduce_min = (float)zero_c / channel > 0 && (float)zero_c / channel < conv_reduce_min ? (float)zero_c / channel : conv_reduce_min;

    // printf("Conv layer, total parm: %d, saved param: %d, zeros: %d\n", width * height * channel, width * height * zero_c, zero_sum);
    // printf("In summary, total load = %d, saved = %d, zeros = %d, reduced = %.2f\%, fmap sparsity:%.2f\%\n",
    //        fmap_total_load, fmap_total_reduce, fmap_total_zero, (float)fmap_total_reduce / fmap_total_load * 100, (float)fmap_total_zero / fmap_total_load * 100);

    // printf("%.2f\t%.2f\n", 16 * (float)(width * height * zero_c) / 1000 / 1000, 16 * (float)(width * height * (channel - zero_c)) / 1000 / 1000);
    // printf("%d/%d reduced min = %.2f\%, max = %.2f\%\n", conv_layer_reduced, layer_conv_sum, conv_reduce_min * 100, conv_reduce_max * 100);
    // printf("layer_sparsity:%.2f\n", (float)zero_sum / (width * height * channel)); //get sparsity

    /*END TL 181203 adding for dynamic pruning test.*/
}
void print_channel(float *output, const int channel, const int width, const int height)
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
