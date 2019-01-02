#include "prune.h"
#ifdef PRUNE
long int total_load_param, total_saved_param, zero_param, conn_total, conn_zero;
float conv_reduce_max, conv_reduce_min;
int conv_layer_cnt, conv_layer_reduced;

void prune_channel(float *output, int channel, int width, int height)
{
    int zero_n = 0, zero_c = 0, zero_sum = 0;
    for (int k = 0; k < channel; k++)
    {   // per channle
        // check if all < DP_EPSILON
// #pragma omp parallel for
        for (int i = 0; i < height * width; i++)
            if (fabs(output[k * height * width + i]) <= DP_EPSILON)
                zero_n++;
        // if so, clean this channle
        if (zero_n == height * width)
        {
            zero_c++;
            memset(&output[k * height * width], 0x0, height * width * sizeof(float));
            //l.prune[channel] = 1;
        }
        zero_param += zero_n;
        zero_sum += zero_n;
        zero_n = 0;
    }
    // printf("%d * %d, channel = %d, zero_c = %d\n", width, height, channel, zero_c);
    // printf("Conv layer, total parm: %d, saved param: %d, zeros: %ld\n", width * height * channel, width * height * zero_c, zero_sum);
    // printf("In summary, total load = %ld, saved = %ld, zeros = %ld, reduced = %.2f\%, fmap sparsity:%.2f\%\n", total_load_param += width * height * channel, total_saved_param += width * height * zero_c, zero_param, ((float)total_saved_param / total_load_param) * 100, (float)zero_param / total_load_param * 100);
    conv_layer_cnt++;
    conv_layer_reduced += (zero_c > 0);
    conv_reduce_max = (float)zero_c / channel > conv_reduce_max ? (float)zero_c / channel : conv_reduce_max;
    conv_reduce_min = (float)zero_c / channel > 0 && (float)zero_c / channel < conv_reduce_min ? (float)zero_c / channel : conv_reduce_min;
    // printf("%.2f\t%.2f\n", 16 * (float)(width * height * zero_c) / 1000 / 1000, 16 * (float)(width * height * (channel - zero_c)) / 1000 / 1000);
    // printf("%d/%d reduced min = %.2f\%, max = %.2f\%\n", conv_layer_reduced, conv_layer_cnt, conv_reduce_min * 100, conv_reduce_max * 100);
    // printf("layer_sparsity:%.2f\n", (float)zero_sum / (width * height * channel)); //get sparsity
    /*END TL 181203 adding for dynamic pruning test.*/
}

#endif