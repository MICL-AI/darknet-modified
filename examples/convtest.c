#include "darknet.h"
#include <stdio.h>
#include"convolutional_layer.c"
void test_one_convolutional_layer(char *datacfg, char *cfgfile, char *weightfile, char *filename)
{   
    
    
    network *net = load_network(cfgfile, weightfile, 0);//parse_network_cfg(cfgifile);
   // fprintf(stderr, "layer   filters     size              input              output\n");
   //int batch, int h, int w, int c, int n, int groups, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam
    
    convolutional_layer l = net->layers[0];
    /*
    char buff[256];
    char *input = buff;
    if(filename){
            strncpy(input, filename, 256);
        }else{
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
    image im = load_image_color(input, 0, 0);
    image r = letterbox_image(im, net->w, net->h);
    float *X = r.data;
    */
 
    //filename = "image.dta";
    float *X = calloc(l.h * l.w * l.c,sizeof(float));
    FILE *fp = fopen(filename, "rb");    
    //fwrite(X, sizeof(float), net->w * net->h * im.c, fp); 
    fread(X, sizeof(float), l.w * l.h * l.c, fp);
    fclose(fp);

    net->input = X;
    forward_convolutional_layer(l,*net);
    exit(0);
} 
