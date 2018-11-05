# darknet-modified  
A Darknet branch modified to FP16 excution.  
No weights file uploaded.  
Run with `\./darknet TL predict tiny`  
in which predict could be valid/predic16/valid16 and nets could be whoever with both `cfg` and `weights` files.  

Original args still available.

Now the nets are:
- alexnet
- mobilenet
- resnet50
- squeezenet
- tiny
- vgg16

## My Modification

- argvs to simplify the execution

- fixed the mobilenet -O0 NaN bug with `variance[f] = (variance[f] < 0) ? 0 : variance[f];`

- jliu added a entire F16 datapath, I tested and debuged net above

- move all weights file in ./weights and all cfg file in ./cfg

- specialized path for mobilenet/squeezenet whos converted form caffe (uses caffe lables)




## REF

### squeezenet-source
convert from https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1

validating on ImageNet 

you should build you image database as 

https://pjreddie.com/darknet/imagenet/

then 

./darknet classifier valid cfg/imgenet1k.data squeezenet.cfg  squeezenet_darknetformat.weights
