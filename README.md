# darknet-modified  
A Darknet branch modified to FP16 excution.  
No weights file uploaded.  
---
Run with `` rm -f log.tl/log && make && ./darknet TL predict tiny >> log.tllog ``
in which predict could be valid/predic16/valid16 and nets could be whoever with both ``cfg`` and ``weights`` files.  

# squeezenet-source
convert from https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1

validating on ImageNet 

you should build you image database as 

https://pjreddie.com/darknet/imagenet/

then 

./darknet classifier valid cfg/imgenet1k.data squeezenet.cfg  squeezenet_darknetformat.weights
