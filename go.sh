#!bin/bash

# valid OR predict to select behaviour
beh=$1

for prune in P0 P1 P2 P3 P4 P5
do
    for net in alexnet mobilenet resnet50 squeezenet tiny vgg16
        do
            ./darknet$prune TL $beh $net  > $prune'_'$net'.log'
    done
done
