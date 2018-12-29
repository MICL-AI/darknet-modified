#!bin/bash

beh='predict'

for prune in 00 01 02 03 04 05
    do
    for net in tiny vgg16 resnet50 mobilenet squeezenet alexnet
        do
            ./darknet$prune TL $beh $net  > 'ep='$prune'_'$net'.log'
        # sleep 20s
    done
done