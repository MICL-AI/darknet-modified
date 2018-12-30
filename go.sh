#!bin/bash

beh='valid'

for prune in 00
    do
    for net in tiny vgg16 resnet50 mobilenet squeezenet alexnet
        do
            ./darknet TL $beh $net  > 'ep='$prune'_'$net'.log'
        # sleep 20s
    done
done