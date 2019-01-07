#!bin/bash

# valid OR predict to select behaviour
beh=$1

for prune in 0.0 0.1 0.2 0.3 0.4 0.5
do
    for net in alexnet mobilenet resnet50 squeezenet tiny vgg16
        do
            ./darknetP$prune TL $beh $net  > log/$prune'_'$net'.log'
    done
done
