#!bin/bash

prune='0.0'
beh='valid'
for net in tiny vgg16 resnet50
    do
        ./darknet TL $beh $net  > 'ep='$prune'_'$net'.log'
    # sleep 20s
done

for net in mobilenet squeezenet alexnet 
    do
        ./darknet TL $beh $net  > 'ep='$prune'_'$net'.log'
    # sleep 20s
done