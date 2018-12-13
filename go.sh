#!bin/bash
# for net in mobilenet squeezenet tiny alexnet
#     do
#         ./darknet01 TL valid $net  > '01_'$net'.log'
#     # sleep 20s
# done
for net in mobilenet squeezenet tiny alexnet vgg16 resnet50
    do
        ./darknet TL predict $net  > 'bw_req'$1'_'$net'.log'
    # sleep 20s
done