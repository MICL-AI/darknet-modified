#!bin/bash
for prune in 0.0 0.1 0.2 0.3 0.4 0.5
do
    #clean is needed as the previous .o files may infact the result, specificlly the pruning epsilong here.
    make clean
    make 'P'=$prune $1 $2 $3 -j6
done