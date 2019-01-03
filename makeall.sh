#!bin/bash
for prune in P0 P1 P2 P3 P4 P5
do
    #clean is needed as the previous .o files may infact the result, specificlly the pruning epsilong here.
    make clean
    make $prune'=1' $1 $2 $3 -j6
done