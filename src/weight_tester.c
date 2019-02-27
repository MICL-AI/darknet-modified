
#include "darknet.h"

#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv)
{
    
    network *net = load_network(cfgfile, weightfile, 0);

    return 0;
}
