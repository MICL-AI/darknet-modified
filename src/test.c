#include <stdio.h>
int main()
{
    int a = 1;
    int b[a];
    b[0] = 0;
    b[1] = 1;
    printf("%d%d", b[0], b[1]);
}