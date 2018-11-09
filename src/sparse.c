
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

#include "sparse.h"

/* for sparse mat */

#define index(a) ((a).col_index)
#define offset(a) ((a).offset)

/* for ele */
#define val(a) ((a).val)
#define col(a) ((a).p.col)
#define row(a) ((a).p.row)


/* zero valued elements occupied memory */
ele *vec2ele(val_t *vec, count_t num_row, count_t num_col)
{
    ele *e = calloc(num_row * num_col, sizeof(ele));
    count_t k = 0;
    for (count_t i = 0; i < num_row; i++)
    {
        for (count_t j = 0; j < num_col; j++)
        {
            if (vec[k])
            {
                col(e[k]) = j;
                row(e[k]) = i;
                val(e[k]) = vec[k];
            }
            //printf("%f\t", val(e[k]));
            k += 1;
        }
    }
    return e;
}

/* zero valued elements occupied memory */
val_t *ele2vec(ele *e, count_t num_row, count_t num_col)
{
    val_t *vec = calloc(num_col * num_row, sizeof(val_t));
    for (count_t i = 0; i < num_col * num_row; i++)
    {
        if (val(e[i]))
            vec[num_col * row(e[i]) + col(e[i])] = val(e[i]);
        else
            vec[i] = 0;
    }
    return vec;
}

spa_mat compress(ele *e, size_t num_row, size_t num_col)
{
    size_t ofst = 0;
    size_t last_nzd = 0;
    spa_mat spmt = {};

    /*spmt init*/
    spmt.num_nzd = 0,
    spmt.num_zd = 0;
    spmt.num_row = num_row;
    spmt.num_col = num_col;
    spmt.val = calloc(num_row * num_col, sizeof(val_t));
    spmt.col_index = calloc(num_row * num_col, sizeof(index_t));
    spmt.offset = calloc(num_row + 1, sizeof(offset_t));

    /* put into CSR */
    for (int i = 0; i < num_row * num_col; i++)
    {
        if (val(e[i]) != 0)
        {

            if (row(e[i]) > row(e[last_nzd]) || !spmt.num_nzd && !last_nzd)
            {
                offset(spmt)[ofst] = spmt.num_nzd;
                ofst += 1;
            }
            val(spmt)[spmt.num_nzd] = val(e[i]);
            index(spmt)[spmt.num_nzd] = col(e[i]);
            spmt.num_nzd += 1;
            last_nzd = i;
        }
        else
            spmt.num_zd += 1;
    }
    //finish offset val
    offset(spmt)[ofst] = spmt.num_nzd;
    // printf("nzd = %ld, zd = %ld\n", num_nz, num_z);
    return spmt;
}

ele *decompress(spa_mat *m, size_t num_row, size_t num_col)
{
    int row = 0;
    ele *e = calloc(num_row * num_col, sizeof(ele));
    for (int j = 0; j < num_row; j++)
    {
        for (int i = 0; i <= offset(*m)[row]; i++)
        {
            val(e[i]) = val(*m)[i];
            col(e[i]) = index(*m)[i];
            row(e[i]) = row;
        }
        row += 1;
        // else val(e[i]) = 0;
    }
    return e;
}

void print_compress_sparse(spa_mat *m)
{
    printf("\nvalue =");
    for (unsigned long i = 0; i < m->num_nzd; i++)
        printf(" %f", val(*m)[i]);
    printf("\n");

    printf("index =");
    for (unsigned long i = 0; i < m->num_nzd; i++)
        printf(" %lu", index(*m)[i]);
    printf("\n");

    printf("ofset =");
    for (unsigned long i = 0; i < m->num_row + 1; i++)
        printf(" %lu", offset(*m)[i]);
    printf("\n");
}

void print_elements(ele *e, size_t num_row, size_t num_col)
{
    for (int i = 0; i < num_col * num_row; i++)
    {
        if (i % num_col)
            printf("%f ", val(e[i]));
        else
            printf("\n%f ", val(e[i]));
    }
}

/*  Function name:  print_vec
    Discreption:    print vector as a matrix
    Parameters:
        @a          
        @num_row    
        @num_col    
    Return: 
*/
void print_vec(val_t *a, size_t num_row, size_t num_col)
{
    for (int i = 0; i < num_row * num_col; i++)
    {
        if (i % num_col)
            printf("%f ", a[i]);
        else
            printf("\n%f ", a[i]);
    }
}

int main(int argc, char const *argv[])
{
    float a[] = {
        1, 7, 0, 0, 0, 0,
        0, 2, 8, 0, 0, 5,
        5, 0, 3, 9, 0, 6,
        0, 6, 0, 4, 0, 0,
        0, 0, 0, 0, 1, 1,
        1, 0, 1, 0, 0, 0};
    float b[] = {
        1, 7, 0, 0,
        0, 2, 8, 0,
        5, 0, 3, 9,
        0, 6, 0, 4};
    // number of ele
    // size_t num_row = 6, num_col = 6;
    size_t num_row = 4, num_col = 4;

    // ele *ele_test = calloc(num_row * num_col, sizeof(ele));

    //ele,index -> num of ele; offset -> number of row

    // float *c = calloc(num_col * num_row, sizeof(val_t));

    puts("\noriginal sparse");
    print_vec(b, num_row, num_col);

    puts("\nafter vec2ele");
    ele *ele_test = vec2ele(b, num_row, num_col);
    print_elements(ele_test, num_row, num_col);
    //spa_mat spmat_test = compress(ele_test, num_row, num_col);
    
    puts("\nafter ele2vec");
    print_vec(ele2vec(ele_test, num_row, num_col), num_row, num_col);

    // free(ele_test);

    // puts("\nafter compress");
    // print_compress_sparse(&spmat_test);

    // puts("\nafter decompress");
    // ele *ele_decomp = decompress(&spmat_test, num_row, num_col);
    // print_elements(ele_decomp, num_row, num_col);

    // // float *c = ele2vec(ele_test, num_row, num_col);
    // puts("\nafter ele2vec");
    // print_vec(ele2vec(ele_test, num_row, num_col), num_row, num_col);

    return 0;
}
