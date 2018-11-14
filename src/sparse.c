/*********************************************************************************
  *Copyright(C)     hxGPT && USTB
  *FileName:        sparse.c
  *Author:          Tailin (Terrance) Liang 
  *Contact:         tailin.liang@outlook.com
  *Version:         1.0
  *Date:            2018.11.12
  *Description:     transfer sparse matrix to csr format
  *Others:
  *Function List:
  *History:
     1.Date:
       Author:
       Modification:
     2.…………
**********************************************************************************/
#include <stdio.h>
#include <stdlib.h>
/* users */
#include "sparse.h"

/* for fun */
#define vodi void
#define viod void
#define retrun return
#define mian main

/* for sparse mat */
#define index(a) ((a).col_index) //get index in csr
#define offset(a) ((a).offset)   //get offset in csr
#define smval(a) (a->val)        //get vals in csr, using like smval(a)[i]

/* for ele */
#define val(a) ((a).val)
#define col(a) ((a).p.col)
#define row(a) ((a).p.row)

/*  Function name:  vec2ele
    Discreption:    transfer none-zero data in sparse vectors into elements structs.
    Parameters:     
        @vec        input vector.
        @num_row    row number of matrix like vector.
        @num_col    row number of matrix like vector.
    Return:         A group of elements struct with non-zero data coordinates and values.
*/
ele *vec2ele(val_t *vec, count_t num_row, count_t num_col)
{
    ele *e = calloc(num_row * num_col, sizeof(ele));
    count_t k = 0;
    for (count_t i = 0; i < num_row; i++)
    {
        for (count_t j = 0; j < num_col; j++)
        {
            if (vec[i * num_col + j])
            {
                col(e[k]) = j;
                row(e[k]) = i;
                val(e[k]) = vec[i * num_col + j];
                k += 1;
            }
        }
    }
    return e;
}

/*  Function name:  ele2vec
    Discreption:    transfer none-zero data elements to sparse vectors.
    Parameters:     
        @vec        input elements.
        @num_row    row number of matrix like vector.
        @num_col    row number of matrix like vector.
    Return:         A sparse matrix stored in vector.
*/
val_t *ele2vec(ele *e, count_t num_row, count_t num_col)
{
    count_t k = 0;
    val_t *vec = calloc(num_col * num_row, sizeof(val_t));
    for (count_t i = 0; i < num_col * num_row; i++)
    {
        if (val(e[k]))
        {
            // vec[num_row * row(e[k]) + col(e[k])] = val(e[k]);
            vec[num_col * row(e[k]) + col(e[k])] = val(e[k]);
            k++;
        }
    }
    return vec;
}

/*  Function name:  ele2csr
    Discreption:    transfer none-zero elements to csr style.
    Parameters:
        @e          input elements without zero.
        @num_row    row number of matrix like vector.
        @num_col    row number of matrix like vector.
    Return:         CSR style storage.
*/
spa_mat ele2csr(ele *e, count_t num_row, count_t num_col, count_t num_ele)
{
    count_t ofst = 0;
    count_t last_nzd = 0;
    spa_mat spmt = {};

    /*spmt init*/
    spmt.num_nzd = 0,
    spmt.num_zd = 0;
    spmt.num_row = num_row;
    spmt.num_col = num_col;
    spmt.val = calloc(num_ele, sizeof(val_t));
    spmt.col_index = calloc(num_ele, sizeof(index_t));
    spmt.offset = calloc(num_row + 1, sizeof(offset_t));

    /* put into CSR */
    for (count_t i = 0; i < num_ele; i++)
    // for (count_t i = 0; i < num_row * num_col; i++)
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
    return spmt;
}

/*  Function name:  csr2ele
    Discreption:    transfer csr style to none-zero elements.
    Parameters:
        @m          struct of sparse matrix.
    Return:         elements without zeros.
*/
ele *csr2ele(spa_mat *m)
{
    count_t row = 0, k = 0;
    ele *e = calloc(m->num_row * m->num_col, sizeof(ele));
    for (count_t j = 0; j < m->num_nzd; j++)
    {
        row = j < offset(*m)[row + 1] ? row : row + 1;
        // k = j == offset(*m)[k] ? k : k + 1;
        {
            val(e[j]) = val(*m)[j];
            col(e[j]) = index(*m)[j];
            row(e[j]) = row;
        }
    }
    return e;
}
/*  Function name:  print_compress_sparse
    Discreption:    print csr
    Parameters:
        @m          struct of sparse matrix.
    Return:         null
*/
void print_csr(spa_mat *m)
{
    printf("\nvalue =");
    for (unsigned long i = 0; i < m->num_nzd; i++)
    {
        printf(" %f", val(*m)[i]);
    }
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

/*  Function name:  print_ele
    Discreption:    print elements
    Parameters:
        @a          vector
        @num_row    row number of matrix
        @num_col    col number of matrix
    Return:         null
    Note:           honestly it's useless unless when debug
    FIXME:          parameters should have a indicate of size of elements, or I can calculate with sizeof(.)
*/
void print_ele(ele *e, count_t num_row, count_t num_col)
{
    for (count_t i = 0; i < 12; i++)
    {
        printf("element %d, val = %f, pos = %d,%d\n", i, val(e[i]), row(e[i]), col(e[i]));
    }
}

/*  Function name:  print_vec
    Discreption:    print vector as a matrix
    Parameters:
        @a          vector
        @num_row    row number of matrix
        @num_col    col number of matrix
    Return:         null
*/
void print_vec(val_t *a, count_t num_row, count_t num_col)
{
    for (count_t i = 0; i < num_row * num_col; i++)
    {
        if (i % num_col)
            printf("%f ", a[i]);
        else
            printf("\n%f ", a[i]);
    }
}

val_t *csr2vec(spa_mat *m)
{
    return ele2vec(csr2ele(m), m->num_row, m->num_col);
}

spa_mat vec2csr(val_t *a, count_t num_row, count_t num_col)
{
    count_t k = 0;
    for (count_t i = 0; i < num_col * num_row; i++)
        k = a[i] ? k + 1 : k;
    ele *e = vec2ele(a, num_row, num_col);
    return ele2csr(e, num_row, num_col, k);
}

// #define DBG_SPAR

#ifdef DBG_SPAR
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
    count_t num_row = 6, num_col = 6;
    //count_t num_row = 4, num_col = 4;

    // ele *ele_test = calloc(num_row * num_col, sizeof(ele));

    //ele,index -> num of ele; offset -> number of row
    print_vec(a, 2, 6);
    puts("\n---------------------------------");
    ele *e = vec2ele(a, 2, 6);
    print_ele(e, 2, 6);
    float *c = ele2vec(e, 2, 6);
    puts("\n---------------------------------");
    print_vec(c, 2, 6);
    puts("\n---------------------------------");
    spa_mat spmat = vec2csr(a, 2, 6);
    print_csr(&spmat);

    // float *c = calloc(num_col * num_row, sizeof(val_t));

    puts("---------------");
    print_vec(csr2vec(&spmat), 2, 6);
    // puts("\noriginal sparse");
    // print_vec(b, num_row, num_col);

    // puts("\nafter vec2ele");
    // ele *ele_test = vec2ele(a, num_row, num_col);
    // print_ele(ele_test, num_row, num_col);
    // spa_mat spmat_test = ele2csr(ele_test, num_row, num_col);

    // puts("\nafter ele2vec");
    // print_vec(ele2vec(ele_test, num_row, num_col), num_row, num_col);

    // free(ele_test);

    // puts("\nafter compress");
    // print_csr(&spmat_test);

    // puts("\nafter decompress");
    // ele *ele_decomp = csr2ele(&spmat_test);
    // print_ele(ele_decomp, num_row, num_col);

    // float *c = ele2vec(ele_decomp, num_row, num_col);
    // puts("\nafter ele2vec");
    // print_vec(c, num_row, num_col);

    // float *d = csr2vec(&spmat_test);
    // print_vec(d, num_row, num_col);

    return 0;
}
#endif