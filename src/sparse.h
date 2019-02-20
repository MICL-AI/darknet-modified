#ifndef SPARSE_H
#define SPARSE_H
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef unsigned long index_t;
typedef unsigned long offset_t;
typedef long offset_row_t; //for acc 8/9bit 190219
typedef unsigned long count_t;
typedef unsigned long pos_t;
typedef float val_t;

typedef struct position
{
    pos_t col;
    pos_t row;
} pos;

typedef struct element
{ //elements with values and positions
    val_t val;
    pos p;
} ele;

typedef struct sparse_matrix
{
    val_t *val;
    index_t *col_index;
    offset_t *offset;
    offset_row_t *offset_row;   //for acc 8/9bit 190219
    count_t num_nzd, num_zd;    //num of elements
    count_t num_col, num_row;   //shape
    count_t idx_col, idx_row;   //index for the partation
    count_t part_col, part_row; //total number of partation
    count_t ori_col, ori_row;   //original shap
    struct sparse_matrix *next; //link to next part
} spa_mat;

//functions

/* ele trans vec */
val_t *ele2vec(ele *e, count_t num_row, count_t num_col);
ele *vec2ele(val_t *vec, count_t num_row, count_t num_col);

/* compression */
spa_mat ele2csr(ele *e, count_t num_row, count_t num_col, count_t num_ele);
ele *csr2ele(spa_mat *m);

/* print */
void print_csr(spa_mat *m);
void print_ele(ele *e, count_t num_row, count_t num_col);
void print_vec(val_t *a, count_t num_row, count_t num_col);

/* combination */
val_t *csr2vec(spa_mat *m);
spa_mat vec2csr(val_t *a, count_t num_row, count_t num_col);

spa_mat mat2csr(val_t *a, count_t num_row, count_t num_col);
val_t *csr2mat(spa_mat *m);
val_t *csr2mat_new(spa_mat *m);

spa_mat mat2csr_partation(val_t *a, count_t src_row, count_t src_col, count_t dst_row, count_t dst_col, count_t index_row, count_t index_col);
// spa_mat mat2csr_divide(val_t *a, count_t src_row, count_t src_col, count_t dst_row, count_t dst_col);

#endif