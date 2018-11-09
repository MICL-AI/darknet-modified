#ifndef SPARSE_H
#define SPARSE_H

typedef unsigned long index_t;
typedef unsigned long offset_t;
typedef unsigned long count_t;
typedef unsigned long pos_t;
typedef float val_t;

typedef struct position
{
    pos_t col;
    pos_t row;
} pos;

typedef struct element
{
    val_t val;
    pos p;
} ele;

typedef struct sparse_matrix
{
    val_t *val;
    index_t *col_index;
    offset_t *offset;
    count_t num_nzd, num_zd;        //num of elements
    count_t num_col, num_row;       //shape
} spa_mat;


//functions
/*ele trans vec*/
val_t* ele2vec(ele *e, count_t num_row, count_t num_col);
ele* vec2ele(val_t *vec, count_t num_row, count_t num_col);

/*compression*/
spa_mat compress(ele *e, count_t num_row, count_t num_col);
ele* decompress(spa_mat *m, size_t num_row, size_t num_col);

/*print*/
void print_compress_sparse(spa_mat *m);
void print_elements(ele *e, size_t num_row, size_t num_col);
void print_vec(val_t *a, size_t num_row, size_t num_col);

#endif