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
/* users */
#include "sparse.h"

/* for fun */
#define vodi void
#define viod void
#define retrun return
#define mian main

/* for sparse mat */
#define index(a) ((a).col_index)       //get index in csr
#define offset(a) ((a).offset)         //get offset in csr
#define offset_row(a) ((a).offset_row) //self defined offset 190219
#define smval(a) (a->val)              //get vals in csr, using like smval(a)[i]

/* for ele */
#define val(a) ((a).val)
#define col(a) ((a).p.col)
#define row(a) ((a).p.row)
//----------------------------------------COMPRESSION----------------------------------------//

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
    spmt.offset_row = calloc(num_row, sizeof(offset_row_t));

    /* put into CSR */
    for (count_t i = 0; i < num_ele; i++)
    // for (count_t i = 0; i < num_row * num_col; i++)
    {
        if (val(e[i]) != 0) //actually every element != 0
        {
            if (row(e[i]) > row(e[last_nzd]) || !spmt.num_nzd && !last_nzd)
            {
                offset(spmt)[ofst] = spmt.num_nzd;
                if (ofst > 0)
                    offset_row(spmt)[ofst - 1] = offset(spmt)[ofst] - offset(spmt)[ofst - 1];
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
    offset_row(spmt)[ofst - 1] = offset(spmt)[ofst] - offset(spmt)[ofst - 1];
    spmt.num_zd = num_col * num_row - spmt.num_nzd;
    return spmt;
}

/*  Function name:  vec2csr
    Discreption:    combination of vec2ele and ele2csr
    Parameters:
    Return:         
*/
spa_mat vec2csr(val_t *a, count_t num_row, count_t num_col)
{
    count_t k = 0;
    for (count_t i = 0; i < num_col * num_row; i++)
        k = a[i] ? k + 1 : k;
    ele *e = vec2ele(a, num_row, num_col);
    return ele2csr(e, num_row, num_col, k);
}

spa_mat mat2csr(val_t *a, count_t num_row, count_t num_col)
{

    count_t num_nzd = 0;
    /*count number of none zero data*/
    for (count_t i = 0; i < num_row * num_col; i++)
    {
        if (*(a + i))
            num_nzd += 1;
    }
    /*spmt init*/
    count_t ofst = 0;
    count_t last_nzd_row = 0;
    spa_mat spmt = {};
    spmt.num_row = num_row;
    spmt.num_col = num_col;
    spmt.val = calloc(num_nzd, sizeof(val_t));
    spmt.col_index = calloc(num_nzd, sizeof(index_t));
    spmt.offset = calloc(num_row + 1, sizeof(offset_t));
    spmt.offset_row = calloc(num_row, sizeof(offset_row_t));

    /* put into CSR */
    for (count_t i = 0; i < num_row; i++) //row number of ele
    {
        for (count_t j = 0; j < num_col; j++) //col number of ele
        {
            if (*(a + num_col * i + j) != 0)
            {
                //this block should be optimized
                if (i > last_nzd_row || !spmt.num_nzd && !last_nzd_row) //next row or first row
                {
                    offset(spmt)[ofst] = spmt.num_nzd;

                    if (ofst > 0) //calc offset_row (ele num of row)
                        offset_row(spmt)[ofst - 1] = offset(spmt)[ofst] - offset(spmt)[ofst - 1];

                    for (count_t k = 1; i > (last_nzd_row + k); k++)
                    { //solving all zero row
                        offset(spmt)[ofst] = spmt.num_nzd;
                        ofst += 1;
                        offset(spmt)[ofst] = spmt.num_nzd;
                        offset_row(spmt)[ofst - 1] = 0; // nothing in this row
                    }

                    if (last_nzd_row == 0)
                    { // first row is zero
                        offset(spmt)[ofst] = 0;
                        ofst += 1;
                    }

                    ofst += 1;

                    // ofst += 1;
                }
                val(spmt)[spmt.num_nzd] = *(a + num_col * i + j);
                index(spmt)[spmt.num_nzd] = j;
                spmt.num_nzd += 1;
                last_nzd_row = i;
            }
            else
                spmt.num_zd += 1;
        }
    }
    //finish offset val
    offset(spmt)[ofst] = spmt.num_nzd;
    offset_row(spmt)[ofst - 1] = offset(spmt)[ofst] - offset(spmt)[ofst - 1];
    return spmt;
}

spa_mat mat2csr_partation(val_t *a, count_t src_row, count_t src_col, count_t dst_row, count_t dst_col, count_t index_row, count_t index_col)
{

    count_t start_row = index_row * dst_row;
    count_t start_col = index_col * dst_col;
    //solve corner cases
    size_t part_col = ceil((float)src_col / dst_col);
    size_t part_row = ceil((float)src_row / dst_row);
    if (index_row == part_row - 1 && index_col < part_col - 1) //bottom (south)
        dst_row = (src_row % dst_row) ? (src_row % dst_row) : dst_row;
    else if (index_row < part_row - 1 && index_col == part_col - 1) //right (east)
        dst_col = (src_col % dst_col) ? (src_col % dst_col) : dst_col;
    else if (index_row == part_row - 1 && index_col == part_col - 1) //corner (south east)
    {
        dst_row = (src_row % dst_row) ? (src_row % dst_row) : dst_row;
        dst_col = (src_col % dst_col) ? (src_col % dst_col) : dst_col;
    }

    // printf("------sub(%d,%d)start pos:%d,%d\tshape(%d,%d)------\n", index_row, index_col, start_row, start_col, dst_row, dst_col);
    count_t num_nzd = 0;
    /*count number of none zero data*/
    for (count_t i = start_row; i < start_row + dst_row; i++)
        for (count_t j = start_col; j < start_col + dst_col; j++)
        {
            if (*(a + i * src_col + j))
            {
                num_nzd += 1;
                // printf("ele(%d,%d)=%.2f\t", i, j, *(a + i * src_col + j));
            }
        }
    // printf("num of nzd =%d\t", num_nzd);
    // puts("counting finished");
    /*spmt init*/
    count_t ofst = 0;
    count_t last_nzd_row = 0;
    spa_mat spmt = {};

    spmt.idx_col = index_col;
    spmt.idx_row = index_row;
    spmt.ori_col = src_col;
    spmt.ori_row = src_row;
    spmt.dst_col = dst_col;
    spmt.dst_row = dst_row;
    spmt.num_row = dst_row;
    spmt.num_col = dst_col;
    spmt.part_col = part_col;
    spmt.part_row = part_row;

    spmt.val = calloc(num_nzd, sizeof(val_t));
    spmt.col_index = calloc(num_nzd, sizeof(index_t));
    spmt.offset = calloc(dst_row + 1, sizeof(offset_t));
    spmt.offset_row = calloc(dst_row, sizeof(offset_row_t));
    // puts("init finished");
    /* put into CSR */
    for (count_t i = 0; i < dst_row; i++) //row number of ele
    {
        for (count_t j = 0; j < dst_col; j++) //col number of ele
        {
            if (*(a + src_col * (i + start_row) + j + start_col) != 0)
            {
                offset_row(spmt)[i]++;
                // printf("sub info:ele(%d,%d)=%.2f,while last_row=%d,nzd=%d\n", i, j, *(a + src_col * (i + start_row) + j + start_col), last_nzd_row, spmt.num_nzd);

                //this block should be optimized, it's for the offset value
                // if (i > last_nzd_row || !spmt.num_nzd && !last_nzd_row)
                // //next row or first row
                // {
                //     // in case of first row all zero
                //     if (last_nzd_row == 0)
                //     {
                //         offset(spmt)[ofst] = 0;
                //         ofst += 1;
                //     }

                //     for (count_t k = 1; i > (last_nzd_row + k); k++)
                //     { //solving all zero row
                //         offset(spmt)[ofst] = spmt.num_nzd;
                //         ofst += 1;
                //         offset(spmt)[ofst] = spmt.num_nzd;
                //     }

                //     offset(spmt)[ofst] = spmt.num_nzd;

                //     if (ofst > 0) //calc offset_row (ele num of row)

                //     ofst += 1;

                //     // ofst += 1;
                // }
                //give value and index info to spmt
                val(spmt)[spmt.num_nzd] = *(a + src_col * (i + start_row) + j + start_col);
                index(spmt)[spmt.num_nzd] = j;
                //count nzd number
                spmt.num_nzd += 1;
                //remember last row number
                last_nzd_row = i;
            }
            else
                spmt.num_zd += 1;
        }
    }
    //finish offset val
    offset(spmt)[ofst] = spmt.num_nzd;
    offset_row(spmt)[ofst - 1] = offset(spmt)[ofst] - offset(spmt)[ofst - 1];

    return spmt;

    //finish offset val
}

spa_mat *mat2csr_divide(val_t *a, count_t src_row, count_t src_col, count_t dst_row, count_t dst_col)
{
    size_t part_col = ceil((float)src_col / dst_col);
    size_t part_row = ceil((float)src_row / dst_row);
    //so can the index of partition be (part_row, part_col)
    printf("partation number %d * %d\n", part_row, part_col);

    spa_mat *spmt = malloc(part_col * part_row * sizeof(spa_mat));
    // spa_mat spmt[9];

    count_t start_col = 0;
    count_t start_row = 0;
    for (count_t i = 0; i < part_row; i++)
        for (count_t j = 0; j < part_col; j++)
        {
            dst_col = src_col > dst_col ? dst_col : src_col;
            dst_row = src_row > dst_row ? dst_row : src_row;

            if (src_col >= dst_col && src_row >= dst_row) //make sure dst less than src
            {
                // printf("\n-------working on subset %d(%d, %d)-------\n", i * part_col + j, i, j);
                *(spmt + i * part_col + j) = mat2csr_partation(a, src_row, src_col, dst_row, dst_col, i, j);
                // spmt[i * part_col + j] = mat2csr_partation(a, src_row, src_col, dst_row, dst_col, i, j);
                // puts("---->>>");
                // print_csr(&spmt[i * part_col + j], 1);
                // printf("***num_nzd=%d", (spmt + i * part_col + j)->num_nzd);
            }
            else //TODO: add larger partation
                ;
        }
    // printf("szofspmt:%d,%d", sizeof(spa_mat), sizeof(*spmt));
    // puts("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
    // print_csr(&spmt[0], 0);
    // print_csr(&spmt[1], 1);
    // print_csr(&spmt[2], 1);
    // print_csr(&spmt[3], 1);
    //print_csr(spmt, 0);
    return spmt;
}

//----------------------------------------DECOMPRESSION----------------------------------------//
/*  Function name:  csr2ele
    Discreption:    transfer csr style to none-zero elements.
    Parameters:
        @m          struct of sparse matrix.
    Return:         elements without zeros.
*/
ele *csr2ele(spa_mat *m)
{
    count_t row = 0;
    ele *e = calloc(m->num_row * m->num_col, sizeof(ele));
    for (count_t j = 0; j < m->num_nzd; j++)
    {
        row = j < offset(*m)[row + 1] ? row : row + 1;
        {
            val(e[j]) = val(*m)[j];
            col(e[j]) = index(*m)[j];
            row(e[j]) = row;
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
            vec[num_col * row(e[k]) + col(e[k])] = val(e[k]);
            k++;
        }
    }
    return vec;
}

/*  Function name:  csr2vec
    Discreption:    combination of csr2ele and ele2vec
    Parameters:
    Return:         
*/
val_t *csr2vec(spa_mat *m)
{
    return ele2vec(csr2ele(m), m->num_row, m->num_col);
}

val_t *csr2mat(spa_mat *m)
{
    val_t *mat = calloc(m->num_col * m->num_row, sizeof(val_t));
    count_t row_idx = 0;

    for (count_t k = 0; k < m->num_nzd; k++)
    {
        for (; k >= m->offset[row_idx + 1];) //solving all zero row
            row_idx += 1;
        mat[m->num_col * row_idx + m->col_index[k]] = m->val[k];
    }
    return mat;
}

val_t *csr2mat_bynum(spa_mat *m)
{
    val_t *mat = calloc(m->num_col * m->num_row, sizeof(val_t));
    count_t row_idx = 0;
    count_t row_val_used = 0;
    for (count_t k = 0; k < m->num_nzd; k++)
    {
        for (; m->offset_row[row_idx] == 0; row_idx++)
            ; //zero row move to next

        if (row_val_used < m->offset_row[row_idx])
        { //not all used then use one
            mat[m->num_col * row_idx + m->col_index[k]] = m->val[k];
            row_val_used += 1;
        }
        else
        { // all ele used, move to next row
            row_idx += 1;
            for (; m->offset_row[row_idx] == 0; row_idx++)
                ; //zero row move to next
            mat[m->num_col * row_idx + m->col_index[k]] = m->val[k];
            row_val_used = 1;
        }
    }
    return mat;
}

val_t *csr2mat_comb(spa_mat *spmt)
{
    count_t ori_col = spmt->ori_col;
    count_t ori_row = spmt->ori_row;
    count_t part_col = spmt->part_col;
    count_t part_row = spmt->part_row;
    // count_t part_col = spmt->part_col;
    // count_t part_col = spmt->part_col;
    spa_mat spmt_tmp = {};
    count_t row_idx = 0;
    count_t row_val_used = 0;
    count_t submat_base = 0;

    //should be calloc,with zero initiation
    val_t *mat = calloc(ori_col * ori_row, sizeof(val_t));

    for (count_t i = 0; i < part_row; i++)
        for (count_t j = 0; j < part_col; j++)
        {
            spmt_tmp = *(spmt + i * part_col + j);
            //print_csr(&spmt_tmp);
            row_idx = 0;
            row_val_used = 0;
            submat_base = i * ori_col * spmt_tmp.dst_row + j * spmt_tmp.dst_col;
            // printf("base(%d,%d)=%d,shape(%d,%d)\n", i, j, submat_base, spmt_tmp.num_row, spmt_tmp.num_col);
            for (count_t k = 0; k < spmt_tmp.num_nzd; k++)
            {
                for (; spmt_tmp.offset_row[row_idx] == 0; row_idx++)
                    ; //skip zero row

                if (row_val_used < spmt_tmp.offset_row[row_idx])
                { //not all used then use one
                    mat[submat_base + ori_col * row_idx + spmt_tmp.col_index[k]] = spmt_tmp.val[k];
                    row_val_used += 1;
                }
                else
                { // all ele used, move to next row
                    row_idx += 1;
                    for (; spmt_tmp.offset_row[row_idx] == 0; row_idx++)
                        ; //zero row move to next
                    mat[submat_base + ori_col * row_idx + spmt_tmp.col_index[k]] = spmt_tmp.val[k];
                    row_val_used = 1;
                }
            }
        }
    retrun mat;
}

//----------------------------------------PRINT_FUNCTION----------------------------------------//
/*  Function name:  print_compress_sparse
    Discreption:    print csr
    Parameters:
        @m          struct of sparse matrix.
    Return:         null
*/
void print_csr(spa_mat *m, int single)
{

    if (single)
    {
        printf("total partation (%d,%d) index (%d,%d)\n", m->part_row, m->part_col, m->idx_row, m->idx_col);

        printf("num_nzd = %ld, num_zd = %ld, shape(%ld,%ld), ori shape(%ld,%ld)\n",
               m->num_nzd, m->num_zd, m->num_row, m->num_col, m->ori_row, m->ori_col);
        printf("value =");
        for (unsigned long i = 0; i < m->num_nzd; i++)
        {
            printf(" %.1f", val(*m)[i]);
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

        printf("ofstr =");
        for (unsigned long i = 0; i < m->num_row; i++)
            printf(" %lu", offset_row(*m)[i]);
        printf("\n-------------\n");
    }
    else
        for (int k = 0; k < m->part_col * m->part_row; k++)
            print_csr(m + k, 1);
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
            printf("%.1f  ", a[i]);
        else
            printf("\n%.1f  ", a[i]);
    }
}

int valid_mat(val_t *ori, val_t *decomp, count_t num_row, count_t num_col)
{
    int brk = 0;
    for (count_t k = 0; k < num_row * num_col; k++)
        if (*(ori + k) != *(decomp + k))
        {
            printf("WRONG!!!(%d,%d)=%.1f,not%.1f\n", k / num_row, k % num_col, *(ori + k), *(decomp + k));
            brk++;
        }
    if (!brk)
        puts("ALL RIGHT!!");

    //return 1;
}
//----------------------------------------TEST----------------------------------------//
#ifndef CSR
int main(int argc, char const *argv[])
{
    puts("----testing sparse matrix----");
    float mat_allz[] = {
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0};
    float mat_smal[] = {
        1, 7, 0, 0,
        0, 2, 8, 0,
        5, 0, 3, 9,
        0, 6, 0, 4};
    float mat_full[6][6] = {
        1, 7, 9, 9, 9, 9,
        9, 2, 8, 9, 9, 5,
        5, 9, 3, 9, 9, 6,
        9, 6, 9, 4, 9, 9,
        9, 9, 9, 9, 1, 1,
        1, 9, 1, 9, 6, 6};
    float mat_norm[] = {//*****
                        0, 0, 0, 0, 0, 0,
                        0, 2, 8, 0, 0, 5,
                        5, 0, 3, 9, 0, 6,
                        0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0,
                        1, 0, 1, 0, 0, 6};
    // number of ele
    count_t num_row = 6, num_col = 6;
    //count_t num_row = 4, num_col = 4;

    // ele,index -> num of ele; offset -> number of row
    print_vec(mat_norm, num_row, num_col);
    puts("\n---------------------------------");
    spa_mat *spmt;
    val_t *csr;
    // spa_mat *spmt = mat2csr_divide(mat_norm, num_row, num_col, 2, 6);
    for (int kr = 1; kr < 6; kr++)
        for (int kc = 1; kc < 6; kc++)
        {
            printf("\npartation information: kr=%d, kc=%d\n", kr, kc);
            spmt = mat2csr_divide(mat_norm, num_row, num_col, kr, kc);
            csr = csr2mat_comb(spmt);
            valid_mat(mat_norm, csr, num_row, num_col);
        }
           
    // spa_mat *spmt = mat2csr_divide(mat_norm, num_row, num_col, 3, 3);
    // spa_mat *spmt = mat2csr_divide(mat_norm, num_row, num_col, 6, 1);
    // spa_mat *spmt = mat2csr_divide(mat_norm, num_row, num_col, 2, 2);
    // print_csr(spmt, 0);
    puts("\n---------------------------------");
    // print_vec(csr, num_row, num_col);
    
    // print_csr(mat2csr_divide(mat_norm, num_row, num_col, 2, 2),0);
    // puts("\n---------------");
    // print_csr(spmt + 1);
    // puts("\n---------------");
    // print_csr(spmt + 2);
    // puts("\n---------------");
    // print_csr(spmt + 3);
    // puts("\n---------------");
    // print_csr(spmt + 4);
    // puts("\n---------------");
    // print_csr(spmt + 5);
    // puts("\n---------------");

    // spa_mat spmat = vec2csr(mat_norm, num_row, num_col);
    // print_csr(spmt, 0);
    // puts("\n---------------------------------");

    // // ele *e = vec2ele(mat_norm, num_row, num_col);
    // // print_ele(e, num_row, num_col);
    // // puts("\n---------------------------------");

    // float *c = csr2vec(&spmat);
    // float *d = csr2mat(&spmat);
    // float *e = csr2mat_new(&spmat);

    // print_vec(c, num_row, num_col);
    // puts("\n---------------------------------");

    // print_vec(e, num_row, num_col);
    // puts("\n---------------------------------");

    // float *c = calloc(num_col * num_row, sizeof(val_t));

    // puts("---------------");
    // print_vec(csr2vec(&spmat), 6, 6);
    // puts("\n---------------");
    // print_csr(spmt, 1);
    // puts("-----");
    // print_csr(spmt + 1, 1);
    // puts("-----");
    // print_csr(spmt + 2, 1);
    // puts("-----");
    // print_csr(spmt + 3, 1);
    // puts("-----");
    // // print_csr(*spmt_arry[0]);

    // spa_mat spmt_norm = mat2csr(mat_norm, num_row, num_col);
    // print_csr(&spmt_norm);
    // print_vec(csr2mat(&spmt_norm), num_row, num_col);
    // puts("\n---------------");
    // puts("csr2mat_new");
    // print_vec(csr2mat_new(&spmt_norm), num_row, num_col);
    // puts("\n---------------");
    // spa_mat spmt_norm2 = vec2csr(mat_norm, num_row, num_col);
    // print_csr(&spmt_norm2);
    // print_vec(csr2mat(&spmt_norm2), num_row, num_col);
    // puts("\n---------------");
    // spa_mat spmt_full = mat2csr(mat_full, num_row, num_col);
    // print_csr(&spmt_full);
    // spa_mat spmt_full2 = vec2csr(mat_full, num_row, num_col);
    // print_csr(&spmt_full2);
    // puts("\n---------------");
    // spa_mat spmt_allz = mat2csr(mat_allz, num_row, num_col);
    // print_csr(&spmt_allz);
    // spa_mat spmt_allz2 = vec2csr(mat_allz, num_row, num_col);
    // print_csr(&spmt_allz2);
    // puts("\n---------------");
    // num_row = 4, num_col = 4;
    // spa_mat spmt_smal = mat2csr(mat_smal, num_row, num_col);
    // print_csr(&spmt_smal);
    // spa_mat spmt_smal2 = vec2csr(mat_smal, num_row, num_col);
    // print_csr(&spmt_smal2);
    // puts("\n---------------");
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