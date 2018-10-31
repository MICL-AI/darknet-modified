#include "gemm.h"
#include "utils.h"
#include "cuda.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
//#include <vec.c>
#include "time.h"
#include "mac_opt2.h"
void gemm_bin(int M, int N, int K, float ALPHA,
              char *A, int lda,
              float *B, int ldb,
              float *C, int ldc)
{
    int i, j, k;
    for (i = 0; i < M; ++i)
    {
        for (k = 0; k < K; ++k)
        {
            char A_PART = A[i * lda + k];
            if (A_PART)
            {
                for (j = 0; j < N; ++j)
                {
                    C[i * ldc + j] += B[k * ldb + j];
                }
            }
            else
            {
                for (j = 0; j < N; ++j)
                {
                    C[i * ldc + j] -= B[k * ldb + j];
                }
            }
        }
    }
}

float *random_matrix(int rows, int cols)
{
    int i;
    float *m = calloc(rows * cols, sizeof(float));
    for (i = 0; i < rows * cols; ++i)
    {
        m[i] = (float)rand() / RAND_MAX;
    }
    return m;
}

void time_random_matrix(int TA, int TB, int m, int k, int n)
{
    float *a;
    if (!TA)
        a = random_matrix(m, k);
    else
        a = random_matrix(k, m);
    int lda = (!TA) ? k : m;
    float *b;
    if (!TB)
        b = random_matrix(k, n);
    else
        b = random_matrix(n, k);
    int ldb = (!TB) ? n : k;

    float *c = random_matrix(m, n);
    int i;
    clock_t start = clock(), end;
    for (i = 0; i < 10; ++i)
    {
        gemm_cpu(TA, TB, m, n, k, 1, a, lda, b, ldb, 1, c, n);
    }
    end = clock();
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf ms\n", m, k, k, n, TA, TB, (float)(end - start) / CLOCKS_PER_SEC);
    free(a);
    free(b);
    free(c);
}

void gemm16(int TA, int TB, int M, int N, int K, FLT ALPHA,
            FLT *A, int lda,
            FLT *B, int ldb,
            FLT BETA,
            FLT *C, int ldc)
{
    gemm_cpu16(TA, TB, M, N, K, ALPHA, A, lda, B, ldb, BETA, C, ldc);
}

void gemm16m2(int TA, int TB, int M, int N, int K, FLT ALPHA,
              FLT *A, int lda,
              FLT *B, int ldb,
              FLT BETA,
              FLT *C, int ldc)
{
    gemm_cpu16m2(TA, TB, M, N, K, ALPHA, A, lda, B, ldb, BETA, C, ldc);
}

void gemm16m5(int TA, int TB, int M, int N, int K, FLT ALPHA,
              FLT *A, int lda,
              FLT *B, int ldb,
              FLT BETA,
              FLT *C, int ldc)
{
    gemm_cpu16m5(TA, TB, M, N, K, ALPHA, A, lda, B, ldb, BETA, C, ldc);
}

/*void gemm16vec(int TA, int TB, int M, int N, int K, FLT ALPHA, 
        FLT *A, int lda, 
        FLT *B, int ldb,
     	FLT BETA,
        FLT *C, int ldc)
{//printf("gemm16vec\n");

    gemm_cpu16vec( TA,  TB,  M, N, K, ALPHA,A,lda, B, ldb,BETA,C,ldc);
}
*/

void gemm(int TA, int TB, int M, int N, int K, float ALPHA,
          float *A, int lda,
          float *B, int ldb,
          float BETA,
          float *C, int ldc)
{
    gemm_cpu(TA, TB, M, N, K, ALPHA, A, lda, B, ldb, BETA, C, ldc);
}

void gemm_nn(int M, int N, int K, float ALPHA,
             float *A, int lda,
             float *B, int ldb,
             float *C, int ldc)
{

    int i, j, k;
#pragma omp parallel for
    for (i = 0; i < M; ++i)
    {
        for (k = 0; k < K; ++k)
        {
            register float A_PART = ALPHA * A[i * lda + k];
            for (j = 0; j < N; ++j)
            {
                if (B[k * ldb + j] != B[k * ldb + j])
                    B[k + ldb + j] = 0;
                C[i * ldc + j] += A_PART * B[k * ldb + j];
                //printf("C[%d]:%f",i*ldc+j,C[i*ldc+j]);
            } //printf("C[%d]:%f",i*ldc+j-1,C[i*ldc+j-1]); printf("\n");
        }
    }
}
/*void gemm_nn16vec(int M, int N, int K, FLT ALPHA, 
        FLT *A, int lda, 
        FLT *B, int ldb,
        FLT *C, int ldc)
{
	int i , num , len_calculated=0;
	mod_times = calc_mod_and_times(lda,0); 
	
	for(num = 0; num < ldc; num++ ){
		C[num]=0;len_calculated=0;

		if(mod_times.times_[6]!=0){
			union f_64 *vecA, *vecB, vecres ,tt;
			tt.v = __builtin_gptx_fix_len_dupv64hfhf(1.0);			
			for(i=0;i<mod_times.times_[6];i++){//2
				vecA = (union f_64*)&A[ 64 *i];		
				vecB = (union f_64*)&B[lda *num +64 *i];	
				vecres.v  = vecA->v * vecB->v;  

				C[num] +=__builtin_gptx_vmsumfhfv64hfv64hf(vecres.v,tt.v);
				len_calculated +=64;// each time plus 64
			}		
		}
		if(mod_times.times_[5]!=0){
			union f_32 *vecA, *vecB ,vecres ,tt;
			tt.v = __builtin_gptx_fix_len_dupv32hfhf(1.0);
			vecA = (union f_32*)&A[len_calculated ];
			vecB = (union f_32*)&B[len_calculated + lda * num];
			vecres.v  = vecA->v * vecB->v;  

			C[num] +=__builtin_gptx_vmsumfhfv32hfv32hf(vecres.v,tt.v);
			len_calculated +=32;
		}
		if(mod_times.times_[4]!=0){
			union f_16 *vecA, *vecB,vecres ,tt;
			tt.v = __builtin_gptx_fix_len_dupv16hfhf(1.0);
			vecA = (union f_16*)&A[len_calculated ];
			vecB = (union f_16*)&B[len_calculated + lda * num];
			vecres.v  = vecA->v * vecB->v;  

			C[num] +=__builtin_gptx_vmsumfhfv16hfv16hf(vecres.v,tt.v);
			len_calculated +=16;
		}
		if(mod_times.times_[3]!=0){
			union f_8  *vecA,  *vecB,vecres ,tt;
			tt.v = __builtin_gptx_fix_len_dupv8hfhf(1.0);
			vecA = (union f_8*)&A[len_calculated];
			vecB = (union f_8*)&B[len_calculated + lda * num];
			vecres.v  = vecA->v * vecB->v;  

			C[num] +=__builtin_gptx_vmsumfhfv8hfv8hf(vecres.v,tt.v);
			len_calculated += 8;
		}
		if(mod_times.times_[2]!=0){
			union f_4  *vecA,  *vecB,vecres ,tt;
			tt.v = __builtin_gptx_fix_len_dupv4hfhf(1.0);
			vecA = (union f_4*)&A[len_calculated];
			vecB = (union f_4*)&B[len_calculated + lda * num];
			vecres.v  = vecA->v * vecB->v;  

			C[num] +=__builtin_gptx_vmsumfhfv4hfv4hf(vecres.v,tt.v);
			len_calculated += 4;
		}
		if(mod_times.times_[1]!=0){
			C[num] += A[len_calculated    ] *B[len_calculated + lda * num];
			C[num] += A[len_calculated + 1] *B[len_calculated + lda * num +1];
			len_calculated += 2;
		}
		if(mod_times.mod_==1){
			C[num] += A[len_calculated    ] *B[len_calculated + lda * num];
		}
	}

}*/
void gemm_nn16(int M, int N, int K, FLT ALPHA,
               FLT *A, int lda,
               FLT *B, int ldb,
               FLT *C, int ldc)
{
    int i, j, k;
    // printf("in gemm_nn16\n");
#pragma omp parallel for
    for (i = 0; i < M; ++i)
    {
        for (k = 0; k < K; ++k)
        {
            register FLT A_PART = ALPHA * A[i * lda + k];
            for (j = 0; j < N; ++j)
            {
                C[i * ldc + j] += A_PART * B[k * ldb + j];
            }
        }
    }
}

void gemm_nn16t(int M, int N, int K, FLT ALPHA,
                FLT *A, int lda,
                FLT *B, int ldb,
                FLT *C, int ldc)
{
    int i, j, k;
    FLT temp[20] = {0};
    FLT *CT;
    CT = temp;
#pragma omp parallel for
    for (i = 0; i < M; ++i)
    {
        for (j = 0; j < N; ++j)
        {
            for (k = 0; k < K; ++k)
            {
                register FLT A_PART = ALPHA * A[i * lda + k];
                C[i * ldc + j] += A_PART * B[k * ldb + j];
            }
        }
    }
}

void gemm_nn16m2(int M, int N, int K, FLT ALPHA,
                 FLT *A, int lda,
                 FLT *B, int ldb,
                 FLT *C, int ldc)
{
    int i, j, k;
    bitADDWID_t sum = 0;
    bit_t underflow = 0;
    bit6_t sum_exp = 0;
    union var {
        FLT data;
        fp16_t udata;
    };
    union var filter, image;
    FLT Fsum = 0;

    printf("in gemm_nn_mac2\n");

#pragma omp parallel for
    for (i = 0; i < M; ++i)
    {
        for (j = 0; j < N; ++j)
        {
            for (k = 0; k < K; ++k)
            {
                register FLT A_PART = ALPHA * A[i * lda + k];
                // C[i*ldc+j] += A_PART*B[k*ldb+j];
                filter.data = A_PART;
                image.data = B[k * ldb + j];
                //Fsum += filter.data*image.data ;
                /* filter.udata = 0x2052;image.udata=0x3a77;underflow=0;sum_exp=0x3a;sum=0x2bb9f617f010bf8;
                printf("filter:%f, %x\n",(double)filter.data,filter.udata);
                printf("image:%f, %x\n",(double)image.data,image.udata);
                printf("sum_exp:%x,sum:%lx\n",sum_exp,sum);
               */
                //sum_exp0=sum_exp; sum0=sum;
                underflow = ca_mac2(filter.udata, image.udata, sum_exp, sum, &sum_exp, &sum);
                //printf("result underflow:%x,sum_exp:%x,sum:%lx,Fsum:%f\n",underflow,sum_exp,sum,(float)Fsum);
                // fp16_t t = sum_to_fp162(underflow,sum_exp,sum);
                //printf("sum2FLT:%f,Fsum:%f\n",(float)*(FLT*)&t,(float)Fsum);
                /* if(*(FLT*)&t>65500||*(FLT*)&t<-65500   )
                {
                    printf("filter:%f, %x\n",(double)filter.data,filter.udata);
                    printf("image:%f, %x\n",(double)image.data,image.udata);
                    printf("sum_exp:%x,sum:%lx,sum_exp0:%x,sum0:%lx\n",sum_exp,sum,sum_exp0,sum0);
                    printf("result underflow:%x,sum_exp:%x,sum:%lx,Fsum:%f\n",underflow,sum_exp,sum,(float)Fsum);
                    printf("sum2FLT:%f\n",(float)*(FLT*)&t);
                    exit(0);
                }
               */
                // exit(0);
            }

            fp16_t e = sum_to_fp162(underflow, sum_exp, sum);
            C[i * ldc + j] = *(FLT *)&e;
            //printf("Fsum:%f,C[%d*%d+%d]:%f\n",Fsum,i,ldc,j,(float)C[i*ldc+j]);
            Fsum = 0;
            sum = 0;
            underflow = 0;
            sum_exp = 0;
        }
    }
}

void gemm_nn16m5(int M, int N, int K, FLT ALPHA,
                 FLT *A, int lda,
                 FLT *B, int ldb,
                 FLT *C, int ldc)
{
    int i, j, k;
    bit89_t sum = 0;

    printf("in gemm_nn_mac5\n");

#pragma omp parallel for
    for (i = 0; i < M; ++i)
    {
        for (j = 0; j < N; ++j)
        {
            for (k = 0; k < K; ++k)
            {
                FLT A_PART = ALPHA * A[i * lda + k];
                ca_mac5(*(fp16_t *)&A_PART, *(fp16_t *)&B[k * ldb + j], sum, &sum);
            }
            fp16_t e = sum_to_fp165(sum);
            C[i * ldc + j] = *(FLT *)&e;
            sum = 0;
        }
    }
}

void saveresgemm(char *filename, FLT *data, int len)
{
    FILE *fp = fopen(filename, "w+");
    int num;
    for (num = 0; num < len; num++)
    {
        fprintf(fp, "%f ", (float)data[num]);
    }
    fclose(fp);
}

void gemm_nt(int M, int N, int K, float ALPHA,
             float *A, int lda,
             float *B, int ldb,
             float *C, int ldc)
{

    int i, j, k;
    printf("\nMARK:using gemm_nt\nB[32]=%f\tB[33]=%f\tB[34]=%f", B[32], B[33], B[34]);
    
#pragma omp parallel for
    for (i = 0; i < M; ++i)
    {
        for (j = 0; j < N; ++j)
        {
            register float sum = 0;
            for (k = 0; k < K; ++k)
            {
                sum += ALPHA * A[i * lda + k] * B[j * ldb + k];
            }
            C[i * ldc + j] += sum;
            //printf("\nsum:%f",sum);
        }
    }
}
clock_t aa, bb;
void gemm_nt16(int M, int N, int K, FLT ALPHA,
               FLT *A, int lda,
               FLT *B, int ldb,
               FLT *C, int ldc)
{
    int i, j, k;
        printf("\nMARK:using gemm_nt16\nB[32]=%f\tB[33]=%f\tB[34]=%f", B[32], B[33], B[34]);
    for (i = 0; i < M; ++i)
    {
        for (j = 0; j < N; ++j)
        {
            FLT sum = 0;
            for (k = 0; k < K; ++k)
            {
                sum += ALPHA * A[i * lda + k] * B[j * ldb + k];
            }
            C[i * ldc + j] += sum;
            //printf("\nsum:%f",sum);
        }
    }
}

void gemm_nt16m5(int M, int N, int K, FLT ALPHA,
                 FLT *A, int lda,
                 FLT *B, int ldb,
                 FLT *C, int ldc)
{
    int i, j, k;
    //FLT sum = 0;
    bit89_t sum = 0;
    printf("gemm_nt16m5\n");

    for (i = 0; i < M; ++i)
    {
        for (j = 0; j < N; ++j)
        {
            sum = 0;
            for (k = 0; k < K; ++k)
            {
                FLT A_PART = ALPHA * A[i * lda + k];
                //sum += ALPHA*A[i*lda+k]*B[j*ldb + k];
                ca_mac5(*(fp16_t *)&A_PART, *(fp16_t *)&B[j * ldb + k], sum, &sum);
            }
            fp16_t e = sum_to_fp165(sum);
            C[i * ldc + j] += *(FLT *)&e;
            //C[i*ldc+j] += sum;
        }
    }
}

/*void gemm_nt16vec(int M, int N, int K, FLT ALPHA, 
        FLT *A, int lda, 
        FLT *B, int ldb,
        FLT *C, int ldc)
{   
	int i , num , len_calculated=0;
	mod_times = calc_mod_and_times(lda,0); 

	for(num = 0; num < ldc; num++ ){
		C[num]=0;len_calculated=0;

		if(mod_times.times_[6]!=0){
			union f_64 *vecA, *vecB, vecres ,tt;
			tt.v = __builtin_gptx_fix_len_dupv64hfhf(1.0);			
			for(i=0;i<mod_times.times_[6];i++){
				vecA = (union f_64*)&A[ 64 *i];		
				vecB = (union f_64*)&B[lda *num +64 *i];	
				vecres.v  = vecA->v * vecB->v;  

				C[num] +=__builtin_gptx_vmsumfhfv64hfv64hf(vecres.v,tt.v);
				len_calculated +=64;// each time plus 64
			}		
		}
		if(mod_times.times_[5]!=0){
			union f_32 *vecA, *vecB ,vecres ,tt;
			tt.v = __builtin_gptx_fix_len_dupv32hfhf(1.0);
			vecA = (union f_32*)&A[len_calculated ];
			vecB = (union f_32*)&B[len_calculated + lda * num];
			vecres.v  = vecA->v * vecB->v;  

			C[num] +=__builtin_gptx_vmsumfhfv32hfv32hf(vecres.v,tt.v);
			len_calculated +=32;
		}
		if(mod_times.times_[4]!=0){
			union f_16 *vecA, *vecB,vecres ,tt;
			tt.v = __builtin_gptx_fix_len_dupv16hfhf(1.0);
			vecA = (union f_16*)&A[len_calculated ];
			vecB = (union f_16*)&B[len_calculated + lda * num];
			vecres.v  = vecA->v * vecB->v;  

			C[num] +=__builtin_gptx_vmsumfhfv16hfv16hf(vecres.v,tt.v);
			len_calculated +=16;
		}
		if(mod_times.times_[3]!=0){
			union f_8  *vecA,  *vecB,vecres ,tt;
			tt.v = __builtin_gptx_fix_len_dupv8hfhf(1.0);
			vecA = (union f_8*)&A[len_calculated];
			vecB = (union f_8*)&B[len_calculated + lda * num];
			vecres.v  = vecA->v * vecB->v;  

			C[num] +=__builtin_gptx_vmsumfhfv8hfv8hf(vecres.v,tt.v);
			len_calculated += 8;
		}
		if(mod_times.times_[2]!=0){
			union f_4  *vecA,  *vecB,vecres ,tt;
			tt.v = __builtin_gptx_fix_len_dupv4hfhf(1.0);
			vecA = (union f_4*)&A[len_calculated];
			vecB = (union f_4*)&B[len_calculated + lda * num];
			vecres.v  = vecA->v * vecB->v;  

			C[num] +=__builtin_gptx_vmsumfhfv4hfv4hf(vecres.v,tt.v);
			len_calculated += 4;
		}
		if(mod_times.times_[1]!=0){
			C[num] += A[len_calculated    ] *B[len_calculated + lda * num];
			C[num] += A[len_calculated + 1] *B[len_calculated + lda * num +1];
			len_calculated += 2;
		}
		if(mod_times.mod_==1){
			C[num] += A[len_calculated    ] *B[len_calculated + lda * num];
		}
	}

}
*/

void gemm_tn(int M, int N, int K, float ALPHA,
             float *A, int lda,
             float *B, int ldb,
             float *C, int ldc)
{
    int i, j, k;
#pragma omp parallel for
    for (i = 0; i < M; ++i)
    {
        for (k = 0; k < K; ++k)
        {
            register float A_PART = ALPHA * A[k * lda + i];
            for (j = 0; j < N; ++j)
            {
                C[i * ldc + j] += A_PART * B[k * ldb + j];
            }
        }
    }
}
void gemm_tn16(int M, int N, int K, FLT ALPHA,
               FLT *A, int lda,
               FLT *B, int ldb,
               FLT *C, int ldc)
{
    int i, j, k;
#pragma omp parallel for
    for (i = 0; i < M; ++i)
    {
        for (k = 0; k < K; ++k)
        {
            register FLT A_PART = ALPHA * A[k * lda + i];
            for (j = 0; j < N; ++j)
            {
                C[i * ldc + j] += A_PART * B[k * ldb + j];
            }
        }
    }
}

void gemm_tt(int M, int N, int K, float ALPHA,
             float *A, int lda,
             float *B, int ldb,
             float *C, int ldc)
{
    int i, j, k;
#pragma omp parallel for
    for (i = 0; i < M; ++i)
    {
        for (j = 0; j < N; ++j)
        {
            register float sum = 0;
            for (k = 0; k < K; ++k)
            {
                sum += ALPHA * A[i + k * lda] * B[k + j * ldb];
            }
            C[i * ldc + j] += sum;
        }
    }
}
void gemm_tt16(int M, int N, int K, FLT ALPHA,
               FLT *A, int lda,
               FLT *B, int ldb,
               FLT *C, int ldc)
{
    int i, j, k;
#pragma omp parallel for
    for (i = 0; i < M; ++i)
    {
        for (j = 0; j < N; ++j)
        {
            register FLT sum = 0;
            for (k = 0; k < K; ++k)
            {
                sum += ALPHA * A[i + k * lda] * B[k + j * ldb];
            }
            C[i * ldc + j] += sum;
        }
    }
}
void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA,
              float *A, int lda,
              float *B, int ldb,
              float BETA,
              float *C, int ldc)
{
    //printf("cpu: %d %d %d %d %d %f %d %d %f %d\n",TA, TB, M, N, K, ALPHA, lda, ldb, BETA, ldc);
    int i, j;
    for (i = 0; i < M; ++i)
    {
        for (j = 0; j < N; ++j)
        {
            C[i * ldc + j] *= BETA;
        }
    }
    if (!TA && !TB)
        gemm_nn(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
    else if (TA && !TB)
        gemm_tn(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
    else if (!TA && TB)
        gemm_nt(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
    else
        gemm_tt(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
}

void gemm_cpu16(int TA, int TB, int M, int N, int K, FLT ALPHA,
                FLT *A, int lda,
                FLT *B, int ldb,
                FLT BETA,
                FLT *C, int ldc)
{
    int i, j;
    for (i = 0; i < M; ++i)
    {
        for (j = 0; j < N; ++j)
        {
            C[i * ldc + j] *= BETA;
        }
    }
    if (!TA && !TB)
    {
        gemm_nn16(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
    }
    else if (TA && !TB)
    {
        gemm_tn16(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
    }
    else if (!TA && TB)
    {
        gemm_nt16(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
    }
    else
    {
        gemm_tt16(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
    }
}

void gemm_cpu16m2(int TA, int TB, int M, int N, int K, FLT ALPHA,
                  FLT *A, int lda,
                  FLT *B, int ldb,
                  FLT BETA,
                  FLT *C, int ldc)
{
    int i, j;
    for (i = 0; i < M; ++i)
    {
        for (j = 0; j < N; ++j)
        {
            C[i * ldc + j] *= BETA;
        }
    }
    if (!TA && !TB)
    {
        gemm_nn16m2(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
    }
    else if (TA && !TB)
    {
        gemm_tn16(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
    }
    else if (!TA && TB)
    {
        gemm_nt16(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
    }
    else
    {
        gemm_tt16(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
    }
}

void gemm_cpu16m5(int TA, int TB, int M, int N, int K, FLT ALPHA,
                  FLT *A, int lda,
                  FLT *B, int ldb,
                  FLT BETA,
                  FLT *C, int ldc)
{
    int i, j;
    for (i = 0; i < M; ++i)
    {
        for (j = 0; j < N; ++j)
        {
            C[i * ldc + j] *= BETA;
        }
    }
    if (!TA && !TB)
    {
        gemm_nn16m5(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
    }
    else if (TA && !TB)
    {
        gemm_tn16(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
    }
    else if (!TA && TB)
    {
        gemm_nt16m5(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
    }
    else
    {
        gemm_tt16(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
    }
}

/*void gemm_cpu16vec(int TA, int TB, int M, int N, int K, FLT ALPHA, 
        FLT *A, int lda, 
        FLT *B, int ldb,
        FLT BETA,
        FLT *C, int ldc)
{

    if(!TA && !TB){ 
        gemm_nn16vec(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);}
    else if(TA && !TB){
        gemm_tn16(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);}
    else if(!TA && TB){
        gemm_nt16vec(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);}
    else{
        gemm_tt16(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);}
}
*/

#ifdef GPU

#include <math.h>

void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA,
              float *A_gpu, int lda,
              float *B_gpu, int ldb,
              float BETA,
              float *C_gpu, int ldc)
{
    cublasHandle_t handle = blas_handle();
    cudaError_t status = cublasSgemm(handle, (TB ? CUBLAS_OP_T : CUBLAS_OP_N),
                                     (TA ? CUBLAS_OP_T : CUBLAS_OP_N), N, M, K, &ALPHA, B_gpu, ldb, A_gpu, lda, &BETA, C_gpu, ldc);
    check_error(status);
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void time_gpu_random_matrix(int TA, int TB, int m, int k, int n)
{
    float *a;
    if (!TA)
        a = random_matrix(m, k);
    else
        a = random_matrix(k, m);
    int lda = (!TA) ? k : m;
    float *b;
    if (!TB)
        b = random_matrix(k, n);
    else
        b = random_matrix(n, k);
    int ldb = (!TB) ? n : k;

    float *c = random_matrix(m, n);
    int i;
    clock_t start = clock(), end;
    for (i = 0; i < 32; ++i)
    {
        gemm_gpu(TA, TB, m, n, k, 1, a, lda, b, ldb, 1, c, n);
    }
    end = clock();
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf s\n", m, k, k, n, TA, TB, (float)(end - start) / CLOCKS_PER_SEC);
    free(a);
    free(b);
    free(c);
}

void time_gpu(int TA, int TB, int m, int k, int n)
{
    int iter = 10;
    float *a = random_matrix(m, k);
    float *b = random_matrix(k, n);

    int lda = (!TA) ? k : m;
    int ldb = (!TB) ? n : k;

    float *c = random_matrix(m, n);

    float *a_cl = cuda_make_array(a, m * k);
    float *b_cl = cuda_make_array(b, k * n);
    float *c_cl = cuda_make_array(c, m * n);

    int i;
    clock_t start = clock(), end;
    for (i = 0; i < iter; ++i)
    {
        gemm_gpu(TA, TB, m, n, k, 1, a_cl, lda, b_cl, ldb, 1, c_cl, n);
        cudaThreadSynchronize();
    }
    double flop = ((double)m) * n * (2. * k + 2.) * iter;
    double gflop = flop / pow(10., 9);
    end = clock();
    double seconds = sec(end - start);
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf s, %lf GFLOPS\n", m, k, k, n, TA, TB, seconds, gflop / seconds);
    cuda_free(a_cl);
    cuda_free(b_cl);
    cuda_free(c_cl);
    free(a);
    free(b);
    free(c);
}

void test_gpu_accuracy(int TA, int TB, int m, int k, int n)
{
    srand(0);
    float *a;
    if (!TA)
        a = random_matrix(m, k);
    else
        a = random_matrix(k, m);
    int lda = (!TA) ? k : m;
    float *b;
    if (!TB)
        b = random_matrix(k, n);
    else
        b = random_matrix(n, k);
    int ldb = (!TB) ? n : k;

    float *c = random_matrix(m, n);
    float *c_gpu = random_matrix(m, n);
    memset(c, 0, m * n * sizeof(float));
    memset(c_gpu, 0, m * n * sizeof(float));
    int i;
    //pm(m,k,b);
    gemm_gpu(TA, TB, m, n, k, 1, a, lda, b, ldb, 1, c_gpu, n);
    //printf("GPU\n");
    //pm(m, n, c_gpu);

    gemm_cpu(TA, TB, m, n, k, 1, a, lda, b, ldb, 1, c, n);
    //printf("\n\nCPU\n");
    //pm(m, n, c);
    double sse = 0;
    for (i = 0; i < m * n; ++i)
    {
        //printf("%f %f\n", c[i], c_gpu[i]);
        sse += pow(c[i] - c_gpu[i], 2);
    }
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %g SSE\n", m, k, k, n, TA, TB, sse / (m * n));
    free(a);
    free(b);
    free(c);
    free(c_gpu);
}

int test_gpu_blas()
{
    /*
       test_gpu_accuracy(0,0,10,576,75); 

       test_gpu_accuracy(0,0,17,10,10); 
       test_gpu_accuracy(1,0,17,10,10); 
       test_gpu_accuracy(0,1,17,10,10); 
       test_gpu_accuracy(1,1,17,10,10); 

       test_gpu_accuracy(0,0,1000,10,100); 
       test_gpu_accuracy(1,0,1000,10,100); 
       test_gpu_accuracy(0,1,1000,10,100); 
       test_gpu_accuracy(1,1,1000,10,100); 

       test_gpu_accuracy(0,0,10,10,10); 

       time_gpu(0,0,64,2916,363); 
       time_gpu(0,0,64,2916,363); 
       time_gpu(0,0,64,2916,363); 
       time_gpu(0,0,192,729,1600); 
       time_gpu(0,0,384,196,1728); 
       time_gpu(0,0,256,196,3456); 
       time_gpu(0,0,256,196,2304); 
       time_gpu(0,0,128,4096,12544); 
       time_gpu(0,0,128,4096,4096); 
     */
    time_gpu(0, 0, 64, 75, 12544);
    time_gpu(0, 0, 64, 75, 12544);
    time_gpu(0, 0, 64, 75, 12544);
    time_gpu(0, 0, 64, 576, 12544);
    time_gpu(0, 0, 256, 2304, 784);
    time_gpu(1, 1, 2304, 256, 784);
    time_gpu(0, 0, 512, 4608, 196);
    time_gpu(1, 1, 4608, 512, 196);

    return 0;
}
#endif
