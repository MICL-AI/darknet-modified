#ifndef GEMM_H
#define GEMM_H
#include "FP16.h"

void gemm_bin(int M, int N, int K, float ALPHA, 
        char  *A, int lda, 
        float *B, int ldb,
        float *C, int ldc);
        
void gemm16(int TA, int TB, int M, int N, int K, FLT ALPHA, 
                    FLT *A, int lda, 
                    FLT *B, int ldb,
                    FLT BETA,
                    FLT *C, int ldc);

void gemm16m2(int TA, int TB, int M, int N, int K, FLT ALPHA,
                    FLT *A, int lda,
                    FLT *B, int ldb,
                    FLT BETA,
                    FLT *C, int ldc);

void gemm16m5(int TA, int TB, int M, int N, int K, FLT ALPHA,
                    FLT *A, int lda,
                    FLT *B, int ldb,
                    FLT BETA,
                    FLT *C, int ldc);

void gemm_cpu16(int TA, int TB, int M, int N, int K, FLT ALPHA, 
        FLT *A, int lda, 
        FLT *B, int ldb,
        FLT BETA,
        FLT *C, int ldc);


void gemm_cpu16m2(int TA, int TB, int M, int N, int K, FLT ALPHA,
        FLT *A, int lda,
        FLT *B, int ldb,
        FLT BETA,
        FLT *C, int ldc);

void gemm_cpu16m5(int TA, int TB, int M, int N, int K, FLT ALPHA,
        FLT *A, int lda,
        FLT *B, int ldb,
        FLT BETA,
        FLT *C, int ldc);

void gemm16vec(int TA, int TB, int M, int N, int K, FLT ALPHA, 
                    FLT *A, int lda, 
                    FLT *B, int ldb,
                    FLT BETA,
                    FLT *C, int ldc);

void gemm_cpu16vec(int TA, int TB, int M, int N, int K, FLT ALPHA, 
        FLT *A, int lda, 
        FLT *B, int ldb,
        FLT BETA,
        FLT *C, int ldc);

void gemm(int TA, int TB, int M, int N, int K, float ALPHA, 
                    float *A, int lda, 
                    float *B, int ldb,
                    float BETA,
                    float *C, int ldc);

void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc);
#ifdef GPU
void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A_gpu, int lda, 
        float *B_gpu, int ldb,
        float BETA,
        float *C_gpu, int ldc);

void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc);
#endif
#endif
