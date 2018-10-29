
typedef FLT v_f64 __attribute__ ((vector_size(64*2)));
union f_64{
	v_f64 v;
	FLT f[64];
}; 
typedef FLT v_f32 __attribute__ ((vector_size(32*2)));
union f_32{
	v_f32 v;
	FLT f[32];
}; 
typedef FLT v_f16 __attribute__ ((vector_size(16*2)));
union f_16{
	v_f16 v;
	FLT f[16];
}; 
typedef FLT v_f8 __attribute__ ((vector_size(8*2)));
union f_8{
	v_f8 v;
	FLT f[8];
};

typedef FLT v_f4 __attribute__ ((vector_size(4*2)));
union f_4{
	v_f4 v;
	FLT f[4];
};

struct MOD_TIMES
{
 int times_[7];
 int mod_;
}mod_times;//2^7=128
struct MOD_TIMES calc_mod_and_times(int len, int flag_debug);

void mul_or_add_cpu16vec(int len ,FLT* A, FLT* B , int flag);
int  find_max_index_vec(int len ,FLT *data  );
void mul_or_add_scalarbrd_vec(int len , FLT *A ,FLT B, FLT *res ,int flag);
void copy_cpu16vec(int len ,FLT* A, FLT* B );

void set_Max_threshold(FLT *A , int len, FLT MaxValue) ;
void fill_cpu16vec(int N, FLT ALPHA, FLT *X);


