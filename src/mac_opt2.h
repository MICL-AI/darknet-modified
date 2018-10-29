#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>
#include <float.h>

typedef uint16_t    fp16_t;
typedef __uint128_t bit89_t;
typedef __uint128_t bit81_t;
typedef __uint128_t bit80_t;
typedef __uint128_t bit65_t;
typedef uint64_t    bit64_t;
typedef uint64_t    bit35_t;
typedef uint64_t    bit36_t;
typedef uint8_t     bit_t;
typedef uint8_t     bit5_t;
typedef uint8_t     bit6_t;
typedef uint16_t    bit16_t;
typedef uint16_t    bit10_t;
typedef uint16_t    bit11_t;
typedef uint32_t    bit22_t;

typedef uint64_t    bitADDWID_t;

//the range for guard is from 10-50
//the related adder width is 24-64
#define GUARD       22
//Accuracy + FP16 Mant + 2 Carry out + 1 Sign
#define ADDWID      11+1+1+1+GUARD

#define MASK(x) ((((__uint128_t)1)<<(x))-1)

double
fp16_to_double(
    fp16_t  a
    )
{
  unsigned  a_sign  = (a>>15)&0x1;
  unsigned  a_exp   = (a>>10)&0x1f;
  unsigned  a_mant  = a&0x3ff;
  uint64_t  res;
  
  res = ((uint64_t)a_sign) << 63;
  if( a_exp == 0 )
  {
    if( a_mant != 0 ) // convert denormal to normal
    {
      int exp = -15;
      unsigned  mant = a_mant<<1;
      while( (mant&0x400) == 0 )
      {
        exp--;
        mant <<= 1;
      }
      res |= (uint64_t)(exp+1023)<<52;
      res |= (uint64_t)(mant&0x3ff)<<(52-10);
    }
  }
  else if( a_exp == 0x1f )
  {
    res |= 0x7ffULL<<52;
    res |= (uint64_t)a_mant<<(52-10);
  }
  else
  {
    res |= (uint64_t)(a_exp+1023-15)<<52;
   res |= (uint64_t)a_mant<<(52-10);
  }

  union {
    uint64_t  u;
    double    d;
  } cvt;
  cvt.u = res;

  return cvt.d;
}

fp16_t
double_to_fp16(
    double  d
    )
{
  union {
    uint64_t  u;
    double    d;
  } cvt;
  cvt.d = d;
  uint64_t  u = cvt.u;

  unsigned   u_sign = (u>>63)&0x1;
  unsigned   u_exp = (u>>52)&0x7ff;
  uint64_t   u_mant = u&0xfffffffffffffULL;

  fp16_t    res = u_sign<<15;
  if( u_exp == 0 )
    res |= 0<<10;
  else if( u_exp == 0x7ff )
    res |= 0x1f<<10;
  else
  {
    int exp = u_exp - 1023+15;
    if( exp >= 0x1f || exp < -10 )
    {
      res |= 0x1f<<10;
      res |= u_mant>>42;
    }
    else if( exp <= 0 )
    {
      res |= 0<<10;
      unsigned  mant = (u_mant>>42)|(1<<10);
      res |= mant>>(1-exp);
    }
    else
    {
      res |= exp<<10;
      res |= u_mant>>42;
    }
  }

  return res;
}

unsigned
clz_32(
    uint64_t    u,
    unsigned    bits
    )
{
  assert( u != 0 );

  unsigned    n = 0;

  while( u != 0 )
  {
    unsigned sh = u >> (bits-1);
    if( sh != 0 )
      break;
    u <<= 1;
    n++;
  }
  return n;
}

unsigned
clz_ADDWID_3 (
    uint64_t    u,
    unsigned    bits
    )
{
  assert( u != 0 );

  unsigned    n = 0;

  while( u != 0 )
  {
    unsigned sh = u >> (bits-1);
    if( sh != 0 )
      break;
    u <<= 1;
    n++;
  }
  return n;
}

unsigned
clo_ADDWID_3 (
    uint64_t    u1,
    unsigned    bits
    )
{
  uint64_t u = -u1; 
  assert( u != 0 );

  unsigned    n = 0;

  while( u != 0 )
  {
    unsigned sh = u >> (bits-1);
    if( sh != 0 )
      break;
    u <<= 1;
    n++;
  }
  return n;
}

unsigned
clz_33(
    uint64_t    u,
    unsigned    bits
    )
{
  assert( u != 0 );

  unsigned    n = 0;

  while( u != 0 )
  {
    unsigned sh = u >> (bits-1);
    if( sh != 0 )
      break;
    u <<= 1;
    n++;
  }
  return n;
}

unsigned
clz_64(
    uint64_t    u,
    unsigned    bits
    )
{
  assert( u != 0 );

  unsigned    n = 0;

  while( u != 0 )
  {
    unsigned sh = u >> (bits-1);
    if( sh != 0 )
      break;
    u <<= 1;
    n++;
  }
  return n;
}

bit_t
ca_mac2(
    fp16_t         a,
    fp16_t         b,
    bit6_t         sum_exp,
    bitADDWID_t    sum,
    bit6_t      *  p_sum_exp,
    bitADDWID_t *  p_sum
    )
{
  bit_t   debug_en;
 
  bit_t   a_sign  = (a>>15)&0x1;
  bit5_t  a_exp   = (a>>10)&0x1f;
  bit10_t a_mant  = a&0x3ff;

  bit_t   b_sign  = (b>>15)&0x1;
  bit5_t  b_exp   = (b>>10)&0x1f;
  bit10_t b_mant  = b&0x3ff;

  bit6_t  mul_exp_o;
 
  debug_en = 0x0;
  // the basic rule for the mac is to adjust the exp as possible.
  // If overflow happens in the process, the extended exp will add up.
  // If underflow happens, keep current denormal mant as it is now
  // the final part will detect denormal and overflow (sum_to_fp162)
  
  // add the exp for mul
  if( a_exp == 0 || b_exp == 0 )
    mul_exp_o = 0;
  else if( a_exp == 0x1f || b_exp == 0x1f )
    mul_exp_o = 0x1f;
  else
    mul_exp_o = a_exp + (b_exp - 15);

  // the D-value for the exps
  bit_t  mul_exph;
  bit6_t exp_diff;
  bit6_t exp_larger;
   
  // Bugfix!!
  // the product is denormal, so a_exp + b_exp < 15, the mul_exp
  // may act as big uint, and cause overflow assertion.
  //
  // Corner case
  // denormal sum may be added to denormal product
  // denormal handling:
  // if current result generates denormal, shift the exp to zero!
  // or the result will cause a fake right shift for next round product
  // and unnecessary left shift clz
  //
  mul_exph = (mul_exp_o >= sum_exp) & ((a_exp + b_exp) >=15);

  if (mul_exph == 1)
  {
     exp_diff   =(mul_exp_o - sum_exp) & 0x3f;
     exp_larger = mul_exp_o;
  }
  else
  {
     exp_diff   =(sum_exp - mul_exp_o) & 0x3f;
     exp_larger = sum_exp;
  }

     //printf ("the mul_exp is!!!!!! %x \n",mul_exp_o);
     //printf ("the sum_exp is!!!!!! %x \n",sum_exp);

  // mul 
  bit22_t prod;
  if( a_exp == 0 || b_exp == 0 )
    prod = 0;
  else if( a_exp == 0x1f || b_exp == 0x1f )
    prod = 0x3fffff;
  else
    prod = ((1<<10)|a_mant)*((1<<10)|b_mant);

  // printf product
  if(debug_en == 1)
  {
     printf ("the mul Op1 is %x  \n", ((1<<10)|a_mant));
     printf ("the mul Op2 is %x  \n", ((1<<10)|b_mant));
     printf ("the product is %x  \n", prod);

     if (mul_exp_o >= 31)
     printf ("Attention ! probably overflow or hit corner case!");
     else  
     printf ("the mul_exp is %x \n",(mul_exp_o-15));

     printf ("mul larger     %x \n", mul_exph);
  }

  // extend the product
  int prod_o; // 23 bit signed integer
  int64_t sprod;

  if( a_sign == b_sign )
     sprod = prod;
  else
  {
     prod_o=-prod;
     sprod = prod_o;
  }

  // temprary value define
  bit6_t  result_exp;
  bit_t   result_sign;

  bitADDWID_t add_src1;
  bitADDWID_t add_src2;
  int64_t     res_mant_o;
  bitADDWID_t result_mant;

  unsigned shift;
  // smaller exp shift to larger exp, the mant will need to right shift  
  // mul has the lager exp
  
  // sum align to product
  //            cout1                   guard bits
  // sign cout2 product[21:10] product[09:00].. 
  // sign  ...      sum[11:00]     -->      sum[11:00]
  // 
  // the corner case for calculate is shift twice
  //
  // cout1==1 product(MSB==1)
  //          sum    (MSB==1) 

  // product align to sum
  //            cout1                   guard bits
  // sign cout2     sum[11:10] 
  // sign  ...  product[21:10] product[09:00]  -->   product[09:00]
  
  // operand preparation && calculation
  int64_t sum_o = sum;

  if ( mul_exph == 1 )
  {
     add_src1    = sprod << ADDWID-24; // signed integer
     add_src2    = sum_o >> exp_diff;    
  }
  else
  {
     if(exp_diff <= GUARD-10)
     add_src1    = sprod << (GUARD-10-exp_diff); // signed integer
     else
     add_src1    = sprod >> (exp_diff-(GUARD-10)); // signed integer

     add_src2    = sum;
  }

  res_mant_o  = add_src1 + add_src2;
  result_sign = (res_mant_o >> ADDWID-1) & 0x1;

  // debug the adder
  if(debug_en == 1)
  {
     printf ("src1_o is %llx \n", sprod);
     printf ("src1   is %llx \n", add_src1);
     printf ("src2   is %llx \n", add_src2);
     printf ("exp_diff        is %x \n", exp_diff);
     printf ("res_mant_o      is %llx \n", res_mant_o);

     printf ("res_mant_o  abs    is %llx \n", -res_mant_o);
  }
 
  //sign cout2 cout1 default mantissa
  //positive
  // 0     1                    ...  carry out 2!!
  // 0     0     1              ...  carry out 1!!
  // 0     0     0       1      ...  no carry out
  //
  //negative
  // 1     0    ...                  carry out 2!!
  // 1     1     0      ...   all 0s carry out 2!!
  // 1     1     0      ...  !all 0s carry out 1!!
  // 1     1     1       1    all 0s carry_out_1!!

  bit_t nocry_bit = (res_mant_o >> ADDWID-4) & 0x1;
  bit_t cout1_bit = (res_mant_o >> ADDWID-3) & 0x1;
  bit_t cout2_bit = (res_mant_o >> ADDWID-2) & 0x1;

  bit_t cout1     =   (cout2_bit == 0 &&    cout1_bit == 1 && result_sign == 0) 
                   || (cout2_bit == 1 &&(  (cout1_bit == 0 && (res_mant_o & MASK(ADDWID-3)) != 0)
                                        || (cout1_bit == 1 && (res_mant_o & MASK(ADDWID-3)) == 0)));

  bit_t cout2     =(cout2_bit == 1 && result_sign == 0) || ((cout2_bit == 0 || ((res_mant_o & MASK(ADDWID-2)) == 0)) && result_sign == 1);

  bit_t nocry     =(nocry_bit == 1 && result_sign == 0) || (nocry_bit == 0 && result_sign == 1); 
  
  bit_t underflow = 0;

  if ( res_mant_o == 0 ) // detect equal case
  {
     result_exp = 0;
     result_mant= 0;
  } 
  else if (cout2 == 1) // need to right-shift twice
  {
     if(debug_en == 1)
     printf ("result mant >> 2  \n");
     
     if(exp_larger >= 62)
     result_exp = 0x3f;
     else
     result_exp = (exp_larger + 2) & 0x3f;

     result_mant= res_mant_o >> 2;

     assert(exp_larger < 62);
  }
  else if (cout1 == 1) // need to right-shift once
  { 
     if(debug_en == 1)
     printf ("result mant >> 1  \n");
     
     if(exp_larger >= 63) 
     result_exp = 0x3f;
     else    
     result_exp = (exp_larger + 1) & 0x3f;

     result_mant= res_mant_o >> 1;

     assert(exp_larger < 63);
  }

  else if (nocry == 1) // hold the current value
  {
     if(debug_en == 1)
     printf ("result mant hold  \n");
     
     result_exp = exp_larger & 0x3f;
     result_mant= res_mant_o;
  }
  else
  {
     unsigned shift_z = clz_ADDWID_3( (res_mant_o & MASK(ADDWID-3)), ADDWID-3 ); // need to left-shift lz
     //unsigned shift_o = clo_ADDWID_3( (res_mant_o & MASK(ADDWID-3)), ADDWID-3 ); // need to left-shift lz
     unsigned shift_o = clo_ADDWID_3(res_mant_o, ADDWID-3); // need to left-shift lz
    
     if(result_sign == 1)
     shift = shift_o;
     else
     shift = shift_z;

     //printf ("result sign!!!!! << %x  \n", result_sign);
     //printf ("result shifto!!!!! << %d  \n", shift_o);
 
     if(debug_en == 1)
     printf ("result mant << %d  \n", shift);
  
     if(exp_larger < shift)
     {
        //result_exp = exp_larger & 0x3f;
        //result_mant= res_mant_o;

        result_exp = 0x0;
        result_mant= res_mant_o << exp_larger;
        underflow  = 1;
     }
     else
     { 
        result_exp = (exp_larger - shift) & 0x3f;
        result_mant= res_mant_o << shift;
     }
  }

  *p_sum_exp = result_exp;
  *p_sum     = result_mant;
  return(underflow);
}

bit89_t
fp16_to_sum5(
    fp16_t  a
    )
{
  bit_t   a_sign  = (a>>15)&0x1;
  bit5_t  a_exp   = (a>>10)&0x1f;
  bit10_t a_mant  = a&0x3ff;

  if( a_exp == 0 )
    return 0;

  __uint128_t sum;
  if( a_exp == 0x1f )
  {
    sum = 0x3fffff; // 22 1s
    sum <<= 58;
    if( a_sign )
      sum = -sum;
    sum &= MASK(89);
    return sum;
  }

  sum = 0x400|a_mant;
  if( a_sign )
    sum = -sum;
  sum <<= 24;
  sum <<= (a_exp-1);
  sum &= MASK(89);
  return sum;
}


void
ca_mac5(
    fp16_t    a,
    fp16_t    b,
    bit89_t   sum,
    bit89_t * p_sum
    )
{
  bit_t   a_sign  = (a>>15)&0x1;
  bit5_t  a_exp   = (a>>10)&0x1f;
  bit10_t a_mant  = a&0x3ff;
  bit_t   b_sign  = (b>>15)&0x1;
  bit5_t  b_exp   = (b>>10)&0x1f;
  bit10_t b_mant  = b&0x3ff;

  bit22_t prod;
  if( a_exp == 0 || b_exp == 0 )
    prod = 0;
  else if( a_exp == 0x1f || b_exp == 0x1f )
    prod = 0x3fffff;
  else
    prod = ((1<<10)|a_mant)*((1<<10)|b_mant);

  int sprod; // 23 bit signed integer
  if( a_sign == b_sign )
    sprod = prod;
  else
    sprod = -prod;

  bit5_t  shift;
  if( a_exp == 0 || b_exp == 0 )
    shift = 0;
  else if( a_exp == 0x1f || b_exp == 0x1f )
    shift = 58;
  else
    shift = a_exp+b_exp-2;

  // maximum shift is 58; sprod_sh is 23+58 = 81 bits 
  __int128_t  sprod_sh = (__int128_t)(sprod) << shift;
  __uint128_t temp = sum + sprod_sh;

  *p_sum = temp&MASK(89);
}


fp16_t
sum_to_fp165(
    bit89_t   sum
    )
{
  bit65_t strunc = sum>>24;
  bit_t   sign;
  bit64_t trunc;
  if( (strunc>>64)&0x1 )
  {
    sign = 1;
    trunc = -strunc;
  }
  else
  {
    sign = 0;
    trunc = strunc;
  }

  bit16_t eo;

  if( trunc != 0 )
  {
    unsigned lz = clz_64( trunc, 64 );
    if( lz > 53 )
      eo = 0x0000;      // denorm => 0
    else if( lz <= 23 )
      eo = 0x7c00;      // inf
    else
    {
      unsigned  sh = 53-lz;
      unsigned  trunc_sh = trunc>>sh;
      eo = ((sh+1)<<10)|(trunc_sh&0x3ff);
    }
  }
  else
    eo = 0;

  return (sign<<15)|eo;
}

fp16_t
sum_to_fp162(
    bit_t         underflow,
    bit6_t        sum_exp,
    bitADDWID_t   sum
    )
{
  bit_t   sign;
  int64_t sum_cnv;
  if( (sum>>ADDWID-1)&0x1 )
  {
    sign = 1;
    sum_cnv = -sum;
  }
  else
  {
    sign = 0;
    sum_cnv = sum;
  }

  bit16_t eo;

  if( sum_cnv != 0 && underflow == 0)
  {
    if (sum_exp >= 0x1f)
      eo = 0x7bff;
    else
      eo = (sum_exp<<10) | ((sum_cnv >> GUARD) & 0x3ff);
  }
  else
    eo = 0;

  return (sign<<15)|eo;
}

