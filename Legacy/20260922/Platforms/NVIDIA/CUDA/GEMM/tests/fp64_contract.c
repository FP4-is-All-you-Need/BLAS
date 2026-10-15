#include "awe_blas.h"
#include <stdio.h>
int main(void) {
 int count=0;
 for(int method=0;method<=3;++method) for(int empty=0;empty<2;++empty) {
  awe_gemm_desc d; awe_gemm_desc_init(&d); d.data_type=AWE_F64_F64;
  d.m=d.n=d.k=empty?0:1;
  awe_options o; awe_options_init(&o); o.method=(awe_method)method;
  awe_plan* p=(awe_plan*)(uintptr_t)1;
  if(awe_plan_create(&p,&d,&o)!=AWE_NOT_SUPPORTED || p!=NULL)return 1;
  ++count;
 }
 if(awe_gemm_f64(NULL,NULL,NULL,NULL,NULL,0,NULL,NULL)!=AWE_NOT_SUPPORTED)return 2;
 printf("{\"status\":\"PASS\",\"fp64_rejections\":%d,\"gpu_work\":false}\n",count+1);
 return 0;
}
