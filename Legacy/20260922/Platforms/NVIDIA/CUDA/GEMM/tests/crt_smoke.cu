#define main existing_smoke_main
#include "smoke.cu"
#undef main
int main(){try{
 int cases=0;
 for(int buffers:{3,8,9,10,11})for(int streams:{1,3}) {
  for(bool row:{false,true})for(bool ta:{false,true})for(bool tb:{false,true}) {
   run(7,13,16384,AWE_METHOD_OZAKI2,row,ta,tb,row!=ta,streams,0,0,buffers);++cases;
  }
  run(3,5,32769,AWE_METHOD_OZAKI2,false,true,false,false,streams,16384,1,buffers);++cases;
  run(3,5,131072,AWE_METHOD_OZAKI2,true,false,true,false,streams,16384,1,buffers);++cases;
  run(3,5,257,AWE_METHOD_OZAKI2,false,false,false,true,streams,128,0,buffers);++cases;
  run(3,5,0,AWE_METHOD_OZAKI2,true,false,false,true,streams,0,0,buffers);++cases;
 }
 printf("{\"status\":\"PASS\",\"crt_cases\":%d}\n",cases);return 0;
 }catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return 1;}}
