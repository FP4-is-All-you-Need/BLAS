#pragma once
#include "awe_blas.h"
#include <utility>
namespace awe {
/* Move-only ownership, with the same status-returning semantics as the C API. */
class plan {
 awe_plan* p_=nullptr;
public:
 plan() noexcept = default;
 ~plan() { awe_plan_destroy(p_); }
 plan(const plan&)=delete;
 plan& operator=(const plan&)=delete;
 plan(plan&& x) noexcept :p_(std::exchange(x.p_,nullptr)) {}
 awe_status create(const awe_gemm_desc& d,const awe_options* o=nullptr) noexcept {
  if(p_)return AWE_INVALID_ARGUMENT;
  return awe_plan_create(&p_,&d,o);
 }
 awe_status workspace_size(size_t& n) const noexcept { return awe_plan_get_workspace_size(p_,&n); }
 awe_plan* get() const noexcept { return p_; }
 awe_status execute(const void* a,const void* b,void* c,void* w,size_t bytes,cudaStream_t s=nullptr,awe_execution_stats* t=nullptr) noexcept {
  return awe_plan_execute(p_,a,b,c,w,bytes,s,t);
 }
};
}
