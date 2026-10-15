#ifndef AWE_AWE_H
#define AWE_AWE_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* C ABI. All matrices are column-major device arrays. */
typedef struct awe_plan awe_plan;
typedef struct { double real, imag; } awe_complex;
typedef struct { float real, imag; } awe_complex_float;
typedef enum { AWE_SUCCESS=0, AWE_INVALID_ARGUMENT=1, AWE_NOT_SUPPORTED=2,
               AWE_RUNTIME_ERROR=3 } awe_status;
typedef enum { AWE_DGEMM=1, AWE_ZGEMM=2, AWE_DSYRK=3, AWE_DTRMM=4,
               AWE_ZTRMM=5, AWE_ZHERK=6, AWE_DSYMM=7, AWE_DSYR2K=8,
               AWE_ZHEMM=9, AWE_ZHER2K=10, AWE_ZSYMM=11, AWE_ZSYRK=12,
               AWE_ZSYR2K=13, AWE_SGEMM=14, AWE_SSYMM=15, AWE_SSYRK=16,
               AWE_SSYR2K=17, AWE_STRMM=18, AWE_CGEMM=19, AWE_I8GEMM=20,
               AWE_DTRSM=21 } awe_routine;
typedef enum { AWE_K96=96, AWE_K64=64 } awe_backend;
typedef enum { AWE_MODULI_AUTO=0, AWE_MODULI_17=17, AWE_MODULI_18=18,
               AWE_MODULI_21=21 } awe_moduli;
typedef enum { AWE_NO_TRANSPOSE=0, AWE_TRANSPOSE=1, AWE_CONJ_TRANSPOSE=2 } awe_transpose;
typedef enum { AWE_LOWER=0, AWE_UPPER=1 } awe_triangle;
typedef enum { AWE_LEFT=0, AWE_RIGHT=1 } awe_side;
typedef enum { AWE_NON_UNIT=0, AWE_UNIT=1 } awe_diagonal;
typedef struct {
    size_t struct_size;
    awe_routine routine;
    awe_backend backend;
    awe_moduli moduli;
    int64_t m, n, k, lda, ldb, ldc;
    awe_transpose trans_a, trans_b;
    awe_triangle triangle;
    awe_side side;
    awe_diagonal diagonal;
    awe_complex alpha, beta;
} awe_descriptor;
/* Only the measured default policy (with documented fallbacks) is public. */
typedef enum { AWE_ENGINE_DEFAULT=0 } awe_tuning_policy;
typedef struct {
    size_t struct_size;
    awe_tuning_policy pacing, chunks, order, encoder_minb, pingpong, encoder;
    /* 0 selects the documented default; otherwise select an integer width. */
    int quantize_bits;
    /* NULL (0) selects the default. See Docs/api.md for public engine names.
       Read during create only; the caller retains ownership of this string. */
    const char *engine_name;
} awe_options;
typedef struct {
    int pace, chunks, crt_chunks, item_order, encoder_minb, pingpong;
    int skip_b_padding, pdl, shift_form, extra_fx, encoder_fast, moduli;
} awe_effective_options;
void awe_options_init(awe_options *options);
awe_status awe_plan_create_with_options(awe_plan **out, const awe_descriptor *descriptor,
                                       const awe_options *options, const char *directory);
awe_status awe_plan_effective_options(const awe_plan *, awe_effective_options *out);
/* Initialize first, then set dimensions/leading dimensions explicitly. */
void awe_descriptor_init(awe_descriptor *descriptor, awe_routine routine);
/* directory=NULL searches beside libawe.so. No device work occurs in validate. */
awe_status awe_validate(const awe_descriptor *descriptor);
awe_status awe_plan_create(awe_plan **out, const awe_descriptor *descriptor, const char *directory);
/* Execute is synchronous. Inputs must be finite in the used region, on the
   plan's active device. Output must not overlap either input. SYRK/HERK use
   only A (B=NULL), write lower C, and preserve upper C. TRMM is out-of-place. */
awe_status awe_execute(awe_plan *plan, const void *a, const void *b, void *c);
awe_status awe_dgemm(awe_plan *, const double *, const double *, double *);
awe_status awe_zgemm(awe_plan *, const awe_complex *, const awe_complex *, awe_complex *);
awe_status awe_dsyrk(awe_plan *, const double *, double *);
awe_status awe_dtrmm(awe_plan *, const double *, const double *, double *);
awe_status awe_ztrmm(awe_plan *, const awe_complex *, const awe_complex *, awe_complex *);
awe_status awe_zherk(awe_plan *, const awe_complex *, awe_complex *);
awe_status awe_dsymm(awe_plan *p, const double *a, const double *b, double *c);
awe_status awe_dsyr2k(awe_plan *p, const double *a, const double *b, double *c);
awe_status awe_dtrsm(awe_plan *p, const double *a, const double *b, double *c);
awe_status awe_zhemm(awe_plan *p, const awe_complex *a, const awe_complex *b, awe_complex *c);
awe_status awe_zher2k(awe_plan *p, const awe_complex *a, const awe_complex *b, awe_complex *c);
awe_status awe_zsymm(awe_plan *p, const awe_complex *a, const awe_complex *b, awe_complex *c);
awe_status awe_zsyrk(awe_plan *p, const awe_complex *a, awe_complex *c);
awe_status awe_zsyr2k(awe_plan *p, const awe_complex *a, const awe_complex *b, awe_complex *c);
awe_status awe_sgemm(awe_plan *p, const float *a, const float *b, float *c);
awe_status awe_ssymm(awe_plan *p, const float *a, const float *b, float *c);
awe_status awe_ssyrk(awe_plan *p, const float *a, float *c);
awe_status awe_ssyr2k(awe_plan *p, const float *a, const float *b, float *c);
awe_status awe_strmm(awe_plan *p, const float *a, const float *b, float *c);
awe_status awe_cgemm(awe_plan *p, const awe_complex_float *a, const awe_complex_float *b, awe_complex_float *c);
awe_status awe_i8gemm(awe_plan *p, const int8_t *a, const int8_t *b, int32_t *c);
/* JSON copied into caller storage; required includes NUL. NULL buffer queries size. */
awe_status awe_plan_info(const awe_plan *, char *buffer, size_t capacity, size_t *required);
/* Destroy on the device used by create. A failed destroy leaves the plan alive. */
awe_status awe_plan_destroy(awe_plan *plan);
/* Thread-local diagnostic, valid until the next API call on this thread. */
const char *awe_last_error(void);
#ifdef __cplusplus
}
#endif
#endif
