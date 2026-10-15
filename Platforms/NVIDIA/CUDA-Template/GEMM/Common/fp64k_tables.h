// Include the frozen fixed-size modulus metadata for the selected geometry.
#if AWE_GEOMETRY_K == 96
#if AWE_MODULUS_COUNT == 18
#include "../../../CUDA-Inline/GEMM/SM103a_K96/D53_M18/fp64k_tables.h"
#elif AWE_MODULUS_COUNT == 17
#include "../../../CUDA-Inline/GEMM/SM103a_K96/D53_M17/fp64k_tables.h"
#else
#error Unsupported modulus set
#endif
#elif AWE_GEOMETRY_K == 64
#if AWE_MODULUS_COUNT == 18
#include "../../../CUDA-Inline/GEMM/SM100a_K64/D53_M18/fp64k_tables.h"
#elif AWE_MODULUS_COUNT == 17
#include "../../../CUDA-Inline/GEMM/SM100a_K64/D53_M17/fp64k_tables.h"
#else
#error Unsupported modulus set
#endif
#else
#error Unsupported geometry
#endif
