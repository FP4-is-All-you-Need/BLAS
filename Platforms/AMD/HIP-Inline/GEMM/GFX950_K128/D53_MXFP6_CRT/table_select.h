#pragma once
// The modulus table is chosen at compile time by the inner dimension K of the products
// (AWE_MXFP6_TABLE_K, 8192 for this engine). Both tables are exact for K <= 8192. At K = 8192,
// table_d53_k8192.h needs 31 segments (30 products) against 44 (29 products); at K <= 4096 every
// product of either table fits in one segment, and table_d53_k4096.h needs one product fewer.
#ifndef AWE_MXFP6_TABLE_K
#define AWE_MXFP6_TABLE_K 8192
#endif
#if AWE_MXFP6_TABLE_K > 4096
#include "table_d53_k8192.h"
#define AWE_MXFP6_TABLE_PRODUCTS 30
#define AWE_MXFP6_TABLE_WORDS 12772
#define AWE_MXFP6_TABLE_SEGMENTS_AT_8192 31
#else
#include "table_d53_k4096.h"
#define AWE_MXFP6_TABLE_PRODUCTS 29
#define AWE_MXFP6_TABLE_WORDS 11368
#define AWE_MXFP6_TABLE_SEGMENTS_AT_8192 44
#endif
