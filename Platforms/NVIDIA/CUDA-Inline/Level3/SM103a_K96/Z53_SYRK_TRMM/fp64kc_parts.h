// the ZGEMM 2M / 3M passes over the moduli of fp64k_tables.h (complex set, tables_complex_limb2.json).
// part -> (modulus index, kind): kind 0 = A+ = r + j i, 1 = A- = r - j i (2M, j = sqrt(-1) mod p), 2 = r, 3 = i, 4 = r + i (3M).
#pragma once
#define FP64KC_NPART 43
static const int fp64kc_part_mod[FP64KC_NPART] = {
    0,  0,  1,  1,  2,  2,  3,  3,  4,  4,  5,  5,  6,  6,  7,  7,  8,  8,  9,  9,  10, 10,
    11, 11, 12, 12, 13, 13, 14, 14, 14, 15, 15, 15, 16, 16, 16, 17, 17, 17, 18, 18, 18};
static const int fp64kc_part_kind[FP64KC_NPART] = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                                                   1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 3,
                                                   4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4};
static const int fp64kc_sqrt_m1[FP64K_NMOD] = {
    70, 28, 12, 37, 15, 33, 10, 22, 34, 27, 11, 23, 9, 6, 0, 0, 0, 0, 0}; // 0 for the 3M moduli
static const int fp64kc_first_part[FP64K_NMOD] = {0,  2,  4,  6,  8,  10, 12, 14, 16, 18,
                                                  20, 22, 24, 26, 28, 31, 34, 37, 40};
