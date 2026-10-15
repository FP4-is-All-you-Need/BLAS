// clang-format off
#pragma once
// Generated A-side sign flips; retain the matching table digest in provenance.
// Flips absorb negative weights after per-accumulator normalization. Mechanism A
// only (its product i is face pair (i, i), so plane i carries it alone); one product, four products and 169 have none negative
// (checked here). Index = l8_tables.h's modulus order. Do not edit by hand.

static const unsigned l8_flip_a_K4096[21] = {0,0,0,0,0,0,2,0,6,6,0,6,1,1,0,1,0,0,0,0,0};
static const unsigned l8_flip_a_K8192[21] = {0,0,0,0,0,0,2,0,6,6,0,6,1,1,1,6,0,0,0,0,0};
static const unsigned l8_flip_a_K16384[21] = {0,0,0,0,0,0,6,0,6,7,0,6,7,7,1,6,0,0,0,0,0};
// clang-format on
