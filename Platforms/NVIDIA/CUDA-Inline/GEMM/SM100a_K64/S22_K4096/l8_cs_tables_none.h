#pragma once
// A table set without the counting form (FX 1024): cs_tables_ok() is false (the modulus index -1), so lut_x_host adds no counted
// tables and an FX 1024 kernel does not compile (its static_assert). For L8E_CS_TABLES_FILE in a translation unit of another set.
#define L8CS_N 1
static constexpr int l8cs_q[L8CS_N] = {-1};
static constexpr int l8cs_acc[L8CS_N] = {0};
static constexpr int l8cs_emax[L8CS_N] = {0};
static constexpr int l8cs_off[L8CS_N] = {0};
#define L8CS_ENTRIES 2
static const unsigned short l8cs_ea[L8CS_ENTRIES] = {0, 0};
static const unsigned short l8cs_eb[L8CS_ENTRIES] = {0, 0};
