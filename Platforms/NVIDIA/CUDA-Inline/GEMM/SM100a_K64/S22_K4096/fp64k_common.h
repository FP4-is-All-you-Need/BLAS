#pragma once

#include <cstdint>
#include <cstddef>
#include "fp64k_tables.h"

namespace fp64k {
constexpr int Side = 256;
constexpr int Cols = 224;

#ifndef FP64K_MDIM
#define FP64K_MDIM 8192
#endif
#ifndef FP64K_NDIM
#define FP64K_NDIM 8192
#endif
#ifndef FP64K_KDIM
#define FP64K_KDIM 8192
#endif
constexpr int M = FP64K_MDIM, N = FP64K_NDIM, K = FP64K_KDIM;
constexpr int TileRows = (M + 255) / 256;
constexpr int TileCols = (N + 223) / 224;
constexpr int Tiles = TileRows * TileCols; // 1184 = 74 * 16
constexpr int Cuts = ((K + 63) / 64 + 1) / 2 * 2;
constexpr int CutsPerGroup = 2;
constexpr int Groups = Cuts / CutsPerGroup; // 43
constexpr int Slots = 4;
constexpr int Limbs = 2;
constexpr int TileBytes = 4096;
constexpr int Parts = FP64K_NMOD;
constexpr int OperandBytes = CutsPerGroup * Limbs * TileBytes; // 24,576
constexpr int GroupBytes = 2 * OperandBytes;                   // 49,152
constexpr int SharedBytes = Slots * GroupBytes;                // 147,456 <= 228 KiB
constexpr size_t PackedBytes = size_t(2 * Cuts * Limbs) * TileBytes;
constexpr size_t Count = size_t(M) * N; // 67,108,864

#ifdef __CUDACC__
#define FP64K_HD __host__ __device__
#else
#define FP64K_HD
#endif
FP64K_HD constexpr int group_cut(int g) {
    return CutsPerGroup * g;
}
FP64K_HD constexpr int group_cuts(int) {
    return CutsPerGroup;
}

FP64K_HD constexpr size_t tile_offset(int rank, int cut, int limb) {
    return size_t(rank) * (Cuts * Limbs) + cut * Limbs + limb;
}

FP64K_HD constexpr int tile_byte(int row, int w) {
    return (row / 8) * 256 + (w / 16) * 128 + (row % 8) * 16 + w % 16;
}
#undef FP64K_HD
} // namespace fp64k
