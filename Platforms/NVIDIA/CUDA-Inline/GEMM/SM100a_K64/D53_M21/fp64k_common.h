#pragma once
// Geometry of the paired-CTA FP4 kernel.
#include <cstdint>
#include <cstddef>
#include "fp64k_tables.h"

namespace fp64k {
constexpr int Side = 256; // Tile rows.
constexpr int Cols = 224; // Tile columns.
// FP64K_MDIM, FP64K_NDIM, and FP64K_KDIM set the compile-time dimensions.
// The encoder and reference share tile counts, packed offsets, and backend cut width.
// Cuts are rounded up to an even number. Partial rows or k tails require a runtime
// encoder that masks them; fixed-size encoders require their documented divisibility.
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
constexpr int TileRows = (M + 255) / 256;  // Number of row tiles.
constexpr int TileCols = (N + 223) / 224;  // Number of column tiles.
constexpr int Tiles = TileRows * TileCols; // Total tile count.
constexpr int Cuts =
    ((K + 63) / 64 + 1) / 2 * 2; // The encoder zero-fills k values beyond the logical extent.
constexpr int CutsPerGroup = 2;
constexpr int Groups = Cuts / CutsPerGroup; // Number of ring groups.
constexpr int Slots = 4;                    // Shared-memory ring slots.
constexpr int Limbs = 2;
constexpr int TileBytes = 4096;
constexpr int Parts = FP64K_NMOD;                                    // One part per modulus.
constexpr int OperandBytes = CutsPerGroup * Limbs * TileBytes;       // Bytes per operand tile.
constexpr int GroupBytes = 2 * OperandBytes;                         // Bytes per ring group.
constexpr int SharedBytes = Slots * GroupBytes;                      // Shared-memory storage size.
constexpr size_t PackedBytes = size_t(2 * Cuts * Limbs) * TileBytes; // Packed operand storage size.
constexpr size_t Count = size_t(M) * N;                              // Matrix element count.

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
// Plane offset within the packed operand.
FP64K_HD constexpr size_t tile_offset(int rank, int cut, int limb) {
    return size_t(rank) * (Cuts * Limbs) + cut * Limbs + limb;
}
// Byte offset within a packed tile.
FP64K_HD constexpr int tile_byte(int row, int w) {
    return (row / 8) * 256 + (w / 16) * 128 + (row % 8) * 16 + w % 16;
}
#undef FP64K_HD
} // namespace fp64k
