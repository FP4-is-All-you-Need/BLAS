#pragma once
// Compile-time geometry for the paired-CTA kernel. No runtime dispatch.
namespace awe::Template {
template <int Cut> struct GeometryTraits;
template <> struct GeometryTraits<96> {
    static constexpr int cut = 96;
    static constexpr int tile_bytes = 6144;
    static constexpr int row_group_bytes = 384;
    static constexpr int encoder_chunks = 12;
    static constexpr int encoder_rows = 23;
    static constexpr int encoder_grid_x = 6;
    static constexpr int scale_columns = 16;
    static constexpr int scale_slots = 4;
    // d18 values; d17 uses ModulusSetTraits<17>.
    static constexpr int table_max_k = 16384;
    static constexpr bool requires_u8 = true;
    // Shared-memory tile descriptor stride at bit 32, in units of 16 bytes.
    static constexpr unsigned long long descriptor_stride = 24ull;
    // tcgen05.mma instruction descriptor; bit 31 differs between the recorded geometries.
    static constexpr unsigned mma_descriptor = 0x90380480u;
};
template <> struct GeometryTraits<64> {
    static constexpr int cut = 64;
    static constexpr int tile_bytes = 4096;
    static constexpr int row_group_bytes = 256;
    static constexpr int encoder_chunks = 8;
    static constexpr int encoder_rows = 32;
    static constexpr int encoder_grid_x = 4;
    static constexpr int scale_columns = 8;
    static constexpr int scale_slots = 7;
    // d18 values; d17 uses ModulusSetTraits<17>.
    static constexpr int table_max_k = 8192;
    static constexpr bool requires_u8 = false;
    // Shared-memory tile descriptor stride at bit 32, in units of 16 bytes.
    static constexpr unsigned long long descriptor_stride = 16ull;
    // tcgen05.mma instruction descriptor; bit 31 differs between the recorded geometries.
    static constexpr unsigned mma_descriptor = 0x10380480u;
};
using Geometry = GeometryTraits<AWE_GEOMETRY_K>;
} // namespace awe::Template
using Geometry = awe::Template::Geometry;
// String concatenation stays at preprocessing time, preserving the info ABI.
#if AWE_GEOMETRY_K == 96 && AWE_MODULUS_COUNT == 18
#define AWE_GEOMETRY_INFO_KMAX "8254"
#else
#define AWE_GEOMETRY_INFO_KMAX "8192"
#endif
#if AWE_GEOMETRY_K == 96 && AWE_MODULUS_COUNT == 18
#define AWE_GEOMETRY_TABLE_ERROR                                                                   \
    "shape: k > 16384 has no table in this form (K segments are not built)"
#else
#define AWE_GEOMETRY_TABLE_ERROR "shape: k > 8192: this stage-2 set has no K 16384 table"
#endif
