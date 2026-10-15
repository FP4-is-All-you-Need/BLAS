#pragma once
// Select the frozen modulus tables and the generated product/plane branches.
namespace awe::Template {
template <int Count> struct ModulusSetTraits;
template <> struct ModulusSetTraits<18> {
    static constexpr int count = 18;
    static constexpr int table_max_k = Geometry::table_max_k;
    static constexpr bool requires_u8 = Geometry::requires_u8;
};
template <> struct ModulusSetTraits<17> {
    static constexpr int count = 17;
    static constexpr int table_max_k = 8192;
    static constexpr bool requires_u8 = false;
};
using ModulusSet = ModulusSetTraits<AWE_MODULUS_COUNT>;
} // namespace awe::Template
using ModulusSet = awe::Template::ModulusSet;
