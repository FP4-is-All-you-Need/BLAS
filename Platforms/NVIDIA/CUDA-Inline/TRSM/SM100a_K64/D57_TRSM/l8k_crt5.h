#pragma once

#define L8K_CRT5_NMOD 20
#define L8K_CRT5_PBITS 129
#define L8K_CRT5_KMAX 8192
#define L8K_CRT5_BITS 57
static const int l8k_crt5_moduli[20] = {8,   9,   67,  71,  73,  79,  83,  85,  89,  103,
                                        109, 113, 121, 137, 139, 149, 151, 157, 161, 169};
static const unsigned l8k_crt5_lam0inv[20] = {1u,  1u,  41u, 57u, 61u,  36u, 7u, 64u, 9u, 86u,
                                              91u, 85u, 1u,  1u,  107u, 1u,  1u, 1u,  1u, 1u};
static const unsigned l8k_crt5_p[5] = {0x7f325bd8u, 0x3619718bu, 0xcde3ca8du, 0x06251b7cu,
                                       0x00000001u};
static const unsigned l8k_crt5_ph[5] = {0xbf992decu, 0x9b0cb8c5u, 0x66f1e546u, 0x83128dbeu,
                                        0x00000000u};
static const unsigned long long l8k_crt5_magic = 0x00000000f9ffc495ull;
static const unsigned l8k_crt5_w[20][5] = {
    {0x4fb2e271u, 0xf4498a94u, 0xcd356bf4u, 0x624dea4eu, 0x00000000u},
    {0xf1b04318u, 0x77c9f02bu, 0xc18b1681u, 0x1d209146u, 0x00000000u},
    {0x074e5be0u, 0x169bd3c0u, 0x46c28a24u, 0x6d8d98dcu, 0x00000000u},
    {0xaed6dc38u, 0xc89882feu, 0x1aba4d10u, 0x0b139899u, 0x00000000u},
    {0x738185e0u, 0x205b229du, 0x65ff4907u, 0x47d20e8bu, 0x00000000u},
    {0x72949588u, 0x6fd429feu, 0x5b5ade50u, 0x9552b4ecu, 0x00000000u},
    {0xc54d01d0u, 0xf0ed79a1u, 0x786ba332u, 0x1f956f43u, 0x00000000u},
    {0x2f64f760u, 0xcc0d18deu, 0x802c67cau, 0x030f96a5u, 0x00000001u},
    {0xe8a9f160u, 0xb1357015u, 0x2de354ceu, 0x6a0941c5u, 0x00000000u},
    {0x8d43ad20u, 0xb2ec88dbu, 0x2c4b3c01u, 0x84585378u, 0x00000000u},
    {0x3a6f8050u, 0x20f3bb8cu, 0xbf8ed388u, 0x5b63d5e9u, 0x00000000u},
    {0x24d1a628u, 0x9290a7bcu, 0xe3c38003u, 0xb744fc8fu, 0x00000000u},
    {0x300ce6a0u, 0x8c9249a3u, 0x282e4125u, 0xc7511b3fu, 0x00000000u},
    {0x6384f238u, 0x4b40c0c0u, 0xd89fcb89u, 0x935641e6u, 0x00000000u},
    {0x3994b9d8u, 0x5bcf1c1du, 0x9b4b0627u, 0x8d71f334u, 0x00000000u},
    {0x38470890u, 0x5a2c3ee0u, 0x6f37ad0au, 0x893af12eu, 0x00000000u},
    {0x561a11a0u, 0x9c99de5du, 0x6f5e1acau, 0x760d51e3u, 0x00000000u},
    {0x7ea47230u, 0xde693ab9u, 0x75cf9ab0u, 0xb0fd5569u, 0x00000000u},
    {0x696287e8u, 0xee8410aeu, 0x0e792014u, 0x739ab314u, 0x00000000u},
    {0xc04fe950u, 0xe0206752u, 0x36ed001du, 0x15b75597u, 0x00000000u},
};
