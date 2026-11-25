.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.text
.p2align 8
.globl f6gemm_awe_modular_v2
.protected f6gemm_awe_modular_v2
.type f6gemm_awe_modular_v2,@function
f6gemm_awe_modular_v2:
 s_load_dwordx2 s[0:1], s[0:1], 0
 s_waitcnt lgkmcnt(0)
product_loop:
 s_load_dwordx2 s[4:5], s[0:1], 0x0 // original PC 0x1500
 s_load_dwordx2 s[8:9], s[0:1], 0x20 // original PC 0x1508
 s_load_dwordx2 s[12:13], s[0:1], 0x30 // original PC 0x1510
 s_load_dwordx2 s[16:17], s[0:1], 0x110 // original PC 0x1518
 s_load_dwordx2 s[20:21], s[0:1], 0x120 // original PC 0x1520
 s_load_dword s32, s[0:1], 0x40 // original PC 0x1528
 s_load_dword s27, s[0:1], 0x60 // original PC 0x1530
 s_load_dword s30, s[0:1], 0x130 // original PC 0x1538
 s_load_dword s31, s[0:1], 0x150 // original PC 0x1540
 s_load_dword s24, s[0:1], 0xe0 // original PC 0x1548
 s_load_dword s25, s[0:1], 0xf0 // original PC 0x1550
 s_load_dword s26, s[0:1], 0x100 // original PC 0x1558
 s_waitcnt lgkmcnt(0) // original PC 0x1560
 s_mov_b32 s10, 0x80000000 // original PC 0x1564
 s_mov_b32 s11, 0x20000 // original PC 0x156c
 s_mov_b32 s14, 0x80000000 // original PC 0x1574
 s_mov_b32 s15, 0x20000 // original PC 0x157c
 s_mov_b32 s18, 0x80000000 // original PC 0x1584
 s_mov_b32 s19, 0x20000 // original PC 0x158c
 s_mov_b32 s22, 0x80000000 // original PC 0x1594
 s_mov_b32 s23, 0x20000 // original PC 0x159c
 s_mov_b32 s6, 0x80000000 // original PC 0x15a4
 s_mov_b32 s7, 0x20000 // original PC 0x15ac
 s_mov_b32 s56, 0x1000 // original PC 0x15b4
 s_mov_b32 s57, 0x2000 // original PC 0x15bc
 s_mov_b32 s58, 0x3000 // original PC 0x15c4
swz_done:
 s_lshr_b32 s36, s26, 7 // original PC 0x1c5c
 s_add_u32 s33, s36, 2 // original PC 0x1c60
 s_mul_i32 s47, s33, 0x6000 // original PC 0x1c64
 s_mul_i32 s48, s3, s47 // original PC 0x1c6c
 s_mul_i32 s49, s2, s47 // original PC 0x1c70
 s_mov_b32 s50, 0 // original PC 0x1c74
 s_mov_b32 s51, 0xc000 // original PC 0x1c78
 s_mov_b32 s52, 0x18000 // original PC 0x1c80
 v_lshrrev_b32_e32 v1, 6, v0 // original PC 0x1c88
 v_and_b32_e32 v2, 63, v0 // original PC 0x1c8c
 v_and_b32_e32 v3, 15, v2 // original PC 0x1c90
 v_lshrrev_b32_e32 v4, 4, v2 // original PC 0x1c94
 v_and_b32_e32 v4, 3, v4 // original PC 0x1c98
 v_lshrrev_b32_e32 v5, 1, v1 // original PC 0x1c9c
 v_and_b32_e32 v6, 1, v1 // original PC 0x1ca0
 s_nop 4 // original PC 0x1ca4
 v_readfirstlane_b32 s45, v1 // original PC 0x1ca8
 s_lshl_b32 s46, s45, 8 // original PC 0x1cac
 s_lshl_b32 s45, s45, 10 // original PC 0x1cb0
 v_mov_b32_e32 v125, 0 // original PC 0x1cb4
 v_accvgpr_write_b32 a0, v125 // original PC 0x1cb8
 v_accvgpr_write_b32 a1, v125 // original PC 0x1cc0
 v_accvgpr_write_b32 a2, v125 // original PC 0x1cc8
 v_accvgpr_write_b32 a3, v125 // original PC 0x1cd0
 v_accvgpr_write_b32 a4, v125 // original PC 0x1cd8
 v_accvgpr_write_b32 a5, v125 // original PC 0x1ce0
 v_accvgpr_write_b32 a6, v125 // original PC 0x1ce8
 v_accvgpr_write_b32 a7, v125 // original PC 0x1cf0
 v_accvgpr_write_b32 a8, v125 // original PC 0x1cf8
 v_accvgpr_write_b32 a9, v125 // original PC 0x1d00
 v_accvgpr_write_b32 a10, v125 // original PC 0x1d08
 v_accvgpr_write_b32 a11, v125 // original PC 0x1d10
 v_accvgpr_write_b32 a12, v125 // original PC 0x1d18
 v_accvgpr_write_b32 a13, v125 // original PC 0x1d20
 v_accvgpr_write_b32 a14, v125 // original PC 0x1d28
 v_accvgpr_write_b32 a15, v125 // original PC 0x1d30
 v_accvgpr_write_b32 a16, v125 // original PC 0x1d38
 v_accvgpr_write_b32 a17, v125 // original PC 0x1d40
 v_accvgpr_write_b32 a18, v125 // original PC 0x1d48
 v_accvgpr_write_b32 a19, v125 // original PC 0x1d50
 v_accvgpr_write_b32 a20, v125 // original PC 0x1d58
 v_accvgpr_write_b32 a21, v125 // original PC 0x1d60
 v_accvgpr_write_b32 a22, v125 // original PC 0x1d68
 v_accvgpr_write_b32 a23, v125 // original PC 0x1d70
 v_accvgpr_write_b32 a24, v125 // original PC 0x1d78
 v_accvgpr_write_b32 a25, v125 // original PC 0x1d80
 v_accvgpr_write_b32 a26, v125 // original PC 0x1d88
 v_accvgpr_write_b32 a27, v125 // original PC 0x1d90
 v_accvgpr_write_b32 a28, v125 // original PC 0x1d98
 v_accvgpr_write_b32 a29, v125 // original PC 0x1da0
 v_accvgpr_write_b32 a30, v125 // original PC 0x1da8
 v_accvgpr_write_b32 a31, v125 // original PC 0x1db0
 v_accvgpr_write_b32 a32, v125 // original PC 0x1db8
 v_accvgpr_write_b32 a33, v125 // original PC 0x1dc0
 v_accvgpr_write_b32 a34, v125 // original PC 0x1dc8
 v_accvgpr_write_b32 a35, v125 // original PC 0x1dd0
 v_accvgpr_write_b32 a36, v125 // original PC 0x1dd8
 v_accvgpr_write_b32 a37, v125 // original PC 0x1de0
 v_accvgpr_write_b32 a38, v125 // original PC 0x1de8
 v_accvgpr_write_b32 a39, v125 // original PC 0x1df0
 v_accvgpr_write_b32 a40, v125 // original PC 0x1df8
 v_accvgpr_write_b32 a41, v125 // original PC 0x1e00
 v_accvgpr_write_b32 a42, v125 // original PC 0x1e08
 v_accvgpr_write_b32 a43, v125 // original PC 0x1e10
 v_accvgpr_write_b32 a44, v125 // original PC 0x1e18
 v_accvgpr_write_b32 a45, v125 // original PC 0x1e20
 v_accvgpr_write_b32 a46, v125 // original PC 0x1e28
 v_accvgpr_write_b32 a47, v125 // original PC 0x1e30
 v_accvgpr_write_b32 a48, v125 // original PC 0x1e38
 v_accvgpr_write_b32 a49, v125 // original PC 0x1e40
 v_accvgpr_write_b32 a50, v125 // original PC 0x1e48
 v_accvgpr_write_b32 a51, v125 // original PC 0x1e50
 v_accvgpr_write_b32 a52, v125 // original PC 0x1e58
 v_accvgpr_write_b32 a53, v125 // original PC 0x1e60
 v_accvgpr_write_b32 a54, v125 // original PC 0x1e68
 v_accvgpr_write_b32 a55, v125 // original PC 0x1e70
 v_accvgpr_write_b32 a56, v125 // original PC 0x1e78
 v_accvgpr_write_b32 a57, v125 // original PC 0x1e80
 v_accvgpr_write_b32 a58, v125 // original PC 0x1e88
 v_accvgpr_write_b32 a59, v125 // original PC 0x1e90
 v_accvgpr_write_b32 a60, v125 // original PC 0x1e98
 v_accvgpr_write_b32 a61, v125 // original PC 0x1ea0
 v_accvgpr_write_b32 a62, v125 // original PC 0x1ea8
 v_accvgpr_write_b32 a63, v125 // original PC 0x1eb0
 v_accvgpr_write_b32 a64, v125 // original PC 0x1eb8
 v_accvgpr_write_b32 a65, v125 // original PC 0x1ec0
 v_accvgpr_write_b32 a66, v125 // original PC 0x1ec8
 v_accvgpr_write_b32 a67, v125 // original PC 0x1ed0
 v_accvgpr_write_b32 a68, v125 // original PC 0x1ed8
 v_accvgpr_write_b32 a69, v125 // original PC 0x1ee0
 v_accvgpr_write_b32 a70, v125 // original PC 0x1ee8
 v_accvgpr_write_b32 a71, v125 // original PC 0x1ef0
 v_accvgpr_write_b32 a72, v125 // original PC 0x1ef8
 v_accvgpr_write_b32 a73, v125 // original PC 0x1f00
 v_accvgpr_write_b32 a74, v125 // original PC 0x1f08
 v_accvgpr_write_b32 a75, v125 // original PC 0x1f10
 v_accvgpr_write_b32 a76, v125 // original PC 0x1f18
 v_accvgpr_write_b32 a77, v125 // original PC 0x1f20
 v_accvgpr_write_b32 a78, v125 // original PC 0x1f28
 v_accvgpr_write_b32 a79, v125 // original PC 0x1f30
 v_accvgpr_write_b32 a80, v125 // original PC 0x1f38
 v_accvgpr_write_b32 a81, v125 // original PC 0x1f40
 v_accvgpr_write_b32 a82, v125 // original PC 0x1f48
 v_accvgpr_write_b32 a83, v125 // original PC 0x1f50
 v_accvgpr_write_b32 a84, v125 // original PC 0x1f58
 v_accvgpr_write_b32 a85, v125 // original PC 0x1f60
 v_accvgpr_write_b32 a86, v125 // original PC 0x1f68
 v_accvgpr_write_b32 a87, v125 // original PC 0x1f70
 v_accvgpr_write_b32 a88, v125 // original PC 0x1f78
 v_accvgpr_write_b32 a89, v125 // original PC 0x1f80
 v_accvgpr_write_b32 a90, v125 // original PC 0x1f88
 v_accvgpr_write_b32 a91, v125 // original PC 0x1f90
 v_accvgpr_write_b32 a92, v125 // original PC 0x1f98
 v_accvgpr_write_b32 a93, v125 // original PC 0x1fa0
 v_accvgpr_write_b32 a94, v125 // original PC 0x1fa8
 v_accvgpr_write_b32 a95, v125 // original PC 0x1fb0
 v_accvgpr_write_b32 a96, v125 // original PC 0x1fb8
 v_accvgpr_write_b32 a97, v125 // original PC 0x1fc0
 v_accvgpr_write_b32 a98, v125 // original PC 0x1fc8
 v_accvgpr_write_b32 a99, v125 // original PC 0x1fd0
 v_accvgpr_write_b32 a100, v125 // original PC 0x1fd8
 v_accvgpr_write_b32 a101, v125 // original PC 0x1fe0
 v_accvgpr_write_b32 a102, v125 // original PC 0x1fe8
 v_accvgpr_write_b32 a103, v125 // original PC 0x1ff0
 v_accvgpr_write_b32 a104, v125 // original PC 0x1ff8
 v_accvgpr_write_b32 a105, v125 // original PC 0x2000
 v_accvgpr_write_b32 a106, v125 // original PC 0x2008
 v_accvgpr_write_b32 a107, v125 // original PC 0x2010
 v_accvgpr_write_b32 a108, v125 // original PC 0x2018
 v_accvgpr_write_b32 a109, v125 // original PC 0x2020
 v_accvgpr_write_b32 a110, v125 // original PC 0x2028
 v_accvgpr_write_b32 a111, v125 // original PC 0x2030
 v_accvgpr_write_b32 a112, v125 // original PC 0x2038
 v_accvgpr_write_b32 a113, v125 // original PC 0x2040
 v_accvgpr_write_b32 a114, v125 // original PC 0x2048
 v_accvgpr_write_b32 a115, v125 // original PC 0x2050
 v_accvgpr_write_b32 a116, v125 // original PC 0x2058
 v_accvgpr_write_b32 a117, v125 // original PC 0x2060
 v_accvgpr_write_b32 a118, v125 // original PC 0x2068
 v_accvgpr_write_b32 a119, v125 // original PC 0x2070
 v_accvgpr_write_b32 a120, v125 // original PC 0x2078
 v_accvgpr_write_b32 a121, v125 // original PC 0x2080
 v_accvgpr_write_b32 a122, v125 // original PC 0x2088
 v_accvgpr_write_b32 a123, v125 // original PC 0x2090
 v_accvgpr_write_b32 a124, v125 // original PC 0x2098
 v_accvgpr_write_b32 a125, v125 // original PC 0x20a0
 v_accvgpr_write_b32 a126, v125 // original PC 0x20a8
 v_accvgpr_write_b32 a127, v125 // original PC 0x20b0
 v_accvgpr_write_b32 a128, v125 // original PC 0x20b8
 v_accvgpr_write_b32 a129, v125 // original PC 0x20c0
 v_accvgpr_write_b32 a130, v125 // original PC 0x20c8
 v_accvgpr_write_b32 a131, v125 // original PC 0x20d0
 v_accvgpr_write_b32 a132, v125 // original PC 0x20d8
 v_accvgpr_write_b32 a133, v125 // original PC 0x20e0
 v_accvgpr_write_b32 a134, v125 // original PC 0x20e8
 v_accvgpr_write_b32 a135, v125 // original PC 0x20f0
 v_accvgpr_write_b32 a136, v125 // original PC 0x20f8
 v_accvgpr_write_b32 a137, v125 // original PC 0x2100
 v_accvgpr_write_b32 a138, v125 // original PC 0x2108
 v_accvgpr_write_b32 a139, v125 // original PC 0x2110
 v_accvgpr_write_b32 a140, v125 // original PC 0x2118
 v_accvgpr_write_b32 a141, v125 // original PC 0x2120
 v_accvgpr_write_b32 a142, v125 // original PC 0x2128
 v_accvgpr_write_b32 a143, v125 // original PC 0x2130
 v_accvgpr_write_b32 a144, v125 // original PC 0x2138
 v_accvgpr_write_b32 a145, v125 // original PC 0x2140
 v_accvgpr_write_b32 a146, v125 // original PC 0x2148
 v_accvgpr_write_b32 a147, v125 // original PC 0x2150
 v_accvgpr_write_b32 a148, v125 // original PC 0x2158
 v_accvgpr_write_b32 a149, v125 // original PC 0x2160
 v_accvgpr_write_b32 a150, v125 // original PC 0x2168
 v_accvgpr_write_b32 a151, v125 // original PC 0x2170
 v_accvgpr_write_b32 a152, v125 // original PC 0x2178
 v_accvgpr_write_b32 a153, v125 // original PC 0x2180
 v_accvgpr_write_b32 a154, v125 // original PC 0x2188
 v_accvgpr_write_b32 a155, v125 // original PC 0x2190
 v_accvgpr_write_b32 a156, v125 // original PC 0x2198
 v_accvgpr_write_b32 a157, v125 // original PC 0x21a0
 v_accvgpr_write_b32 a158, v125 // original PC 0x21a8
 v_accvgpr_write_b32 a159, v125 // original PC 0x21b0
 v_accvgpr_write_b32 a160, v125 // original PC 0x21b8
 v_accvgpr_write_b32 a161, v125 // original PC 0x21c0
 v_accvgpr_write_b32 a162, v125 // original PC 0x21c8
 v_accvgpr_write_b32 a163, v125 // original PC 0x21d0
 v_accvgpr_write_b32 a164, v125 // original PC 0x21d8
 v_accvgpr_write_b32 a165, v125 // original PC 0x21e0
 v_accvgpr_write_b32 a166, v125 // original PC 0x21e8
 v_accvgpr_write_b32 a167, v125 // original PC 0x21f0
 v_accvgpr_write_b32 a168, v125 // original PC 0x21f8
 v_accvgpr_write_b32 a169, v125 // original PC 0x2200
 v_accvgpr_write_b32 a170, v125 // original PC 0x2208
 v_accvgpr_write_b32 a171, v125 // original PC 0x2210
 v_accvgpr_write_b32 a172, v125 // original PC 0x2218
 v_accvgpr_write_b32 a173, v125 // original PC 0x2220
 v_accvgpr_write_b32 a174, v125 // original PC 0x2228
 v_accvgpr_write_b32 a175, v125 // original PC 0x2230
 v_accvgpr_write_b32 a176, v125 // original PC 0x2238
 v_accvgpr_write_b32 a177, v125 // original PC 0x2240
 v_accvgpr_write_b32 a178, v125 // original PC 0x2248
 v_accvgpr_write_b32 a179, v125 // original PC 0x2250
 v_accvgpr_write_b32 a180, v125 // original PC 0x2258
 v_accvgpr_write_b32 a181, v125 // original PC 0x2260
 v_accvgpr_write_b32 a182, v125 // original PC 0x2268
 v_accvgpr_write_b32 a183, v125 // original PC 0x2270
 v_accvgpr_write_b32 a184, v125 // original PC 0x2278
 v_accvgpr_write_b32 a185, v125 // original PC 0x2280
 v_accvgpr_write_b32 a186, v125 // original PC 0x2288
 v_accvgpr_write_b32 a187, v125 // original PC 0x2290
 v_accvgpr_write_b32 a188, v125 // original PC 0x2298
 v_accvgpr_write_b32 a189, v125 // original PC 0x22a0
 v_accvgpr_write_b32 a190, v125 // original PC 0x22a8
 v_accvgpr_write_b32 a191, v125 // original PC 0x22b0
 v_accvgpr_write_b32 a192, v125 // original PC 0x22b8
 v_accvgpr_write_b32 a193, v125 // original PC 0x22c0
 v_accvgpr_write_b32 a194, v125 // original PC 0x22c8
 v_accvgpr_write_b32 a195, v125 // original PC 0x22d0
 v_accvgpr_write_b32 a196, v125 // original PC 0x22d8
 v_accvgpr_write_b32 a197, v125 // original PC 0x22e0
 v_accvgpr_write_b32 a198, v125 // original PC 0x22e8
 v_accvgpr_write_b32 a199, v125 // original PC 0x22f0
 v_accvgpr_write_b32 a200, v125 // original PC 0x22f8
 v_accvgpr_write_b32 a201, v125 // original PC 0x2300
 v_accvgpr_write_b32 a202, v125 // original PC 0x2308
 v_accvgpr_write_b32 a203, v125 // original PC 0x2310
 v_accvgpr_write_b32 a204, v125 // original PC 0x2318
 v_accvgpr_write_b32 a205, v125 // original PC 0x2320
 v_accvgpr_write_b32 a206, v125 // original PC 0x2328
 v_accvgpr_write_b32 a207, v125 // original PC 0x2330
 v_accvgpr_write_b32 a208, v125 // original PC 0x2338
 v_accvgpr_write_b32 a209, v125 // original PC 0x2340
 v_accvgpr_write_b32 a210, v125 // original PC 0x2348
 v_accvgpr_write_b32 a211, v125 // original PC 0x2350
 v_accvgpr_write_b32 a212, v125 // original PC 0x2358
 v_accvgpr_write_b32 a213, v125 // original PC 0x2360
 v_accvgpr_write_b32 a214, v125 // original PC 0x2368
 v_accvgpr_write_b32 a215, v125 // original PC 0x2370
 v_accvgpr_write_b32 a216, v125 // original PC 0x2378
 v_accvgpr_write_b32 a217, v125 // original PC 0x2380
 v_accvgpr_write_b32 a218, v125 // original PC 0x2388
 v_accvgpr_write_b32 a219, v125 // original PC 0x2390
 v_accvgpr_write_b32 a220, v125 // original PC 0x2398
 v_accvgpr_write_b32 a221, v125 // original PC 0x23a0
 v_accvgpr_write_b32 a222, v125 // original PC 0x23a8
 v_accvgpr_write_b32 a223, v125 // original PC 0x23b0
 v_accvgpr_write_b32 a224, v125 // original PC 0x23b8
 v_accvgpr_write_b32 a225, v125 // original PC 0x23c0
 v_accvgpr_write_b32 a226, v125 // original PC 0x23c8
 v_accvgpr_write_b32 a227, v125 // original PC 0x23d0
 v_accvgpr_write_b32 a228, v125 // original PC 0x23d8
 v_accvgpr_write_b32 a229, v125 // original PC 0x23e0
 v_accvgpr_write_b32 a230, v125 // original PC 0x23e8
 v_accvgpr_write_b32 a231, v125 // original PC 0x23f0
 v_accvgpr_write_b32 a232, v125 // original PC 0x23f8
 v_accvgpr_write_b32 a233, v125 // original PC 0x2400
 v_accvgpr_write_b32 a234, v125 // original PC 0x2408
 v_accvgpr_write_b32 a235, v125 // original PC 0x2410
 v_accvgpr_write_b32 a236, v125 // original PC 0x2418
 v_accvgpr_write_b32 a237, v125 // original PC 0x2420
 v_accvgpr_write_b32 a238, v125 // original PC 0x2428
 v_accvgpr_write_b32 a239, v125 // original PC 0x2430
 v_accvgpr_write_b32 a240, v125 // original PC 0x2438
 v_accvgpr_write_b32 a241, v125 // original PC 0x2440
 v_accvgpr_write_b32 a242, v125 // original PC 0x2448
 v_accvgpr_write_b32 a243, v125 // original PC 0x2450
 v_accvgpr_write_b32 a244, v125 // original PC 0x2458
 v_accvgpr_write_b32 a245, v125 // original PC 0x2460
 v_accvgpr_write_b32 a246, v125 // original PC 0x2468
 v_accvgpr_write_b32 a247, v125 // original PC 0x2470
 v_accvgpr_write_b32 a248, v125 // original PC 0x2478
 v_accvgpr_write_b32 a249, v125 // original PC 0x2480
 v_accvgpr_write_b32 a250, v125 // original PC 0x2488
 v_accvgpr_write_b32 a251, v125 // original PC 0x2490
 v_accvgpr_write_b32 a252, v125 // original PC 0x2498
 v_accvgpr_write_b32 a253, v125 // original PC 0x24a0
 v_accvgpr_write_b32 a254, v125 // original PC 0x24a8
 v_accvgpr_write_b32 a255, v125 // original PC 0x24b0
 v_lshlrev_b32_e32 v123, 8, v4 // original PC 0x24b8
 v_lshlrev_b32_e32 v108, 4, v3 // original PC 0x24bc
 v_add_u32_e32 v123, v123, v108 // original PC 0x24c0
 v_lshlrev_b32_e32 v124, 7, v4 // original PC 0x24c4
 v_lshlrev_b32_e32 v108, 3, v3 // original PC 0x24c8
 v_add_u32_e32 v124, v124, v108 // original PC 0x24cc
 v_lshlrev_b32_e32 v108, 13, v5 // original PC 0x24d0
 v_add_u32_e32 v109, v123, v108 // original PC 0x24d4
 v_lshlrev_b32_e32 v108, 12, v5 // original PC 0x24d8
 v_add_u32_e32 v110, v124, v108 // original PC 0x24dc
 v_add_u32_e32 v110, 0x4000, v110 // original PC 0x24e0
 v_lshlrev_b32_e32 v108, 13, v6 // original PC 0x24e8
 v_add_u32_e32 v111, v123, v108 // original PC 0x24ec
 v_add_u32_e32 v111, 0x6000, v111 // original PC 0x24f0
 v_lshlrev_b32_e32 v108, 12, v6 // original PC 0x24f8
 v_add_u32_e32 v112, v124, v108 // original PC 0x24fc
 v_add_u32_e32 v112, 0xa000, v112 // original PC 0x2500
 v_lshlrev_b32_e32 v108, 4, v0 // original PC 0x2508
 v_add_u32_e32 v115, s48, v108 // original PC 0x250c
 v_add_u32_e32 v117, s49, v108 // original PC 0x2510
 v_add_u32_e32 v116, 0x4000, v115 // original PC 0x2514
 v_add_u32_e32 v118, 0x4000, v117 // original PC 0x251c
 s_add_u32 s33, s36, 2 // original PC 0x2524
 s_mul_i32 s61, s33, 0x400 // original PC 0x2528
 s_mul_i32 s62, s3, s61 // original PC 0x2530
 s_mul_i32 s63, s2, s61 // original PC 0x2534
 v_lshlrev_b32_e32 v108, 2, v0 // original PC 0x2538
 v_add_u32_e32 v119, s62, v108 // original PC 0x253c
 v_add_u32_e32 v120, s63, v108 // original PC 0x2540
 v_lshlrev_b32_e32 v121, 9, v5 // original PC 0x2544
 v_lshlrev_b32_e32 v108, 7, v4 // original PC 0x2548
 v_add_u32_e32 v121, v121, v108 // original PC 0x254c
 v_lshlrev_b32_e32 v108, 3, v3 // original PC 0x2550
 v_add_u32_e32 v121, v121, v108 // original PC 0x2554
 v_add_u32_e32 v121, 0xc000, v121 // original PC 0x2558
 v_lshlrev_b32_e32 v122, 9, v6 // original PC 0x2560
 v_lshlrev_b32_e32 v108, 7, v4 // original PC 0x2564
 v_add_u32_e32 v122, v122, v108 // original PC 0x2568
 v_lshlrev_b32_e32 v108, 3, v3 // original PC 0x256c
 v_add_u32_e32 v122, v122, v108 // original PC 0x2570
 v_add_u32_e32 v122, 0xc400, v122 // original PC 0x2574
 s_mov_b32 s53, s50 // original PC 0x257c
 s_mov_b32 s54, s51 // original PC 0x2580
 s_mov_b32 s55, s52 // original PC 0x2584
 s_add_u32 s59, s53, s45 // original PC 0x2588
 s_add_u32 s60, s53, s46 // original PC 0x258c
 s_add_u32 m0, s59, 0 // original PC 0x2590
 buffer_load_dwordx4 v115, s[8:11], 0 offen lds // original PC 0x2594
 s_add_u32 m0, s59, 0x1000 // original PC 0x259c
 buffer_load_dwordx4 v115, s[8:11], s56 offen lds // original PC 0x25a4
 s_add_u32 m0, s59, 0x2000 // original PC 0x25ac
 buffer_load_dwordx4 v115, s[8:11], s57 offen lds // original PC 0x25b4
 s_add_u32 m0, s59, 0x3000 // original PC 0x25bc
 buffer_load_dwordx4 v115, s[8:11], s58 offen lds // original PC 0x25c4
 s_add_u32 m0, s59, 0x4000 // original PC 0x25cc
 buffer_load_dwordx4 v116, s[8:11], 0 offen lds // original PC 0x25d4
 s_add_u32 m0, s59, 0x5000 // original PC 0x25dc
 buffer_load_dwordx4 v116, s[8:11], s56 offen lds // original PC 0x25e4
 s_add_u32 m0, s59, 0x6000 // original PC 0x25ec
 buffer_load_dwordx4 v117, s[12:15], 0 offen lds // original PC 0x25f4
 s_add_u32 m0, s59, 0x7000 // original PC 0x25fc
 buffer_load_dwordx4 v117, s[12:15], s56 offen lds // original PC 0x2604
 s_add_u32 m0, s59, 0x8000 // original PC 0x260c
 buffer_load_dwordx4 v117, s[12:15], s57 offen lds // original PC 0x2614
 s_add_u32 m0, s59, 0x9000 // original PC 0x261c
 buffer_load_dwordx4 v117, s[12:15], s58 offen lds // original PC 0x2624
 s_add_u32 m0, s59, 0xa000 // original PC 0x262c
 buffer_load_dwordx4 v118, s[12:15], 0 offen lds // original PC 0x2634
 s_add_u32 m0, s59, 0xb000 // original PC 0x263c
 buffer_load_dwordx4 v118, s[12:15], s56 offen lds // original PC 0x2644
 v_add_u32_e32 v115, 0x6000, v115 // original PC 0x266c
 v_add_u32_e32 v116, 0x6000, v116 // original PC 0x2674
 v_add_u32_e32 v117, 0x6000, v117 // original PC 0x267c
 v_add_u32_e32 v118, 0x6000, v118 // original PC 0x2684
 s_add_u32 s59, s54, s45 // original PC 0x269c
 s_add_u32 s60, s54, s46 // original PC 0x26a0
 s_add_u32 m0, s59, 0 // original PC 0x26a4
 buffer_load_dwordx4 v115, s[8:11], 0 offen lds // original PC 0x26a8
 s_add_u32 m0, s59, 0x1000 // original PC 0x26b0
 buffer_load_dwordx4 v115, s[8:11], s56 offen lds // original PC 0x26b8
 s_add_u32 m0, s59, 0x2000 // original PC 0x26c0
 buffer_load_dwordx4 v115, s[8:11], s57 offen lds // original PC 0x26c8
 s_add_u32 m0, s59, 0x3000 // original PC 0x26d0
 buffer_load_dwordx4 v115, s[8:11], s58 offen lds // original PC 0x26d8
 s_add_u32 m0, s59, 0x4000 // original PC 0x26e0
 buffer_load_dwordx4 v116, s[8:11], 0 offen lds // original PC 0x26e8
 s_add_u32 m0, s59, 0x5000 // original PC 0x26f0
 buffer_load_dwordx4 v116, s[8:11], s56 offen lds // original PC 0x26f8
 s_add_u32 m0, s59, 0x6000 // original PC 0x2700
 buffer_load_dwordx4 v117, s[12:15], 0 offen lds // original PC 0x2708
 s_add_u32 m0, s59, 0x7000 // original PC 0x2710
 buffer_load_dwordx4 v117, s[12:15], s56 offen lds // original PC 0x2718
 s_add_u32 m0, s59, 0x8000 // original PC 0x2720
 buffer_load_dwordx4 v117, s[12:15], s57 offen lds // original PC 0x2728
 s_add_u32 m0, s59, 0x9000 // original PC 0x2730
 buffer_load_dwordx4 v117, s[12:15], s58 offen lds // original PC 0x2738
 s_add_u32 m0, s59, 0xa000 // original PC 0x2740
 buffer_load_dwordx4 v118, s[12:15], 0 offen lds // original PC 0x2748
 s_add_u32 m0, s59, 0xb000 // original PC 0x2750
 buffer_load_dwordx4 v118, s[12:15], s56 offen lds // original PC 0x2758
 v_mov_b32 v104, 0x82828282
 s_mov_b32 s35, 0 // original PC 0x2780
kloop:
 s_waitcnt vmcnt(12) // original PC 0x2784
 s_barrier // original PC 0x2788
 v_add_u32_e32 v113, s53, v109 // original PC 0x27b4
 v_add_u32_e32 v114, s53, v111 // original PC 0x27b8
 ds_read_b128 v[8:11], v113 // original PC 0x27bc
 ds_read_b128 v[14:17], v113 offset:1024 // original PC 0x27c4
 ds_read_b128 v[20:23], v113 offset:2048 // original PC 0x27cc
 ds_read_b128 v[26:29], v113 offset:3072 // original PC 0x27d4
 ds_read_b128 v[32:35], v113 offset:4096 // original PC 0x27dc
 ds_read_b128 v[38:41], v113 offset:5120 // original PC 0x27e4
 ds_read_b128 v[44:47], v113 offset:6144 // original PC 0x27ec
 ds_read_b128 v[50:53], v113 offset:7168 // original PC 0x27f4
 ds_read_b128 v[56:59], v114 // original PC 0x27fc
 ds_read_b128 v[62:65], v114 offset:1024 // original PC 0x2804
 ds_read_b128 v[68:71], v114 offset:2048 // original PC 0x280c
 ds_read_b128 v[74:77], v114 offset:3072 // original PC 0x2814
 ds_read_b128 v[80:83], v114 offset:4096 // original PC 0x281c
 ds_read_b128 v[86:89], v114 offset:5120 // original PC 0x2824
 ds_read_b128 v[92:95], v114 offset:6144 // original PC 0x282c
 ds_read_b128 v[98:101], v114 offset:7168 // original PC 0x2834
 v_add_u32_e32 v113, s53, v110 // original PC 0x283c
 v_add_u32_e32 v114, s53, v112 // original PC 0x2840
 ds_read_b64 v[12:13], v113 // original PC 0x2844
 ds_read_b64 v[18:19], v113 offset:512 // original PC 0x284c
 ds_read_b64 v[24:25], v113 offset:1024 // original PC 0x2854
 ds_read_b64 v[30:31], v113 offset:1536 // original PC 0x285c
 ds_read_b64 v[36:37], v113 offset:2048 // original PC 0x2864
 ds_read_b64 v[42:43], v113 offset:2560 // original PC 0x286c
 ds_read_b64 v[48:49], v113 offset:3072 // original PC 0x2874
 ds_read_b64 v[54:55], v113 offset:3584 // original PC 0x287c
 ds_read_b64 v[60:61], v114 // original PC 0x2884
 ds_read_b64 v[66:67], v114 offset:512 // original PC 0x288c
 ds_read_b64 v[72:73], v114 offset:1024 // original PC 0x2894
 ds_read_b64 v[78:79], v114 offset:1536 // original PC 0x289c
 ds_read_b64 v[84:85], v114 offset:2048 // original PC 0x28a4
 ds_read_b64 v[90:91], v114 offset:2560 // original PC 0x28ac
 ds_read_b64 v[96:97], v114 offset:3072 // original PC 0x28b4
 ds_read_b64 v[102:103], v114 offset:3584 // original PC 0x28bc
 v_add_u32_e32 v115, 0x6000, v115 // original PC 0x28c4
 v_add_u32_e32 v116, 0x6000, v116 // original PC 0x28cc
 v_add_u32_e32 v117, 0x6000, v117 // original PC 0x28d4
 v_add_u32_e32 v118, 0x6000, v118 // original PC 0x28dc
 s_add_u32 s59, s55, s45 // original PC 0x28f4
 s_add_u32 s60, s55, s46 // original PC 0x28f8
 s_waitcnt lgkmcnt(0) // original PC 0x28fc
 v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[8:13], v[56:61], a[0:3], v104, v104 op_sel_hi:[0,0,0] cbsz:2 blgp:2 // original PC 0x2900
 v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[8:13], v[62:67], a[4:7], v104, v104 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2 // original PC 0x2910
 v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[8:13], v[68:73], a[8:11], v104, v104 op_sel_hi:[0,1,0] cbsz:2 blgp:2 // original PC 0x2920
 v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[8:13], v[74:79], a[12:15], v104, v104 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2 // original PC 0x2930
 v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[8:13], v[80:85], a[16:19], v104, v104 op_sel_hi:[0,0,0] cbsz:2 blgp:2 // original PC 0x2940
 s_add_u32 m0, s59, 0 // original PC 0x2950
 buffer_load_dwordx4 v115, s[8:11], 0 offen lds // original PC 0x2954
 v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[8:13], v[86:91], a[20:23], v104, v104 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2 // original PC 0x295c
 v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[8:13], v[92:97], a[24:27], v104, v104 op_sel_hi:[0,1,0] cbsz:2 blgp:2 // original PC 0x296c
 v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[8:13], v[98:103], a[28:31], v104, v104 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2 // original PC 0x297c
 v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[14:19], v[56:61], a[32:35], v104, v104 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2 // original PC 0x298c
 v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[14:19], v[62:67], a[36:39], v104, v104 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2 // original PC 0x299c
 s_add_u32 m0, s59, 0x1000 // original PC 0x29ac
 buffer_load_dwordx4 v115, s[8:11], s56 offen lds // original PC 0x29b4
 v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[14:19], v[68:73], a[40:43], v104, v104 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2 // original PC 0x29bc
 v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[14:19], v[74:79], a[44:47], v104, v104 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2 // original PC 0x29cc
 v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[14:19], v[80:85], a[48:51], v104, v104 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2 // original PC 0x29dc
 v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[14:19], v[86:91], a[52:55], v104, v104 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2 // original PC 0x29ec
 s_add_u32 m0, s59, 0x2000 // original PC 0x29fc
 buffer_load_dwordx4 v115, s[8:11], s57 offen lds // original PC 0x2a04
 v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[14:19], v[92:97], a[56:59], v104, v104 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2 // original PC 0x2a0c
 v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[14:19], v[98:103], a[60:63], v104, v104 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2 // original PC 0x2a1c
 v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[20:25], v[56:61], a[64:67], v104, v104 op_sel_hi:[1,0,0] cbsz:2 blgp:2 // original PC 0x2a2c
 v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[20:25], v[62:67], a[68:71], v104, v104 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2 // original PC 0x2a3c
 v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[20:25], v[68:73], a[72:75], v104, v104 op_sel_hi:[1,1,0] cbsz:2 blgp:2 // original PC 0x2a4c
 s_add_u32 m0, s59, 0x3000 // original PC 0x2a5c
 buffer_load_dwordx4 v115, s[8:11], s58 offen lds // original PC 0x2a64
 v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[20:25], v[74:79], a[76:79], v104, v104 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2 // original PC 0x2a6c
 v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[20:25], v[80:85], a[80:83], v104, v104 op_sel_hi:[1,0,0] cbsz:2 blgp:2 // original PC 0x2a7c
 v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[20:25], v[86:91], a[84:87], v104, v104 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2 // original PC 0x2a8c
 v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[20:25], v[92:97], a[88:91], v104, v104 op_sel_hi:[1,1,0] cbsz:2 blgp:2 // original PC 0x2a9c
 s_add_u32 m0, s59, 0x4000 // original PC 0x2aac
 buffer_load_dwordx4 v116, s[8:11], 0 offen lds // original PC 0x2ab4
 v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[20:25], v[98:103], a[92:95], v104, v104 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2 // original PC 0x2abc
 v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[26:31], v[56:61], a[96:99], v104, v104 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2 // original PC 0x2acc
 v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[26:31], v[62:67], a[100:103], v104, v104 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2 // original PC 0x2adc
 v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[26:31], v[68:73], a[104:107], v104, v104 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2 // original PC 0x2aec
 v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[26:31], v[74:79], a[108:111], v104, v104 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2 // original PC 0x2afc
 s_add_u32 m0, s59, 0x5000 // original PC 0x2b0c
 buffer_load_dwordx4 v116, s[8:11], s56 offen lds // original PC 0x2b14
 v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[26:31], v[80:85], a[112:115], v104, v104 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2 // original PC 0x2b1c
 v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[26:31], v[86:91], a[116:119], v104, v104 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2 // original PC 0x2b2c
 v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[26:31], v[92:97], a[120:123], v104, v104 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2 // original PC 0x2b3c
 v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[26:31], v[98:103], a[124:127], v104, v104 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2 // original PC 0x2b4c
 s_add_u32 m0, s59, 0x6000 // original PC 0x2b5c
 buffer_load_dwordx4 v117, s[12:15], 0 offen lds // original PC 0x2b64
 v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[32:37], v[56:61], a[128:131], v104, v104 op_sel_hi:[0,0,0] cbsz:2 blgp:2 // original PC 0x2b6c
 v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[32:37], v[62:67], a[132:135], v104, v104 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2 // original PC 0x2b7c
 v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[32:37], v[68:73], a[136:139], v104, v104 op_sel_hi:[0,1,0] cbsz:2 blgp:2 // original PC 0x2b8c
 v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[32:37], v[74:79], a[140:143], v104, v104 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2 // original PC 0x2b9c
 v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[32:37], v[80:85], a[144:147], v104, v104 op_sel_hi:[0,0,0] cbsz:2 blgp:2 // original PC 0x2bac
 s_add_u32 m0, s59, 0x7000 // original PC 0x2bbc
 buffer_load_dwordx4 v117, s[12:15], s56 offen lds // original PC 0x2bc4
 v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[32:37], v[86:91], a[148:151], v104, v104 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2 // original PC 0x2bcc
 v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[32:37], v[92:97], a[152:155], v104, v104 op_sel_hi:[0,1,0] cbsz:2 blgp:2 // original PC 0x2bdc
 v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[32:37], v[98:103], a[156:159], v104, v104 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2 // original PC 0x2bec
 v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[38:43], v[56:61], a[160:163], v104, v104 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2 // original PC 0x2bfc
 v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[38:43], v[62:67], a[164:167], v104, v104 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2 // original PC 0x2c0c
 s_add_u32 m0, s59, 0x8000 // original PC 0x2c1c
 buffer_load_dwordx4 v117, s[12:15], s57 offen lds // original PC 0x2c24
 v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[38:43], v[68:73], a[168:171], v104, v104 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2 // original PC 0x2c2c
 v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[38:43], v[74:79], a[172:175], v104, v104 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2 // original PC 0x2c3c
 v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[38:43], v[80:85], a[176:179], v104, v104 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2 // original PC 0x2c4c
 v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[38:43], v[86:91], a[180:183], v104, v104 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2 // original PC 0x2c5c
 s_add_u32 m0, s59, 0x9000 // original PC 0x2c6c
 buffer_load_dwordx4 v117, s[12:15], s58 offen lds // original PC 0x2c74
 v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[38:43], v[92:97], a[184:187], v104, v104 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2 // original PC 0x2c7c
 v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[38:43], v[98:103], a[188:191], v104, v104 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2 // original PC 0x2c8c
 v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[44:49], v[56:61], a[192:195], v104, v104 op_sel_hi:[1,0,0] cbsz:2 blgp:2 // original PC 0x2c9c
 v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[44:49], v[62:67], a[196:199], v104, v104 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2 // original PC 0x2cac
 v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[44:49], v[68:73], a[200:203], v104, v104 op_sel_hi:[1,1,0] cbsz:2 blgp:2 // original PC 0x2cbc
 s_add_u32 m0, s59, 0xa000 // original PC 0x2ccc
 buffer_load_dwordx4 v118, s[12:15], 0 offen lds // original PC 0x2cd4
 v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[44:49], v[74:79], a[204:207], v104, v104 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2 // original PC 0x2cdc
 v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[44:49], v[80:85], a[208:211], v104, v104 op_sel_hi:[1,0,0] cbsz:2 blgp:2 // original PC 0x2cec
 v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[44:49], v[86:91], a[212:215], v104, v104 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2 // original PC 0x2cfc
 v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[44:49], v[92:97], a[216:219], v104, v104 op_sel_hi:[1,1,0] cbsz:2 blgp:2 // original PC 0x2d0c
 s_add_u32 m0, s59, 0xb000 // original PC 0x2d1c
 buffer_load_dwordx4 v118, s[12:15], s56 offen lds // original PC 0x2d24
 v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[44:49], v[98:103], a[220:223], v104, v104 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2 // original PC 0x2d2c
 v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[50:55], v[56:61], a[224:227], v104, v104 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2 // original PC 0x2d3c
 v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[50:55], v[62:67], a[228:231], v104, v104 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2 // original PC 0x2d4c
 v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[50:55], v[68:73], a[232:235], v104, v104 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2 // original PC 0x2d5c
 v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[50:55], v[74:79], a[236:239], v104, v104 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2 // original PC 0x2d6c
 v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[50:55], v[80:85], a[240:243], v104, v104 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2 // original PC 0x2d8c
 v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[50:55], v[86:91], a[244:247], v104, v104 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2 // original PC 0x2d9c
 v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[50:55], v[92:97], a[248:251], v104, v104 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2 // original PC 0x2dac
 v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[50:55], v[98:103], a[252:255], v104, v104 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2 // original PC 0x2dbc
 s_mov_b32 s33, s53 // original PC 0x2ddc
 s_mov_b32 s53, s54 // original PC 0x2de0
 s_mov_b32 s54, s55 // original PC 0x2de4
 s_mov_b32 s55, s33 // original PC 0x2de8
 s_add_u32 s35, s35, 1 // original PC 0x2dec
 s_cmp_lt_i32 s35, s36 // original PC 0x2df0
 s_cbranch_scc1 kloop // original PC 0x2df4
 s_waitcnt vmcnt(0) lgkmcnt(0)
 s_load_dwordx2 s[8:9], s[0:1], 384
 s_load_dwordx2 s[12:13], s[0:1], 392
 s_load_dwordx4 s[16:19], s[0:1], 400 // p, weight, reciprocal, multiple of p
 s_load_dwordx4 s[20:23], s[0:1], 416 // accumulate, stride, FP32 limit, last in form
 s_load_dword s24, s[0:1], 440       // last in sequence
 s_waitcnt lgkmcnt(0)
 s_mov_b32 s10, 0x80000000
 s_mov_b32 s11, 0x20000
 s_mov_b32 s14, 0x80000000
 s_mov_b32 s15, 0x20000
 v_mov_b32_e32 v60, 0
 v_lshrrev_b32_e32 v1, 6, v0
 v_and_b32_e32 v2, 63, v0
 v_lshrrev_b32_e32 v3, 4, v2
 v_lshrrev_b32_e32 v4, 1, v1
 v_lshlrev_b32_e32 v8, 7, v4
 v_lshlrev_b32_e32 v3, 2, v3
 v_add_u32_e32 v8, v8, v3
 s_lshl_b32 s28, s3, 8
 v_add_u32_e32 v8, s28, v8
 v_and_b32_e32 v3, 15, v2
 v_and_b32_e32 v4, 1, v1
 v_lshlrev_b32_e32 v9, 7, v4
 v_add_u32_e32 v9, v9, v3
 s_lshl_b32 s28, s2, 8
 v_add_u32_e32 v9, s28, v9
 v_lshlrev_b32_e32 v9, 13, v9
 v_add_u32_e32 v9, v9, v8
 v_lshlrev_b32_e32 v9, 1, v9
// Four accumulators per element group are reduced modulo p. Each lane keeps
// two packed pairs across segments and writes them at the end of the modulus.
.macro reduce_group offset, lo, hi, acc0, acc1, acc2, acc3
 v_add_u32_e32 v12, \offset, v9
 v_mov_b32 v48, 0
 v_mov_b32 v49, 0
 s_cmp_eq_u32 s20, 0
 s_cbranch_scc1 .Lno_old_\@
 v_mov_b32 v48, v\lo
 v_mov_b32 v49, v\hi
.Lno_old_\@:
 v_accvgpr_read_b32 v16, \acc0
 v_accvgpr_read_b32 v17, \acc1
 v_accvgpr_read_b32 v18, \acc2
 v_accvgpr_read_b32 v19, \acc3
 s_nop 1
 v_cvt_i32_f32 v25, v16
 v_cvt_f32_i32 v26, v25
 v_cmp_eq_f32 vcc, v26, v16
 v_cndmask_b32 v26, 1, 0, vcc
 v_or_b32 v60, v60, v26
 v_cmp_lt_f32_e64 vcc, |v16|, s22
 v_cndmask_b32 v26, 1, 0, vcc
 v_or_b32 v60, v60, v26
 v_add_u32_e32 v25, s19, v25
 v_mul_hi_u32 v26, s18, v25
 v_mul_lo_u32 v26, s16, v26
 v_sub_u32_e32 v25, v25, v26
 v_bfe_u32 v27, v48, 0, 16
 v_mad_u32_u24 v25, s17, v25, v27
 v_mul_hi_u32 v26, s18, v25
 v_mul_lo_u32 v26, s16, v26
 v_sub_u32_e32 v25, v25, v26
 v_sub_u32_e64 v26, v25, s16
 v_min_u32_e32 v32, v25, v26
 v_cvt_i32_f32 v25, v17
 v_cvt_f32_i32 v26, v25
 v_cmp_eq_f32 vcc, v26, v17
 v_cndmask_b32 v26, 1, 0, vcc
 v_or_b32 v60, v60, v26
 v_cmp_lt_f32_e64 vcc, |v17|, s22
 v_cndmask_b32 v26, 1, 0, vcc
 v_or_b32 v60, v60, v26
 v_add_u32_e32 v25, s19, v25
 v_mul_hi_u32 v26, s18, v25
 v_mul_lo_u32 v26, s16, v26
 v_sub_u32_e32 v25, v25, v26
 v_bfe_u32 v27, v48, 16, 16
 v_mad_u32_u24 v25, s17, v25, v27
 v_mul_hi_u32 v26, s18, v25
 v_mul_lo_u32 v26, s16, v26
 v_sub_u32_e32 v25, v25, v26
 v_sub_u32_e64 v26, v25, s16
 v_min_u32_e32 v33, v25, v26
 v_cvt_i32_f32 v25, v18
 v_cvt_f32_i32 v26, v25
 v_cmp_eq_f32 vcc, v26, v18
 v_cndmask_b32 v26, 1, 0, vcc
 v_or_b32 v60, v60, v26
 v_cmp_lt_f32_e64 vcc, |v18|, s22
 v_cndmask_b32 v26, 1, 0, vcc
 v_or_b32 v60, v60, v26
 v_add_u32_e32 v25, s19, v25
 v_mul_hi_u32 v26, s18, v25
 v_mul_lo_u32 v26, s16, v26
 v_sub_u32_e32 v25, v25, v26
 v_bfe_u32 v27, v49, 0, 16
 v_mad_u32_u24 v25, s17, v25, v27
 v_mul_hi_u32 v26, s18, v25
 v_mul_lo_u32 v26, s16, v26
 v_sub_u32_e32 v25, v25, v26
 v_sub_u32_e64 v26, v25, s16
 v_min_u32_e32 v34, v25, v26
 v_cvt_i32_f32 v25, v19
 v_cvt_f32_i32 v26, v25
 v_cmp_eq_f32 vcc, v26, v19
 v_cndmask_b32 v26, 1, 0, vcc
 v_or_b32 v60, v60, v26
 v_cmp_lt_f32_e64 vcc, |v19|, s22
 v_cndmask_b32 v26, 1, 0, vcc
 v_or_b32 v60, v60, v26
 v_add_u32_e32 v25, s19, v25
 v_mul_hi_u32 v26, s18, v25
 v_mul_lo_u32 v26, s16, v26
 v_sub_u32_e32 v25, v25, v26
 v_bfe_u32 v27, v49, 16, 16
 v_mad_u32_u24 v25, s17, v25, v27
 v_mul_hi_u32 v26, s18, v25
 v_mul_lo_u32 v26, s16, v26
 v_sub_u32_e32 v25, v25, v26
 v_sub_u32_e64 v26, v25, s16
 v_min_u32_e32 v35, v25, v26
 v_lshl_or_b32 v36, v33, 16, v32
 v_lshl_or_b32 v37, v35, 16, v34
 v_mov_b32 v\lo, v36
 v_mov_b32 v\hi, v37
 s_cmp_eq_u32 s23, 0
 s_cbranch_scc1 .Lskip_store_\@
 buffer_store_dwordx2 v[36:37], v12, s[8:11], 0 offen
.Lskip_store_\@:
.endm
 reduce_group 0, 128, 129, a0, a1, a2, a3
 reduce_group 262144, 130, 131, a4, a5, a6, a7
 reduce_group 524288, 132, 133, a8, a9, a10, a11
 reduce_group 786432, 134, 135, a12, a13, a14, a15
 reduce_group 1048576, 136, 137, a16, a17, a18, a19
 reduce_group 1310720, 138, 139, a20, a21, a22, a23
 reduce_group 1572864, 140, 141, a24, a25, a26, a27
 reduce_group 1835008, 142, 143, a28, a29, a30, a31
 reduce_group 32, 144, 145, a32, a33, a34, a35
 reduce_group 262176, 146, 147, a36, a37, a38, a39
 reduce_group 524320, 148, 149, a40, a41, a42, a43
 reduce_group 786464, 150, 151, a44, a45, a46, a47
 reduce_group 1048608, 152, 153, a48, a49, a50, a51
 reduce_group 1310752, 154, 155, a52, a53, a54, a55
 reduce_group 1572896, 156, 157, a56, a57, a58, a59
 reduce_group 1835040, 158, 159, a60, a61, a62, a63
 reduce_group 64, 160, 161, a64, a65, a66, a67
 reduce_group 262208, 162, 163, a68, a69, a70, a71
 reduce_group 524352, 164, 165, a72, a73, a74, a75
 reduce_group 786496, 166, 167, a76, a77, a78, a79
 reduce_group 1048640, 168, 169, a80, a81, a82, a83
 reduce_group 1310784, 170, 171, a84, a85, a86, a87
 reduce_group 1572928, 172, 173, a88, a89, a90, a91
 reduce_group 1835072, 174, 175, a92, a93, a94, a95
 reduce_group 96, 176, 177, a96, a97, a98, a99
 reduce_group 262240, 178, 179, a100, a101, a102, a103
 reduce_group 524384, 180, 181, a104, a105, a106, a107
 reduce_group 786528, 182, 183, a108, a109, a110, a111
 reduce_group 1048672, 184, 185, a112, a113, a114, a115
 reduce_group 1310816, 186, 187, a116, a117, a118, a119
 reduce_group 1572960, 188, 189, a120, a121, a122, a123
 reduce_group 1835104, 190, 191, a124, a125, a126, a127
 reduce_group 128, 192, 193, a128, a129, a130, a131
 reduce_group 262272, 194, 195, a132, a133, a134, a135
 reduce_group 524416, 196, 197, a136, a137, a138, a139
 reduce_group 786560, 198, 199, a140, a141, a142, a143
 reduce_group 1048704, 200, 201, a144, a145, a146, a147
 reduce_group 1310848, 202, 203, a148, a149, a150, a151
 reduce_group 1572992, 204, 205, a152, a153, a154, a155
 reduce_group 1835136, 206, 207, a156, a157, a158, a159
 reduce_group 160, 208, 209, a160, a161, a162, a163
 reduce_group 262304, 210, 211, a164, a165, a166, a167
 reduce_group 524448, 212, 213, a168, a169, a170, a171
 reduce_group 786592, 214, 215, a172, a173, a174, a175
 reduce_group 1048736, 216, 217, a176, a177, a178, a179
 reduce_group 1310880, 218, 219, a180, a181, a182, a183
 reduce_group 1573024, 220, 221, a184, a185, a186, a187
 reduce_group 1835168, 222, 223, a188, a189, a190, a191
 reduce_group 192, 224, 225, a192, a193, a194, a195
 reduce_group 262336, 226, 227, a196, a197, a198, a199
 reduce_group 524480, 228, 229, a200, a201, a202, a203
 reduce_group 786624, 230, 231, a204, a205, a206, a207
 reduce_group 1048768, 232, 233, a208, a209, a210, a211
 reduce_group 1310912, 234, 235, a212, a213, a214, a215
 reduce_group 1573056, 236, 237, a216, a217, a218, a219
 reduce_group 1835200, 238, 239, a220, a221, a222, a223
 reduce_group 224, 240, 241, a224, a225, a226, a227
 reduce_group 262368, 242, 243, a228, a229, a230, a231
 reduce_group 524512, 244, 245, a232, a233, a234, a235
 reduce_group 786656, 246, 247, a236, a237, a238, a239
 reduce_group 1048800, 248, 249, a240, a241, a242, a243
 reduce_group 1310944, 250, 251, a244, a245, a246, a247
 reduce_group 1573088, 252, 253, a248, a249, a250, a251
 reduce_group 1835232, 254, 255, a252, a253, a254, a255
// Any invalid lane sets the shared flag. Valid launches issue no atomics.
 v_cmp_ne_u32_e32 vcc, 0, v60
 s_and_saveexec_b64 s[28:29], vcc
 s_cbranch_execz no_invalid
 v_mov_b32_e32 v14, 0
 v_mov_b32_e32 v15, 1
 buffer_atomic_or v15, v14, s[12:15], 0 offen
no_invalid:
 s_mov_b64 exec, s[28:29]
 s_waitcnt vmcnt(0)
 s_cmp_eq_u32 s24, 1
 s_cbranch_scc1 all_done
 s_add_u32 s0, s0, 448
 s_addc_u32 s1, s1, 0
 s_barrier
 s_branch product_loop
all_done:
 s_endpgm
.size f6gemm_awe_modular_v2, .-f6gemm_awe_modular_v2
.section .rodata,"a",@progbits
.p2align 6
.globl f6gemm_awe_modular_v2.kd
.type f6gemm_awe_modular_v2.kd,@object
f6gemm_awe_modular_v2.kd:
 .incbin "modular_descriptor.bin",0,16
 .quad f6gemm_awe_modular_v2-f6gemm_awe_modular_v2.kd
 .incbin "modular_descriptor.bin",24,40
.size f6gemm_awe_modular_v2.kd,64
.section .note,"a",@note
.p2align 2
 .incbin "modular_note.bin"
