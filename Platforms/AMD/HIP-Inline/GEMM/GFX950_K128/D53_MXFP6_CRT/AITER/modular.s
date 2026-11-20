.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.text
.p2align 8
.globl f6gemm_awe_modular_v1
.protected f6gemm_awe_modular_v1
.type f6gemm_awe_modular_v1,@function
f6gemm_awe_modular_v1:
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
 s_lshr_b32 s45, s26, 7 // original PC 0x15cc
 s_cmp_gt_u32 s45, 48 // original PC 0x15d0
 s_branch swz_done // original PC 0x15d4
 s_add_u32 s37, s25, 0xff // original PC 0x15d8
 s_lshr_b32 s37, s37, 8 // original PC 0x15e0
 s_add_u32 s38, s24, 0xff // original PC 0x15e4
 s_lshr_b32 s38, s38, 8 // original PC 0x15ec
 s_mul_i32 s39, s3, s37 // original PC 0x15f0
 s_add_u32 s39, s39, s2 // original PC 0x15f4
 s_mul_i32 s44, s37, 16 // original PC 0x15f8
 s_mov_b32 s40, 0 // original PC 0x15fc
 s_mov_b32 s43, s39 // original PC 0x1600
 s_cmp_ge_u32 s43, s44 // original PC 0x1604
 s_cselect_b32 s45, s44, 0 // original PC 0x1608
 s_cselect_b32 s46, 1, 0 // original PC 0x160c
 s_sub_u32 s43, s43, s45 // original PC 0x1610
 s_add_u32 s40, s40, s46 // original PC 0x1614
 s_cmp_ge_u32 s43, s44 // original PC 0x1618
 s_cselect_b32 s45, s44, 0 // original PC 0x161c
 s_cselect_b32 s46, 1, 0 // original PC 0x1620
 s_sub_u32 s43, s43, s45 // original PC 0x1624
 s_add_u32 s40, s40, s46 // original PC 0x1628
 s_cmp_ge_u32 s43, s44 // original PC 0x162c
 s_cselect_b32 s45, s44, 0 // original PC 0x1630
 s_cselect_b32 s46, 1, 0 // original PC 0x1634
 s_sub_u32 s43, s43, s45 // original PC 0x1638
 s_add_u32 s40, s40, s46 // original PC 0x163c
 s_cmp_ge_u32 s43, s44 // original PC 0x1640
 s_cselect_b32 s45, s44, 0 // original PC 0x1644
 s_cselect_b32 s46, 1, 0 // original PC 0x1648
 s_sub_u32 s43, s43, s45 // original PC 0x164c
 s_add_u32 s40, s40, s46 // original PC 0x1650
 s_cmp_ge_u32 s43, s44 // original PC 0x1654
 s_cselect_b32 s45, s44, 0 // original PC 0x1658
 s_cselect_b32 s46, 1, 0 // original PC 0x165c
 s_sub_u32 s43, s43, s45 // original PC 0x1660
 s_add_u32 s40, s40, s46 // original PC 0x1664
 s_cmp_ge_u32 s43, s44 // original PC 0x1668
 s_cselect_b32 s45, s44, 0 // original PC 0x166c
 s_cselect_b32 s46, 1, 0 // original PC 0x1670
 s_sub_u32 s43, s43, s45 // original PC 0x1674
 s_add_u32 s40, s40, s46 // original PC 0x1678
 s_cmp_ge_u32 s43, s44 // original PC 0x167c
 s_cselect_b32 s45, s44, 0 // original PC 0x1680
 s_cselect_b32 s46, 1, 0 // original PC 0x1684
 s_sub_u32 s43, s43, s45 // original PC 0x1688
 s_add_u32 s40, s40, s46 // original PC 0x168c
 s_cmp_ge_u32 s43, s44 // original PC 0x1690
 s_cselect_b32 s45, s44, 0 // original PC 0x1694
 s_cselect_b32 s46, 1, 0 // original PC 0x1698
 s_sub_u32 s43, s43, s45 // original PC 0x169c
 s_add_u32 s40, s40, s46 // original PC 0x16a0
 s_cmp_ge_u32 s43, s44 // original PC 0x16a4
 s_cselect_b32 s45, s44, 0 // original PC 0x16a8
 s_cselect_b32 s46, 1, 0 // original PC 0x16ac
 s_sub_u32 s43, s43, s45 // original PC 0x16b0
 s_add_u32 s40, s40, s46 // original PC 0x16b4
 s_cmp_ge_u32 s43, s44 // original PC 0x16b8
 s_cselect_b32 s45, s44, 0 // original PC 0x16bc
 s_cselect_b32 s46, 1, 0 // original PC 0x16c0
 s_sub_u32 s43, s43, s45 // original PC 0x16c4
 s_add_u32 s40, s40, s46 // original PC 0x16c8
 s_cmp_ge_u32 s43, s44 // original PC 0x16cc
 s_cselect_b32 s45, s44, 0 // original PC 0x16d0
 s_cselect_b32 s46, 1, 0 // original PC 0x16d4
 s_sub_u32 s43, s43, s45 // original PC 0x16d8
 s_add_u32 s40, s40, s46 // original PC 0x16dc
 s_cmp_ge_u32 s43, s44 // original PC 0x16e0
 s_cselect_b32 s45, s44, 0 // original PC 0x16e4
 s_cselect_b32 s46, 1, 0 // original PC 0x16e8
 s_sub_u32 s43, s43, s45 // original PC 0x16ec
 s_add_u32 s40, s40, s46 // original PC 0x16f0
 s_cmp_ge_u32 s43, s44 // original PC 0x16f4
 s_cselect_b32 s45, s44, 0 // original PC 0x16f8
 s_cselect_b32 s46, 1, 0 // original PC 0x16fc
 s_sub_u32 s43, s43, s45 // original PC 0x1700
 s_add_u32 s40, s40, s46 // original PC 0x1704
 s_cmp_ge_u32 s43, s44 // original PC 0x1708
 s_cselect_b32 s45, s44, 0 // original PC 0x170c
 s_cselect_b32 s46, 1, 0 // original PC 0x1710
 s_sub_u32 s43, s43, s45 // original PC 0x1714
 s_add_u32 s40, s40, s46 // original PC 0x1718
 s_cmp_ge_u32 s43, s44 // original PC 0x171c
 s_cselect_b32 s45, s44, 0 // original PC 0x1720
 s_cselect_b32 s46, 1, 0 // original PC 0x1724
 s_sub_u32 s43, s43, s45 // original PC 0x1728
 s_add_u32 s40, s40, s46 // original PC 0x172c
 s_cmp_ge_u32 s43, s44 // original PC 0x1730
 s_cselect_b32 s45, s44, 0 // original PC 0x1734
 s_cselect_b32 s46, 1, 0 // original PC 0x1738
 s_sub_u32 s43, s43, s45 // original PC 0x173c
 s_add_u32 s40, s40, s46 // original PC 0x1740
 s_lshl_b32 s41, s40, 4 // original PC 0x1744
 s_sub_u32 s42, s38, s41 // original PC 0x1748
 s_min_u32 s42, s42, 16 // original PC 0x174c
 s_mov_b32 s2, 0 // original PC 0x1750
 s_mov_b32 s44, s43 // original PC 0x1754
 s_cmp_ge_u32 s44, s42 // original PC 0x1758
 s_cselect_b32 s45, s42, 0 // original PC 0x175c
 s_cselect_b32 s46, 1, 0 // original PC 0x1760
 s_sub_u32 s44, s44, s45 // original PC 0x1764
 s_add_u32 s2, s2, s46 // original PC 0x1768
 s_cmp_ge_u32 s44, s42 // original PC 0x176c
 s_cselect_b32 s45, s42, 0 // original PC 0x1770
 s_cselect_b32 s46, 1, 0 // original PC 0x1774
 s_sub_u32 s44, s44, s45 // original PC 0x1778
 s_add_u32 s2, s2, s46 // original PC 0x177c
 s_cmp_ge_u32 s44, s42 // original PC 0x1780
 s_cselect_b32 s45, s42, 0 // original PC 0x1784
 s_cselect_b32 s46, 1, 0 // original PC 0x1788
 s_sub_u32 s44, s44, s45 // original PC 0x178c
 s_add_u32 s2, s2, s46 // original PC 0x1790
 s_cmp_ge_u32 s44, s42 // original PC 0x1794
 s_cselect_b32 s45, s42, 0 // original PC 0x1798
 s_cselect_b32 s46, 1, 0 // original PC 0x179c
 s_sub_u32 s44, s44, s45 // original PC 0x17a0
 s_add_u32 s2, s2, s46 // original PC 0x17a4
 s_cmp_ge_u32 s44, s42 // original PC 0x17a8
 s_cselect_b32 s45, s42, 0 // original PC 0x17ac
 s_cselect_b32 s46, 1, 0 // original PC 0x17b0
 s_sub_u32 s44, s44, s45 // original PC 0x17b4
 s_add_u32 s2, s2, s46 // original PC 0x17b8
 s_cmp_ge_u32 s44, s42 // original PC 0x17bc
 s_cselect_b32 s45, s42, 0 // original PC 0x17c0
 s_cselect_b32 s46, 1, 0 // original PC 0x17c4
 s_sub_u32 s44, s44, s45 // original PC 0x17c8
 s_add_u32 s2, s2, s46 // original PC 0x17cc
 s_cmp_ge_u32 s44, s42 // original PC 0x17d0
 s_cselect_b32 s45, s42, 0 // original PC 0x17d4
 s_cselect_b32 s46, 1, 0 // original PC 0x17d8
 s_sub_u32 s44, s44, s45 // original PC 0x17dc
 s_add_u32 s2, s2, s46 // original PC 0x17e0
 s_cmp_ge_u32 s44, s42 // original PC 0x17e4
 s_cselect_b32 s45, s42, 0 // original PC 0x17e8
 s_cselect_b32 s46, 1, 0 // original PC 0x17ec
 s_sub_u32 s44, s44, s45 // original PC 0x17f0
 s_add_u32 s2, s2, s46 // original PC 0x17f4
 s_cmp_ge_u32 s44, s42 // original PC 0x17f8
 s_cselect_b32 s45, s42, 0 // original PC 0x17fc
 s_cselect_b32 s46, 1, 0 // original PC 0x1800
 s_sub_u32 s44, s44, s45 // original PC 0x1804
 s_add_u32 s2, s2, s46 // original PC 0x1808
 s_cmp_ge_u32 s44, s42 // original PC 0x180c
 s_cselect_b32 s45, s42, 0 // original PC 0x1810
 s_cselect_b32 s46, 1, 0 // original PC 0x1814
 s_sub_u32 s44, s44, s45 // original PC 0x1818
 s_add_u32 s2, s2, s46 // original PC 0x181c
 s_cmp_ge_u32 s44, s42 // original PC 0x1820
 s_cselect_b32 s45, s42, 0 // original PC 0x1824
 s_cselect_b32 s46, 1, 0 // original PC 0x1828
 s_sub_u32 s44, s44, s45 // original PC 0x182c
 s_add_u32 s2, s2, s46 // original PC 0x1830
 s_cmp_ge_u32 s44, s42 // original PC 0x1834
 s_cselect_b32 s45, s42, 0 // original PC 0x1838
 s_cselect_b32 s46, 1, 0 // original PC 0x183c
 s_sub_u32 s44, s44, s45 // original PC 0x1840
 s_add_u32 s2, s2, s46 // original PC 0x1844
 s_cmp_ge_u32 s44, s42 // original PC 0x1848
 s_cselect_b32 s45, s42, 0 // original PC 0x184c
 s_cselect_b32 s46, 1, 0 // original PC 0x1850
 s_sub_u32 s44, s44, s45 // original PC 0x1854
 s_add_u32 s2, s2, s46 // original PC 0x1858
 s_cmp_ge_u32 s44, s42 // original PC 0x185c
 s_cselect_b32 s45, s42, 0 // original PC 0x1860
 s_cselect_b32 s46, 1, 0 // original PC 0x1864
 s_sub_u32 s44, s44, s45 // original PC 0x1868
 s_add_u32 s2, s2, s46 // original PC 0x186c
 s_cmp_ge_u32 s44, s42 // original PC 0x1870
 s_cselect_b32 s45, s42, 0 // original PC 0x1874
 s_cselect_b32 s46, 1, 0 // original PC 0x1878
 s_sub_u32 s44, s44, s45 // original PC 0x187c
 s_add_u32 s2, s2, s46 // original PC 0x1880
 s_cmp_ge_u32 s44, s42 // original PC 0x1884
 s_cselect_b32 s45, s42, 0 // original PC 0x1888
 s_cselect_b32 s46, 1, 0 // original PC 0x188c
 s_sub_u32 s44, s44, s45 // original PC 0x1890
 s_add_u32 s2, s2, s46 // original PC 0x1894
 s_cmp_ge_u32 s44, s42 // original PC 0x1898
 s_cselect_b32 s45, s42, 0 // original PC 0x189c
 s_cselect_b32 s46, 1, 0 // original PC 0x18a0
 s_sub_u32 s44, s44, s45 // original PC 0x18a4
 s_add_u32 s2, s2, s46 // original PC 0x18a8
 s_cmp_ge_u32 s44, s42 // original PC 0x18ac
 s_cselect_b32 s45, s42, 0 // original PC 0x18b0
 s_cselect_b32 s46, 1, 0 // original PC 0x18b4
 s_sub_u32 s44, s44, s45 // original PC 0x18b8
 s_add_u32 s2, s2, s46 // original PC 0x18bc
 s_cmp_ge_u32 s44, s42 // original PC 0x18c0
 s_cselect_b32 s45, s42, 0 // original PC 0x18c4
 s_cselect_b32 s46, 1, 0 // original PC 0x18c8
 s_sub_u32 s44, s44, s45 // original PC 0x18cc
 s_add_u32 s2, s2, s46 // original PC 0x18d0
 s_cmp_ge_u32 s44, s42 // original PC 0x18d4
 s_cselect_b32 s45, s42, 0 // original PC 0x18d8
 s_cselect_b32 s46, 1, 0 // original PC 0x18dc
 s_sub_u32 s44, s44, s45 // original PC 0x18e0
 s_add_u32 s2, s2, s46 // original PC 0x18e4
 s_cmp_ge_u32 s44, s42 // original PC 0x18e8
 s_cselect_b32 s45, s42, 0 // original PC 0x18ec
 s_cselect_b32 s46, 1, 0 // original PC 0x18f0
 s_sub_u32 s44, s44, s45 // original PC 0x18f4
 s_add_u32 s2, s2, s46 // original PC 0x18f8
 s_cmp_ge_u32 s44, s42 // original PC 0x18fc
 s_cselect_b32 s45, s42, 0 // original PC 0x1900
 s_cselect_b32 s46, 1, 0 // original PC 0x1904
 s_sub_u32 s44, s44, s45 // original PC 0x1908
 s_add_u32 s2, s2, s46 // original PC 0x190c
 s_cmp_ge_u32 s44, s42 // original PC 0x1910
 s_cselect_b32 s45, s42, 0 // original PC 0x1914
 s_cselect_b32 s46, 1, 0 // original PC 0x1918
 s_sub_u32 s44, s44, s45 // original PC 0x191c
 s_add_u32 s2, s2, s46 // original PC 0x1920
 s_cmp_ge_u32 s44, s42 // original PC 0x1924
 s_cselect_b32 s45, s42, 0 // original PC 0x1928
 s_cselect_b32 s46, 1, 0 // original PC 0x192c
 s_sub_u32 s44, s44, s45 // original PC 0x1930
 s_add_u32 s2, s2, s46 // original PC 0x1934
 s_cmp_ge_u32 s44, s42 // original PC 0x1938
 s_cselect_b32 s45, s42, 0 // original PC 0x193c
 s_cselect_b32 s46, 1, 0 // original PC 0x1940
 s_sub_u32 s44, s44, s45 // original PC 0x1944
 s_add_u32 s2, s2, s46 // original PC 0x1948
 s_cmp_ge_u32 s44, s42 // original PC 0x194c
 s_cselect_b32 s45, s42, 0 // original PC 0x1950
 s_cselect_b32 s46, 1, 0 // original PC 0x1954
 s_sub_u32 s44, s44, s45 // original PC 0x1958
 s_add_u32 s2, s2, s46 // original PC 0x195c
 s_cmp_ge_u32 s44, s42 // original PC 0x1960
 s_cselect_b32 s45, s42, 0 // original PC 0x1964
 s_cselect_b32 s46, 1, 0 // original PC 0x1968
 s_sub_u32 s44, s44, s45 // original PC 0x196c
 s_add_u32 s2, s2, s46 // original PC 0x1970
 s_cmp_ge_u32 s44, s42 // original PC 0x1974
 s_cselect_b32 s45, s42, 0 // original PC 0x1978
 s_cselect_b32 s46, 1, 0 // original PC 0x197c
 s_sub_u32 s44, s44, s45 // original PC 0x1980
 s_add_u32 s2, s2, s46 // original PC 0x1984
 s_cmp_ge_u32 s44, s42 // original PC 0x1988
 s_cselect_b32 s45, s42, 0 // original PC 0x198c
 s_cselect_b32 s46, 1, 0 // original PC 0x1990
 s_sub_u32 s44, s44, s45 // original PC 0x1994
 s_add_u32 s2, s2, s46 // original PC 0x1998
 s_cmp_ge_u32 s44, s42 // original PC 0x199c
 s_cselect_b32 s45, s42, 0 // original PC 0x19a0
 s_cselect_b32 s46, 1, 0 // original PC 0x19a4
 s_sub_u32 s44, s44, s45 // original PC 0x19a8
 s_add_u32 s2, s2, s46 // original PC 0x19ac
 s_cmp_ge_u32 s44, s42 // original PC 0x19b0
 s_cselect_b32 s45, s42, 0 // original PC 0x19b4
 s_cselect_b32 s46, 1, 0 // original PC 0x19b8
 s_sub_u32 s44, s44, s45 // original PC 0x19bc
 s_add_u32 s2, s2, s46 // original PC 0x19c0
 s_cmp_ge_u32 s44, s42 // original PC 0x19c4
 s_cselect_b32 s45, s42, 0 // original PC 0x19c8
 s_cselect_b32 s46, 1, 0 // original PC 0x19cc
 s_sub_u32 s44, s44, s45 // original PC 0x19d0
 s_add_u32 s2, s2, s46 // original PC 0x19d4
 s_cmp_ge_u32 s44, s42 // original PC 0x19d8
 s_cselect_b32 s45, s42, 0 // original PC 0x19dc
 s_cselect_b32 s46, 1, 0 // original PC 0x19e0
 s_sub_u32 s44, s44, s45 // original PC 0x19e4
 s_add_u32 s2, s2, s46 // original PC 0x19e8
 s_cmp_ge_u32 s44, s42 // original PC 0x19ec
 s_cselect_b32 s45, s42, 0 // original PC 0x19f0
 s_cselect_b32 s46, 1, 0 // original PC 0x19f4
 s_sub_u32 s44, s44, s45 // original PC 0x19f8
 s_add_u32 s2, s2, s46 // original PC 0x19fc
 s_cmp_ge_u32 s44, s42 // original PC 0x1a00
 s_cselect_b32 s45, s42, 0 // original PC 0x1a04
 s_cselect_b32 s46, 1, 0 // original PC 0x1a08
 s_sub_u32 s44, s44, s45 // original PC 0x1a0c
 s_add_u32 s2, s2, s46 // original PC 0x1a10
 s_cmp_ge_u32 s44, s42 // original PC 0x1a14
 s_cselect_b32 s45, s42, 0 // original PC 0x1a18
 s_cselect_b32 s46, 1, 0 // original PC 0x1a1c
 s_sub_u32 s44, s44, s45 // original PC 0x1a20
 s_add_u32 s2, s2, s46 // original PC 0x1a24
 s_cmp_ge_u32 s44, s42 // original PC 0x1a28
 s_cselect_b32 s45, s42, 0 // original PC 0x1a2c
 s_cselect_b32 s46, 1, 0 // original PC 0x1a30
 s_sub_u32 s44, s44, s45 // original PC 0x1a34
 s_add_u32 s2, s2, s46 // original PC 0x1a38
 s_cmp_ge_u32 s44, s42 // original PC 0x1a3c
 s_cselect_b32 s45, s42, 0 // original PC 0x1a40
 s_cselect_b32 s46, 1, 0 // original PC 0x1a44
 s_sub_u32 s44, s44, s45 // original PC 0x1a48
 s_add_u32 s2, s2, s46 // original PC 0x1a4c
 s_cmp_ge_u32 s44, s42 // original PC 0x1a50
 s_cselect_b32 s45, s42, 0 // original PC 0x1a54
 s_cselect_b32 s46, 1, 0 // original PC 0x1a58
 s_sub_u32 s44, s44, s45 // original PC 0x1a5c
 s_add_u32 s2, s2, s46 // original PC 0x1a60
 s_cmp_ge_u32 s44, s42 // original PC 0x1a64
 s_cselect_b32 s45, s42, 0 // original PC 0x1a68
 s_cselect_b32 s46, 1, 0 // original PC 0x1a6c
 s_sub_u32 s44, s44, s45 // original PC 0x1a70
 s_add_u32 s2, s2, s46 // original PC 0x1a74
 s_cmp_ge_u32 s44, s42 // original PC 0x1a78
 s_cselect_b32 s45, s42, 0 // original PC 0x1a7c
 s_cselect_b32 s46, 1, 0 // original PC 0x1a80
 s_sub_u32 s44, s44, s45 // original PC 0x1a84
 s_add_u32 s2, s2, s46 // original PC 0x1a88
 s_cmp_ge_u32 s44, s42 // original PC 0x1a8c
 s_cselect_b32 s45, s42, 0 // original PC 0x1a90
 s_cselect_b32 s46, 1, 0 // original PC 0x1a94
 s_sub_u32 s44, s44, s45 // original PC 0x1a98
 s_add_u32 s2, s2, s46 // original PC 0x1a9c
 s_cmp_ge_u32 s44, s42 // original PC 0x1aa0
 s_cselect_b32 s45, s42, 0 // original PC 0x1aa4
 s_cselect_b32 s46, 1, 0 // original PC 0x1aa8
 s_sub_u32 s44, s44, s45 // original PC 0x1aac
 s_add_u32 s2, s2, s46 // original PC 0x1ab0
 s_cmp_ge_u32 s44, s42 // original PC 0x1ab4
 s_cselect_b32 s45, s42, 0 // original PC 0x1ab8
 s_cselect_b32 s46, 1, 0 // original PC 0x1abc
 s_sub_u32 s44, s44, s45 // original PC 0x1ac0
 s_add_u32 s2, s2, s46 // original PC 0x1ac4
 s_cmp_ge_u32 s44, s42 // original PC 0x1ac8
 s_cselect_b32 s45, s42, 0 // original PC 0x1acc
 s_cselect_b32 s46, 1, 0 // original PC 0x1ad0
 s_sub_u32 s44, s44, s45 // original PC 0x1ad4
 s_add_u32 s2, s2, s46 // original PC 0x1ad8
 s_cmp_ge_u32 s44, s42 // original PC 0x1adc
 s_cselect_b32 s45, s42, 0 // original PC 0x1ae0
 s_cselect_b32 s46, 1, 0 // original PC 0x1ae4
 s_sub_u32 s44, s44, s45 // original PC 0x1ae8
 s_add_u32 s2, s2, s46 // original PC 0x1aec
 s_cmp_ge_u32 s44, s42 // original PC 0x1af0
 s_cselect_b32 s45, s42, 0 // original PC 0x1af4
 s_cselect_b32 s46, 1, 0 // original PC 0x1af8
 s_sub_u32 s44, s44, s45 // original PC 0x1afc
 s_add_u32 s2, s2, s46 // original PC 0x1b00
 s_cmp_ge_u32 s44, s42 // original PC 0x1b04
 s_cselect_b32 s45, s42, 0 // original PC 0x1b08
 s_cselect_b32 s46, 1, 0 // original PC 0x1b0c
 s_sub_u32 s44, s44, s45 // original PC 0x1b10
 s_add_u32 s2, s2, s46 // original PC 0x1b14
 s_cmp_ge_u32 s44, s42 // original PC 0x1b18
 s_cselect_b32 s45, s42, 0 // original PC 0x1b1c
 s_cselect_b32 s46, 1, 0 // original PC 0x1b20
 s_sub_u32 s44, s44, s45 // original PC 0x1b24
 s_add_u32 s2, s2, s46 // original PC 0x1b28
 s_cmp_ge_u32 s44, s42 // original PC 0x1b2c
 s_cselect_b32 s45, s42, 0 // original PC 0x1b30
 s_cselect_b32 s46, 1, 0 // original PC 0x1b34
 s_sub_u32 s44, s44, s45 // original PC 0x1b38
 s_add_u32 s2, s2, s46 // original PC 0x1b3c
 s_cmp_ge_u32 s44, s42 // original PC 0x1b40
 s_cselect_b32 s45, s42, 0 // original PC 0x1b44
 s_cselect_b32 s46, 1, 0 // original PC 0x1b48
 s_sub_u32 s44, s44, s45 // original PC 0x1b4c
 s_add_u32 s2, s2, s46 // original PC 0x1b50
 s_cmp_ge_u32 s44, s42 // original PC 0x1b54
 s_cselect_b32 s45, s42, 0 // original PC 0x1b58
 s_cselect_b32 s46, 1, 0 // original PC 0x1b5c
 s_sub_u32 s44, s44, s45 // original PC 0x1b60
 s_add_u32 s2, s2, s46 // original PC 0x1b64
 s_cmp_ge_u32 s44, s42 // original PC 0x1b68
 s_cselect_b32 s45, s42, 0 // original PC 0x1b6c
 s_cselect_b32 s46, 1, 0 // original PC 0x1b70
 s_sub_u32 s44, s44, s45 // original PC 0x1b74
 s_add_u32 s2, s2, s46 // original PC 0x1b78
 s_cmp_ge_u32 s44, s42 // original PC 0x1b7c
 s_cselect_b32 s45, s42, 0 // original PC 0x1b80
 s_cselect_b32 s46, 1, 0 // original PC 0x1b84
 s_sub_u32 s44, s44, s45 // original PC 0x1b88
 s_add_u32 s2, s2, s46 // original PC 0x1b8c
 s_cmp_ge_u32 s44, s42 // original PC 0x1b90
 s_cselect_b32 s45, s42, 0 // original PC 0x1b94
 s_cselect_b32 s46, 1, 0 // original PC 0x1b98
 s_sub_u32 s44, s44, s45 // original PC 0x1b9c
 s_add_u32 s2, s2, s46 // original PC 0x1ba0
 s_cmp_ge_u32 s44, s42 // original PC 0x1ba4
 s_cselect_b32 s45, s42, 0 // original PC 0x1ba8
 s_cselect_b32 s46, 1, 0 // original PC 0x1bac
 s_sub_u32 s44, s44, s45 // original PC 0x1bb0
 s_add_u32 s2, s2, s46 // original PC 0x1bb4
 s_cmp_ge_u32 s44, s42 // original PC 0x1bb8
 s_cselect_b32 s45, s42, 0 // original PC 0x1bbc
 s_cselect_b32 s46, 1, 0 // original PC 0x1bc0
 s_sub_u32 s44, s44, s45 // original PC 0x1bc4
 s_add_u32 s2, s2, s46 // original PC 0x1bc8
 s_cmp_ge_u32 s44, s42 // original PC 0x1bcc
 s_cselect_b32 s45, s42, 0 // original PC 0x1bd0
 s_cselect_b32 s46, 1, 0 // original PC 0x1bd4
 s_sub_u32 s44, s44, s45 // original PC 0x1bd8
 s_add_u32 s2, s2, s46 // original PC 0x1bdc
 s_cmp_ge_u32 s44, s42 // original PC 0x1be0
 s_cselect_b32 s45, s42, 0 // original PC 0x1be4
 s_cselect_b32 s46, 1, 0 // original PC 0x1be8
 s_sub_u32 s44, s44, s45 // original PC 0x1bec
 s_add_u32 s2, s2, s46 // original PC 0x1bf0
 s_cmp_ge_u32 s44, s42 // original PC 0x1bf4
 s_cselect_b32 s45, s42, 0 // original PC 0x1bf8
 s_cselect_b32 s46, 1, 0 // original PC 0x1bfc
 s_sub_u32 s44, s44, s45 // original PC 0x1c00
 s_add_u32 s2, s2, s46 // original PC 0x1c04
 s_cmp_ge_u32 s44, s42 // original PC 0x1c08
 s_cselect_b32 s45, s42, 0 // original PC 0x1c0c
 s_cselect_b32 s46, 1, 0 // original PC 0x1c10
 s_sub_u32 s44, s44, s45 // original PC 0x1c14
 s_add_u32 s2, s2, s46 // original PC 0x1c18
 s_cmp_ge_u32 s44, s42 // original PC 0x1c1c
 s_cselect_b32 s45, s42, 0 // original PC 0x1c20
 s_cselect_b32 s46, 1, 0 // original PC 0x1c24
 s_sub_u32 s44, s44, s45 // original PC 0x1c28
 s_add_u32 s2, s2, s46 // original PC 0x1c2c
 s_cmp_ge_u32 s44, s42 // original PC 0x1c30
 s_cselect_b32 s45, s42, 0 // original PC 0x1c34
 s_cselect_b32 s46, 1, 0 // original PC 0x1c38
 s_sub_u32 s44, s44, s45 // original PC 0x1c3c
 s_add_u32 s2, s2, s46 // original PC 0x1c40
 s_cmp_ge_u32 s44, s42 // original PC 0x1c44
 s_cselect_b32 s45, s42, 0 // original PC 0x1c48
 s_cselect_b32 s46, 1, 0 // original PC 0x1c4c
 s_sub_u32 s44, s44, s45 // original PC 0x1c50
 s_add_u32 s2, s2, s46 // original PC 0x1c54
 s_add_u32 s3, s41, s44 // original PC 0x1c58
swz_done:
 s_lshr_b32 s36, s26, 7 // original PC 0x1c5c
 s_add_u32 s33, s36, 2 // original PC 0x1c60
 s_mul_i32 s47, s33, 0x6000 // original PC 0x1c64
 s_mul_i32 s48, s3, s47 // original PC 0x1c6c
 s_mul_i32 s49, s2, s47 // original PC 0x1c70
 s_mov_b32 s50, 0 // original PC 0x1c74
 s_mov_b32 s51, 0xc800 // original PC 0x1c78
 s_mov_b32 s52, 0x19000 // original PC 0x1c80
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
 s_add_u32 m0, s60, 0xc000 // original PC 0x264c
 buffer_load_dword v119, s[16:19], 0 offen lds // original PC 0x2654
 s_add_u32 m0, s60, 0xc400 // original PC 0x265c
 buffer_load_dword v120, s[20:23], 0 offen lds // original PC 0x2664
 v_add_u32_e32 v115, 0x6000, v115 // original PC 0x266c
 v_add_u32_e32 v116, 0x6000, v116 // original PC 0x2674
 v_add_u32_e32 v117, 0x6000, v117 // original PC 0x267c
 v_add_u32_e32 v118, 0x6000, v118 // original PC 0x2684
 v_add_u32_e32 v119, 0x400, v119 // original PC 0x268c
 v_add_u32_e32 v120, 0x400, v120 // original PC 0x2694
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
 s_add_u32 m0, s60, 0xc000 // original PC 0x2760
 buffer_load_dword v119, s[16:19], 0 offen lds // original PC 0x2768
 s_add_u32 m0, s60, 0xc400 // original PC 0x2770
 buffer_load_dword v120, s[20:23], 0 offen lds // original PC 0x2778
 s_mov_b32 s35, 0 // original PC 0x2780
kloop:
 s_waitcnt vmcnt(14) // original PC 0x2784
 s_barrier // original PC 0x2788
 v_add_u32_e32 v113, s53, v121 // original PC 0x278c
 v_add_u32_e32 v114, s53, v122 // original PC 0x2790
 ds_read_b32 v104, v113 // original PC 0x2794
 ds_read_b32 v105, v113 offset:4 // original PC 0x279c
 ds_read_b32 v106, v114 // original PC 0x27a4
 ds_read_b32 v107, v114 offset:4 // original PC 0x27ac
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
 v_add_u32_e32 v119, 0x400, v119 // original PC 0x28e4
 v_add_u32_e32 v120, 0x400, v120 // original PC 0x28ec
 s_add_u32 s59, s55, s45 // original PC 0x28f4
 s_add_u32 s60, s55, s46 // original PC 0x28f8
 s_waitcnt lgkmcnt(0) // original PC 0x28fc
 v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[8:13], v[56:61], a[0:3], v104, v106 op_sel_hi:[0,0,0] cbsz:2 blgp:2 // original PC 0x2900
 v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[8:13], v[62:67], a[4:7], v104, v106 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2 // original PC 0x2910
 v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[8:13], v[68:73], a[8:11], v104, v106 op_sel_hi:[0,1,0] cbsz:2 blgp:2 // original PC 0x2920
 v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[8:13], v[74:79], a[12:15], v104, v106 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2 // original PC 0x2930
 v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[8:13], v[80:85], a[16:19], v104, v107 op_sel_hi:[0,0,0] cbsz:2 blgp:2 // original PC 0x2940
 s_add_u32 m0, s59, 0 // original PC 0x2950
 buffer_load_dwordx4 v115, s[8:11], 0 offen lds // original PC 0x2954
 v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[8:13], v[86:91], a[20:23], v104, v107 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2 // original PC 0x295c
 v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[8:13], v[92:97], a[24:27], v104, v107 op_sel_hi:[0,1,0] cbsz:2 blgp:2 // original PC 0x296c
 v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[8:13], v[98:103], a[28:31], v104, v107 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2 // original PC 0x297c
 v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[14:19], v[56:61], a[32:35], v104, v106 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2 // original PC 0x298c
 v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[14:19], v[62:67], a[36:39], v104, v106 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2 // original PC 0x299c
 s_add_u32 m0, s59, 0x1000 // original PC 0x29ac
 buffer_load_dwordx4 v115, s[8:11], s56 offen lds // original PC 0x29b4
 v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[14:19], v[68:73], a[40:43], v104, v106 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2 // original PC 0x29bc
 v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[14:19], v[74:79], a[44:47], v104, v106 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2 // original PC 0x29cc
 v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[14:19], v[80:85], a[48:51], v104, v107 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2 // original PC 0x29dc
 v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[14:19], v[86:91], a[52:55], v104, v107 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2 // original PC 0x29ec
 s_add_u32 m0, s59, 0x2000 // original PC 0x29fc
 buffer_load_dwordx4 v115, s[8:11], s57 offen lds // original PC 0x2a04
 v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[14:19], v[92:97], a[56:59], v104, v107 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2 // original PC 0x2a0c
 v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[14:19], v[98:103], a[60:63], v104, v107 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2 // original PC 0x2a1c
 v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[20:25], v[56:61], a[64:67], v104, v106 op_sel_hi:[1,0,0] cbsz:2 blgp:2 // original PC 0x2a2c
 v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[20:25], v[62:67], a[68:71], v104, v106 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2 // original PC 0x2a3c
 v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[20:25], v[68:73], a[72:75], v104, v106 op_sel_hi:[1,1,0] cbsz:2 blgp:2 // original PC 0x2a4c
 s_add_u32 m0, s59, 0x3000 // original PC 0x2a5c
 buffer_load_dwordx4 v115, s[8:11], s58 offen lds // original PC 0x2a64
 v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[20:25], v[74:79], a[76:79], v104, v106 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2 // original PC 0x2a6c
 v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[20:25], v[80:85], a[80:83], v104, v107 op_sel_hi:[1,0,0] cbsz:2 blgp:2 // original PC 0x2a7c
 v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[20:25], v[86:91], a[84:87], v104, v107 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2 // original PC 0x2a8c
 v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[20:25], v[92:97], a[88:91], v104, v107 op_sel_hi:[1,1,0] cbsz:2 blgp:2 // original PC 0x2a9c
 s_add_u32 m0, s59, 0x4000 // original PC 0x2aac
 buffer_load_dwordx4 v116, s[8:11], 0 offen lds // original PC 0x2ab4
 v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[20:25], v[98:103], a[92:95], v104, v107 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2 // original PC 0x2abc
 v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[26:31], v[56:61], a[96:99], v104, v106 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2 // original PC 0x2acc
 v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[26:31], v[62:67], a[100:103], v104, v106 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2 // original PC 0x2adc
 v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[26:31], v[68:73], a[104:107], v104, v106 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2 // original PC 0x2aec
 v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[26:31], v[74:79], a[108:111], v104, v106 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2 // original PC 0x2afc
 s_add_u32 m0, s59, 0x5000 // original PC 0x2b0c
 buffer_load_dwordx4 v116, s[8:11], s56 offen lds // original PC 0x2b14
 v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[26:31], v[80:85], a[112:115], v104, v107 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2 // original PC 0x2b1c
 v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[26:31], v[86:91], a[116:119], v104, v107 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2 // original PC 0x2b2c
 v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[26:31], v[92:97], a[120:123], v104, v107 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2 // original PC 0x2b3c
 v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[26:31], v[98:103], a[124:127], v104, v107 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2 // original PC 0x2b4c
 s_add_u32 m0, s59, 0x6000 // original PC 0x2b5c
 buffer_load_dwordx4 v117, s[12:15], 0 offen lds // original PC 0x2b64
 v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[32:37], v[56:61], a[128:131], v105, v106 op_sel_hi:[0,0,0] cbsz:2 blgp:2 // original PC 0x2b6c
 v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[32:37], v[62:67], a[132:135], v105, v106 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2 // original PC 0x2b7c
 v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[32:37], v[68:73], a[136:139], v105, v106 op_sel_hi:[0,1,0] cbsz:2 blgp:2 // original PC 0x2b8c
 v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[32:37], v[74:79], a[140:143], v105, v106 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2 // original PC 0x2b9c
 v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[32:37], v[80:85], a[144:147], v105, v107 op_sel_hi:[0,0,0] cbsz:2 blgp:2 // original PC 0x2bac
 s_add_u32 m0, s59, 0x7000 // original PC 0x2bbc
 buffer_load_dwordx4 v117, s[12:15], s56 offen lds // original PC 0x2bc4
 v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[32:37], v[86:91], a[148:151], v105, v107 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2 // original PC 0x2bcc
 v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[32:37], v[92:97], a[152:155], v105, v107 op_sel_hi:[0,1,0] cbsz:2 blgp:2 // original PC 0x2bdc
 v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[32:37], v[98:103], a[156:159], v105, v107 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2 // original PC 0x2bec
 v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[38:43], v[56:61], a[160:163], v105, v106 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2 // original PC 0x2bfc
 v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[38:43], v[62:67], a[164:167], v105, v106 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2 // original PC 0x2c0c
 s_add_u32 m0, s59, 0x8000 // original PC 0x2c1c
 buffer_load_dwordx4 v117, s[12:15], s57 offen lds // original PC 0x2c24
 v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[38:43], v[68:73], a[168:171], v105, v106 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2 // original PC 0x2c2c
 v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[38:43], v[74:79], a[172:175], v105, v106 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2 // original PC 0x2c3c
 v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[38:43], v[80:85], a[176:179], v105, v107 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2 // original PC 0x2c4c
 v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[38:43], v[86:91], a[180:183], v105, v107 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:2 blgp:2 // original PC 0x2c5c
 s_add_u32 m0, s59, 0x9000 // original PC 0x2c6c
 buffer_load_dwordx4 v117, s[12:15], s58 offen lds // original PC 0x2c74
 v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[38:43], v[92:97], a[184:187], v105, v107 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2 // original PC 0x2c7c
 v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[38:43], v[98:103], a[188:191], v105, v107 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:2 blgp:2 // original PC 0x2c8c
 v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[44:49], v[56:61], a[192:195], v105, v106 op_sel_hi:[1,0,0] cbsz:2 blgp:2 // original PC 0x2c9c
 v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[44:49], v[62:67], a[196:199], v105, v106 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2 // original PC 0x2cac
 v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[44:49], v[68:73], a[200:203], v105, v106 op_sel_hi:[1,1,0] cbsz:2 blgp:2 // original PC 0x2cbc
 s_add_u32 m0, s59, 0xa000 // original PC 0x2ccc
 buffer_load_dwordx4 v118, s[12:15], 0 offen lds // original PC 0x2cd4
 v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[44:49], v[74:79], a[204:207], v105, v106 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2 // original PC 0x2cdc
 v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[44:49], v[80:85], a[208:211], v105, v107 op_sel_hi:[1,0,0] cbsz:2 blgp:2 // original PC 0x2cec
 v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[44:49], v[86:91], a[212:215], v105, v107 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2 // original PC 0x2cfc
 v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[44:49], v[92:97], a[216:219], v105, v107 op_sel_hi:[1,1,0] cbsz:2 blgp:2 // original PC 0x2d0c
 s_add_u32 m0, s59, 0xb000 // original PC 0x2d1c
 buffer_load_dwordx4 v118, s[12:15], s56 offen lds // original PC 0x2d24
 v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[44:49], v[98:103], a[220:223], v105, v107 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2 // original PC 0x2d2c
 v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[50:55], v[56:61], a[224:227], v105, v106 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2 // original PC 0x2d3c
 v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[50:55], v[62:67], a[228:231], v105, v106 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2 // original PC 0x2d4c
 v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[50:55], v[68:73], a[232:235], v105, v106 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2 // original PC 0x2d5c
 v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[50:55], v[74:79], a[236:239], v105, v106 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2 // original PC 0x2d6c
 s_add_u32 m0, s60, 0xc000 // original PC 0x2d7c
 buffer_load_dword v119, s[16:19], 0 offen lds // original PC 0x2d84
 v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[50:55], v[80:85], a[240:243], v105, v107 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2 // original PC 0x2d8c
 v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[50:55], v[86:91], a[244:247], v105, v107 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:2 blgp:2 // original PC 0x2d9c
 v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[50:55], v[92:97], a[248:251], v105, v107 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2 // original PC 0x2dac
 v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[50:55], v[98:103], a[252:255], v105, v107 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:2 blgp:2 // original PC 0x2dbc
 s_add_u32 m0, s60, 0xc400 // original PC 0x2dcc
 buffer_load_dword v120, s[20:23], 0 offen lds // original PC 0x2dd4
 s_mov_b32 s33, s53 // original PC 0x2ddc
 s_mov_b32 s53, s54 // original PC 0x2de0
 s_mov_b32 s54, s55 // original PC 0x2de4
 s_mov_b32 s55, s33 // original PC 0x2de8
 s_add_u32 s35, s35, 1 // original PC 0x2dec
 s_cmp_lt_i32 s35, s36 // original PC 0x2df0
 s_cbranch_scc1 kloop // original PC 0x2df4
// Epilogue below: not in AITER's code object; residue update of each accumulator.
 s_waitcnt vmcnt(0) lgkmcnt(0)
 s_load_dwordx2 s[8:9], s[0:1], 0x180
 s_load_dwordx2 s[12:13], s[0:1], 0x188
 s_load_dwordx4 s[16:19], s[0:1], 0x190
 s_load_dwordx4 s[20:23], s[0:1], 0x1a0
 s_load_dwordx2 s[24:25], s[0:1], 0x1b0
 s_waitcnt lgkmcnt(0)
 s_mov_b32 s10, 0x80000000
 s_mov_b32 s11, 0x20000
 s_mov_b32 s14, 0x80000000
 s_mov_b32 s15, 0x20000
 s_mov_b32 s26, 0x80000000
 s_mov_b32 s27, 0x20000
 v_cvt_f64_u32 v[40:41], s17
 v_cvt_f64_u32 v[42:43], s16
 v_mov_b32_e32 v44, s18
 v_mov_b32_e32 v45, s19
 v_mov_b32_e32 v60, 0
 // v8 base row, v9 base column. v0 still holds original workitem id.
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
// group 0, AGPR 0..3, row+0, col+0
v_add_u32_e32 v10, 0, v8
 v_add_u32_e32 v11, 0, v9
 v_mul_lo_u32 v12, s21, v11
 v_add_u32_e32 v12, v12, v10
 v_lshlrev_b32_e32 v12, 1, v12
 v_mov_b32_e32 v48, 0
 v_mov_b32_e32 v49, 0
 s_cmp_eq_u32 s20, 0
 s_cbranch_scc1 no_old_0
 buffer_load_dwordx2 v[48:49], v12, s[8:11], 0 offen
no_old_0:
v_accvgpr_read_b32 v16, a0
v_accvgpr_read_b32 v17, a1
v_accvgpr_read_b32 v18, a2
v_accvgpr_read_b32 v19, a3
s_nop 1
s_or_b32 s28, s24, s25
 s_cmp_eq_u32 s28, 0
 s_cbranch_scc1 no_shadow_0
 v_mul_lo_u32 v13, s21, v10
 v_add_u32_e32 v13, v13, v11
 v_lshlrev_b32_e32 v13, 2, v13
 s_lshl_b32 s28, s21, 2
buffer_store_dword v16, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v17, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v18, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v19, v13, s[24:27], 0 offen
no_shadow_0:
v_trunc_f32_e32 v14, v16
 v_cmp_eq_f32_e32 vcc, v14, v16
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v16
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v16
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v32, v[28:29]
v_trunc_f32_e32 v14, v17
 v_cmp_eq_f32_e32 vcc, v14, v17
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v17
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v17
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v33, v[28:29]
v_trunc_f32_e32 v14, v18
 v_cmp_eq_f32_e32 vcc, v14, v18
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v18
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v18
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v34, v[28:29]
v_trunc_f32_e32 v14, v19
 v_cmp_eq_f32_e32 vcc, v14, v19
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v19
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v19
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v35, v[28:29]
s_waitcnt vmcnt(0)
v_bfe_u32 v14, v48, 0, 16
 v_add_u32_e32 v32, v32, v14
 v_sub_u32_e64 v15, v32, s16
 v_cmp_ge_u32_e64 vcc, v32, s16
 v_cndmask_b32_e32 v32, v32, v15, vcc
v_bfe_u32 v14, v48, 16, 16
 v_add_u32_e32 v33, v33, v14
 v_sub_u32_e64 v15, v33, s16
 v_cmp_ge_u32_e64 vcc, v33, s16
 v_cndmask_b32_e32 v33, v33, v15, vcc
v_bfe_u32 v14, v49, 0, 16
 v_add_u32_e32 v34, v34, v14
 v_sub_u32_e64 v15, v34, s16
 v_cmp_ge_u32_e64 vcc, v34, s16
 v_cndmask_b32_e32 v34, v34, v15, vcc
v_bfe_u32 v14, v49, 16, 16
 v_add_u32_e32 v35, v35, v14
 v_sub_u32_e64 v15, v35, s16
 v_cmp_ge_u32_e64 vcc, v35, s16
 v_cndmask_b32_e32 v35, v35, v15, vcc
v_lshlrev_b32_e32 v33, 16, v33
 v_or_b32_e32 v36, v32, v33
 v_lshlrev_b32_e32 v35, 16, v35
 v_or_b32_e32 v37, v34, v35
 buffer_store_dwordx2 v[36:37], v12, s[8:11], 0 offen
// group 1, AGPR 4..7, row+0, col+16
v_add_u32_e32 v10, 0, v8
 v_add_u32_e32 v11, 16, v9
 v_mul_lo_u32 v12, s21, v11
 v_add_u32_e32 v12, v12, v10
 v_lshlrev_b32_e32 v12, 1, v12
 v_mov_b32_e32 v48, 0
 v_mov_b32_e32 v49, 0
 s_cmp_eq_u32 s20, 0
 s_cbranch_scc1 no_old_1
 buffer_load_dwordx2 v[48:49], v12, s[8:11], 0 offen
no_old_1:
v_accvgpr_read_b32 v16, a4
v_accvgpr_read_b32 v17, a5
v_accvgpr_read_b32 v18, a6
v_accvgpr_read_b32 v19, a7
s_nop 1
s_or_b32 s28, s24, s25
 s_cmp_eq_u32 s28, 0
 s_cbranch_scc1 no_shadow_1
 v_mul_lo_u32 v13, s21, v10
 v_add_u32_e32 v13, v13, v11
 v_lshlrev_b32_e32 v13, 2, v13
 s_lshl_b32 s28, s21, 2
buffer_store_dword v16, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v17, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v18, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v19, v13, s[24:27], 0 offen
no_shadow_1:
v_trunc_f32_e32 v14, v16
 v_cmp_eq_f32_e32 vcc, v14, v16
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v16
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v16
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v32, v[28:29]
v_trunc_f32_e32 v14, v17
 v_cmp_eq_f32_e32 vcc, v14, v17
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v17
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v17
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v33, v[28:29]
v_trunc_f32_e32 v14, v18
 v_cmp_eq_f32_e32 vcc, v14, v18
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v18
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v18
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v34, v[28:29]
v_trunc_f32_e32 v14, v19
 v_cmp_eq_f32_e32 vcc, v14, v19
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v19
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v19
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v35, v[28:29]
s_waitcnt vmcnt(0)
v_bfe_u32 v14, v48, 0, 16
 v_add_u32_e32 v32, v32, v14
 v_sub_u32_e64 v15, v32, s16
 v_cmp_ge_u32_e64 vcc, v32, s16
 v_cndmask_b32_e32 v32, v32, v15, vcc
v_bfe_u32 v14, v48, 16, 16
 v_add_u32_e32 v33, v33, v14
 v_sub_u32_e64 v15, v33, s16
 v_cmp_ge_u32_e64 vcc, v33, s16
 v_cndmask_b32_e32 v33, v33, v15, vcc
v_bfe_u32 v14, v49, 0, 16
 v_add_u32_e32 v34, v34, v14
 v_sub_u32_e64 v15, v34, s16
 v_cmp_ge_u32_e64 vcc, v34, s16
 v_cndmask_b32_e32 v34, v34, v15, vcc
v_bfe_u32 v14, v49, 16, 16
 v_add_u32_e32 v35, v35, v14
 v_sub_u32_e64 v15, v35, s16
 v_cmp_ge_u32_e64 vcc, v35, s16
 v_cndmask_b32_e32 v35, v35, v15, vcc
v_lshlrev_b32_e32 v33, 16, v33
 v_or_b32_e32 v36, v32, v33
 v_lshlrev_b32_e32 v35, 16, v35
 v_or_b32_e32 v37, v34, v35
 buffer_store_dwordx2 v[36:37], v12, s[8:11], 0 offen
// group 2, AGPR 8..11, row+0, col+32
v_add_u32_e32 v10, 0, v8
 v_add_u32_e32 v11, 32, v9
 v_mul_lo_u32 v12, s21, v11
 v_add_u32_e32 v12, v12, v10
 v_lshlrev_b32_e32 v12, 1, v12
 v_mov_b32_e32 v48, 0
 v_mov_b32_e32 v49, 0
 s_cmp_eq_u32 s20, 0
 s_cbranch_scc1 no_old_2
 buffer_load_dwordx2 v[48:49], v12, s[8:11], 0 offen
no_old_2:
v_accvgpr_read_b32 v16, a8
v_accvgpr_read_b32 v17, a9
v_accvgpr_read_b32 v18, a10
v_accvgpr_read_b32 v19, a11
s_nop 1
s_or_b32 s28, s24, s25
 s_cmp_eq_u32 s28, 0
 s_cbranch_scc1 no_shadow_2
 v_mul_lo_u32 v13, s21, v10
 v_add_u32_e32 v13, v13, v11
 v_lshlrev_b32_e32 v13, 2, v13
 s_lshl_b32 s28, s21, 2
buffer_store_dword v16, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v17, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v18, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v19, v13, s[24:27], 0 offen
no_shadow_2:
v_trunc_f32_e32 v14, v16
 v_cmp_eq_f32_e32 vcc, v14, v16
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v16
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v16
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v32, v[28:29]
v_trunc_f32_e32 v14, v17
 v_cmp_eq_f32_e32 vcc, v14, v17
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v17
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v17
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v33, v[28:29]
v_trunc_f32_e32 v14, v18
 v_cmp_eq_f32_e32 vcc, v14, v18
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v18
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v18
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v34, v[28:29]
v_trunc_f32_e32 v14, v19
 v_cmp_eq_f32_e32 vcc, v14, v19
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v19
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v19
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v35, v[28:29]
s_waitcnt vmcnt(0)
v_bfe_u32 v14, v48, 0, 16
 v_add_u32_e32 v32, v32, v14
 v_sub_u32_e64 v15, v32, s16
 v_cmp_ge_u32_e64 vcc, v32, s16
 v_cndmask_b32_e32 v32, v32, v15, vcc
v_bfe_u32 v14, v48, 16, 16
 v_add_u32_e32 v33, v33, v14
 v_sub_u32_e64 v15, v33, s16
 v_cmp_ge_u32_e64 vcc, v33, s16
 v_cndmask_b32_e32 v33, v33, v15, vcc
v_bfe_u32 v14, v49, 0, 16
 v_add_u32_e32 v34, v34, v14
 v_sub_u32_e64 v15, v34, s16
 v_cmp_ge_u32_e64 vcc, v34, s16
 v_cndmask_b32_e32 v34, v34, v15, vcc
v_bfe_u32 v14, v49, 16, 16
 v_add_u32_e32 v35, v35, v14
 v_sub_u32_e64 v15, v35, s16
 v_cmp_ge_u32_e64 vcc, v35, s16
 v_cndmask_b32_e32 v35, v35, v15, vcc
v_lshlrev_b32_e32 v33, 16, v33
 v_or_b32_e32 v36, v32, v33
 v_lshlrev_b32_e32 v35, 16, v35
 v_or_b32_e32 v37, v34, v35
 buffer_store_dwordx2 v[36:37], v12, s[8:11], 0 offen
// group 3, AGPR 12..15, row+0, col+48
v_add_u32_e32 v10, 0, v8
 v_add_u32_e32 v11, 48, v9
 v_mul_lo_u32 v12, s21, v11
 v_add_u32_e32 v12, v12, v10
 v_lshlrev_b32_e32 v12, 1, v12
 v_mov_b32_e32 v48, 0
 v_mov_b32_e32 v49, 0
 s_cmp_eq_u32 s20, 0
 s_cbranch_scc1 no_old_3
 buffer_load_dwordx2 v[48:49], v12, s[8:11], 0 offen
no_old_3:
v_accvgpr_read_b32 v16, a12
v_accvgpr_read_b32 v17, a13
v_accvgpr_read_b32 v18, a14
v_accvgpr_read_b32 v19, a15
s_nop 1
s_or_b32 s28, s24, s25
 s_cmp_eq_u32 s28, 0
 s_cbranch_scc1 no_shadow_3
 v_mul_lo_u32 v13, s21, v10
 v_add_u32_e32 v13, v13, v11
 v_lshlrev_b32_e32 v13, 2, v13
 s_lshl_b32 s28, s21, 2
buffer_store_dword v16, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v17, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v18, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v19, v13, s[24:27], 0 offen
no_shadow_3:
v_trunc_f32_e32 v14, v16
 v_cmp_eq_f32_e32 vcc, v14, v16
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v16
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v16
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v32, v[28:29]
v_trunc_f32_e32 v14, v17
 v_cmp_eq_f32_e32 vcc, v14, v17
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v17
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v17
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v33, v[28:29]
v_trunc_f32_e32 v14, v18
 v_cmp_eq_f32_e32 vcc, v14, v18
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v18
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v18
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v34, v[28:29]
v_trunc_f32_e32 v14, v19
 v_cmp_eq_f32_e32 vcc, v14, v19
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v19
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v19
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v35, v[28:29]
s_waitcnt vmcnt(0)
v_bfe_u32 v14, v48, 0, 16
 v_add_u32_e32 v32, v32, v14
 v_sub_u32_e64 v15, v32, s16
 v_cmp_ge_u32_e64 vcc, v32, s16
 v_cndmask_b32_e32 v32, v32, v15, vcc
v_bfe_u32 v14, v48, 16, 16
 v_add_u32_e32 v33, v33, v14
 v_sub_u32_e64 v15, v33, s16
 v_cmp_ge_u32_e64 vcc, v33, s16
 v_cndmask_b32_e32 v33, v33, v15, vcc
v_bfe_u32 v14, v49, 0, 16
 v_add_u32_e32 v34, v34, v14
 v_sub_u32_e64 v15, v34, s16
 v_cmp_ge_u32_e64 vcc, v34, s16
 v_cndmask_b32_e32 v34, v34, v15, vcc
v_bfe_u32 v14, v49, 16, 16
 v_add_u32_e32 v35, v35, v14
 v_sub_u32_e64 v15, v35, s16
 v_cmp_ge_u32_e64 vcc, v35, s16
 v_cndmask_b32_e32 v35, v35, v15, vcc
v_lshlrev_b32_e32 v33, 16, v33
 v_or_b32_e32 v36, v32, v33
 v_lshlrev_b32_e32 v35, 16, v35
 v_or_b32_e32 v37, v34, v35
 buffer_store_dwordx2 v[36:37], v12, s[8:11], 0 offen
// group 4, AGPR 16..19, row+0, col+64
v_add_u32_e32 v10, 0, v8
 v_add_u32_e32 v11, 64, v9
 v_mul_lo_u32 v12, s21, v11
 v_add_u32_e32 v12, v12, v10
 v_lshlrev_b32_e32 v12, 1, v12
 v_mov_b32_e32 v48, 0
 v_mov_b32_e32 v49, 0
 s_cmp_eq_u32 s20, 0
 s_cbranch_scc1 no_old_4
 buffer_load_dwordx2 v[48:49], v12, s[8:11], 0 offen
no_old_4:
v_accvgpr_read_b32 v16, a16
v_accvgpr_read_b32 v17, a17
v_accvgpr_read_b32 v18, a18
v_accvgpr_read_b32 v19, a19
s_nop 1
s_or_b32 s28, s24, s25
 s_cmp_eq_u32 s28, 0
 s_cbranch_scc1 no_shadow_4
 v_mul_lo_u32 v13, s21, v10
 v_add_u32_e32 v13, v13, v11
 v_lshlrev_b32_e32 v13, 2, v13
 s_lshl_b32 s28, s21, 2
buffer_store_dword v16, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v17, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v18, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v19, v13, s[24:27], 0 offen
no_shadow_4:
v_trunc_f32_e32 v14, v16
 v_cmp_eq_f32_e32 vcc, v14, v16
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v16
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v16
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v32, v[28:29]
v_trunc_f32_e32 v14, v17
 v_cmp_eq_f32_e32 vcc, v14, v17
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v17
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v17
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v33, v[28:29]
v_trunc_f32_e32 v14, v18
 v_cmp_eq_f32_e32 vcc, v14, v18
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v18
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v18
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v34, v[28:29]
v_trunc_f32_e32 v14, v19
 v_cmp_eq_f32_e32 vcc, v14, v19
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v19
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v19
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v35, v[28:29]
s_waitcnt vmcnt(0)
v_bfe_u32 v14, v48, 0, 16
 v_add_u32_e32 v32, v32, v14
 v_sub_u32_e64 v15, v32, s16
 v_cmp_ge_u32_e64 vcc, v32, s16
 v_cndmask_b32_e32 v32, v32, v15, vcc
v_bfe_u32 v14, v48, 16, 16
 v_add_u32_e32 v33, v33, v14
 v_sub_u32_e64 v15, v33, s16
 v_cmp_ge_u32_e64 vcc, v33, s16
 v_cndmask_b32_e32 v33, v33, v15, vcc
v_bfe_u32 v14, v49, 0, 16
 v_add_u32_e32 v34, v34, v14
 v_sub_u32_e64 v15, v34, s16
 v_cmp_ge_u32_e64 vcc, v34, s16
 v_cndmask_b32_e32 v34, v34, v15, vcc
v_bfe_u32 v14, v49, 16, 16
 v_add_u32_e32 v35, v35, v14
 v_sub_u32_e64 v15, v35, s16
 v_cmp_ge_u32_e64 vcc, v35, s16
 v_cndmask_b32_e32 v35, v35, v15, vcc
v_lshlrev_b32_e32 v33, 16, v33
 v_or_b32_e32 v36, v32, v33
 v_lshlrev_b32_e32 v35, 16, v35
 v_or_b32_e32 v37, v34, v35
 buffer_store_dwordx2 v[36:37], v12, s[8:11], 0 offen
// group 5, AGPR 20..23, row+0, col+80
v_add_u32_e32 v10, 0, v8
 v_add_u32_e32 v11, 80, v9
 v_mul_lo_u32 v12, s21, v11
 v_add_u32_e32 v12, v12, v10
 v_lshlrev_b32_e32 v12, 1, v12
 v_mov_b32_e32 v48, 0
 v_mov_b32_e32 v49, 0
 s_cmp_eq_u32 s20, 0
 s_cbranch_scc1 no_old_5
 buffer_load_dwordx2 v[48:49], v12, s[8:11], 0 offen
no_old_5:
v_accvgpr_read_b32 v16, a20
v_accvgpr_read_b32 v17, a21
v_accvgpr_read_b32 v18, a22
v_accvgpr_read_b32 v19, a23
s_nop 1
s_or_b32 s28, s24, s25
 s_cmp_eq_u32 s28, 0
 s_cbranch_scc1 no_shadow_5
 v_mul_lo_u32 v13, s21, v10
 v_add_u32_e32 v13, v13, v11
 v_lshlrev_b32_e32 v13, 2, v13
 s_lshl_b32 s28, s21, 2
buffer_store_dword v16, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v17, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v18, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v19, v13, s[24:27], 0 offen
no_shadow_5:
v_trunc_f32_e32 v14, v16
 v_cmp_eq_f32_e32 vcc, v14, v16
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v16
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v16
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v32, v[28:29]
v_trunc_f32_e32 v14, v17
 v_cmp_eq_f32_e32 vcc, v14, v17
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v17
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v17
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v33, v[28:29]
v_trunc_f32_e32 v14, v18
 v_cmp_eq_f32_e32 vcc, v14, v18
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v18
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v18
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v34, v[28:29]
v_trunc_f32_e32 v14, v19
 v_cmp_eq_f32_e32 vcc, v14, v19
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v19
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v19
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v35, v[28:29]
s_waitcnt vmcnt(0)
v_bfe_u32 v14, v48, 0, 16
 v_add_u32_e32 v32, v32, v14
 v_sub_u32_e64 v15, v32, s16
 v_cmp_ge_u32_e64 vcc, v32, s16
 v_cndmask_b32_e32 v32, v32, v15, vcc
v_bfe_u32 v14, v48, 16, 16
 v_add_u32_e32 v33, v33, v14
 v_sub_u32_e64 v15, v33, s16
 v_cmp_ge_u32_e64 vcc, v33, s16
 v_cndmask_b32_e32 v33, v33, v15, vcc
v_bfe_u32 v14, v49, 0, 16
 v_add_u32_e32 v34, v34, v14
 v_sub_u32_e64 v15, v34, s16
 v_cmp_ge_u32_e64 vcc, v34, s16
 v_cndmask_b32_e32 v34, v34, v15, vcc
v_bfe_u32 v14, v49, 16, 16
 v_add_u32_e32 v35, v35, v14
 v_sub_u32_e64 v15, v35, s16
 v_cmp_ge_u32_e64 vcc, v35, s16
 v_cndmask_b32_e32 v35, v35, v15, vcc
v_lshlrev_b32_e32 v33, 16, v33
 v_or_b32_e32 v36, v32, v33
 v_lshlrev_b32_e32 v35, 16, v35
 v_or_b32_e32 v37, v34, v35
 buffer_store_dwordx2 v[36:37], v12, s[8:11], 0 offen
// group 6, AGPR 24..27, row+0, col+96
v_add_u32_e32 v10, 0, v8
 v_add_u32_e32 v11, 96, v9
 v_mul_lo_u32 v12, s21, v11
 v_add_u32_e32 v12, v12, v10
 v_lshlrev_b32_e32 v12, 1, v12
 v_mov_b32_e32 v48, 0
 v_mov_b32_e32 v49, 0
 s_cmp_eq_u32 s20, 0
 s_cbranch_scc1 no_old_6
 buffer_load_dwordx2 v[48:49], v12, s[8:11], 0 offen
no_old_6:
v_accvgpr_read_b32 v16, a24
v_accvgpr_read_b32 v17, a25
v_accvgpr_read_b32 v18, a26
v_accvgpr_read_b32 v19, a27
s_nop 1
s_or_b32 s28, s24, s25
 s_cmp_eq_u32 s28, 0
 s_cbranch_scc1 no_shadow_6
 v_mul_lo_u32 v13, s21, v10
 v_add_u32_e32 v13, v13, v11
 v_lshlrev_b32_e32 v13, 2, v13
 s_lshl_b32 s28, s21, 2
buffer_store_dword v16, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v17, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v18, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v19, v13, s[24:27], 0 offen
no_shadow_6:
v_trunc_f32_e32 v14, v16
 v_cmp_eq_f32_e32 vcc, v14, v16
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v16
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v16
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v32, v[28:29]
v_trunc_f32_e32 v14, v17
 v_cmp_eq_f32_e32 vcc, v14, v17
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v17
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v17
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v33, v[28:29]
v_trunc_f32_e32 v14, v18
 v_cmp_eq_f32_e32 vcc, v14, v18
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v18
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v18
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v34, v[28:29]
v_trunc_f32_e32 v14, v19
 v_cmp_eq_f32_e32 vcc, v14, v19
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v19
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v19
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v35, v[28:29]
s_waitcnt vmcnt(0)
v_bfe_u32 v14, v48, 0, 16
 v_add_u32_e32 v32, v32, v14
 v_sub_u32_e64 v15, v32, s16
 v_cmp_ge_u32_e64 vcc, v32, s16
 v_cndmask_b32_e32 v32, v32, v15, vcc
v_bfe_u32 v14, v48, 16, 16
 v_add_u32_e32 v33, v33, v14
 v_sub_u32_e64 v15, v33, s16
 v_cmp_ge_u32_e64 vcc, v33, s16
 v_cndmask_b32_e32 v33, v33, v15, vcc
v_bfe_u32 v14, v49, 0, 16
 v_add_u32_e32 v34, v34, v14
 v_sub_u32_e64 v15, v34, s16
 v_cmp_ge_u32_e64 vcc, v34, s16
 v_cndmask_b32_e32 v34, v34, v15, vcc
v_bfe_u32 v14, v49, 16, 16
 v_add_u32_e32 v35, v35, v14
 v_sub_u32_e64 v15, v35, s16
 v_cmp_ge_u32_e64 vcc, v35, s16
 v_cndmask_b32_e32 v35, v35, v15, vcc
v_lshlrev_b32_e32 v33, 16, v33
 v_or_b32_e32 v36, v32, v33
 v_lshlrev_b32_e32 v35, 16, v35
 v_or_b32_e32 v37, v34, v35
 buffer_store_dwordx2 v[36:37], v12, s[8:11], 0 offen
// group 7, AGPR 28..31, row+0, col+112
v_add_u32_e32 v10, 0, v8
 v_add_u32_e32 v11, 112, v9
 v_mul_lo_u32 v12, s21, v11
 v_add_u32_e32 v12, v12, v10
 v_lshlrev_b32_e32 v12, 1, v12
 v_mov_b32_e32 v48, 0
 v_mov_b32_e32 v49, 0
 s_cmp_eq_u32 s20, 0
 s_cbranch_scc1 no_old_7
 buffer_load_dwordx2 v[48:49], v12, s[8:11], 0 offen
no_old_7:
v_accvgpr_read_b32 v16, a28
v_accvgpr_read_b32 v17, a29
v_accvgpr_read_b32 v18, a30
v_accvgpr_read_b32 v19, a31
s_nop 1
s_or_b32 s28, s24, s25
 s_cmp_eq_u32 s28, 0
 s_cbranch_scc1 no_shadow_7
 v_mul_lo_u32 v13, s21, v10
 v_add_u32_e32 v13, v13, v11
 v_lshlrev_b32_e32 v13, 2, v13
 s_lshl_b32 s28, s21, 2
buffer_store_dword v16, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v17, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v18, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v19, v13, s[24:27], 0 offen
no_shadow_7:
v_trunc_f32_e32 v14, v16
 v_cmp_eq_f32_e32 vcc, v14, v16
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v16
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v16
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v32, v[28:29]
v_trunc_f32_e32 v14, v17
 v_cmp_eq_f32_e32 vcc, v14, v17
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v17
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v17
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v33, v[28:29]
v_trunc_f32_e32 v14, v18
 v_cmp_eq_f32_e32 vcc, v14, v18
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v18
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v18
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v34, v[28:29]
v_trunc_f32_e32 v14, v19
 v_cmp_eq_f32_e32 vcc, v14, v19
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v19
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v19
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v35, v[28:29]
s_waitcnt vmcnt(0)
v_bfe_u32 v14, v48, 0, 16
 v_add_u32_e32 v32, v32, v14
 v_sub_u32_e64 v15, v32, s16
 v_cmp_ge_u32_e64 vcc, v32, s16
 v_cndmask_b32_e32 v32, v32, v15, vcc
v_bfe_u32 v14, v48, 16, 16
 v_add_u32_e32 v33, v33, v14
 v_sub_u32_e64 v15, v33, s16
 v_cmp_ge_u32_e64 vcc, v33, s16
 v_cndmask_b32_e32 v33, v33, v15, vcc
v_bfe_u32 v14, v49, 0, 16
 v_add_u32_e32 v34, v34, v14
 v_sub_u32_e64 v15, v34, s16
 v_cmp_ge_u32_e64 vcc, v34, s16
 v_cndmask_b32_e32 v34, v34, v15, vcc
v_bfe_u32 v14, v49, 16, 16
 v_add_u32_e32 v35, v35, v14
 v_sub_u32_e64 v15, v35, s16
 v_cmp_ge_u32_e64 vcc, v35, s16
 v_cndmask_b32_e32 v35, v35, v15, vcc
v_lshlrev_b32_e32 v33, 16, v33
 v_or_b32_e32 v36, v32, v33
 v_lshlrev_b32_e32 v35, 16, v35
 v_or_b32_e32 v37, v34, v35
 buffer_store_dwordx2 v[36:37], v12, s[8:11], 0 offen
// group 8, AGPR 32..35, row+16, col+0
v_add_u32_e32 v10, 16, v8
 v_add_u32_e32 v11, 0, v9
 v_mul_lo_u32 v12, s21, v11
 v_add_u32_e32 v12, v12, v10
 v_lshlrev_b32_e32 v12, 1, v12
 v_mov_b32_e32 v48, 0
 v_mov_b32_e32 v49, 0
 s_cmp_eq_u32 s20, 0
 s_cbranch_scc1 no_old_8
 buffer_load_dwordx2 v[48:49], v12, s[8:11], 0 offen
no_old_8:
v_accvgpr_read_b32 v16, a32
v_accvgpr_read_b32 v17, a33
v_accvgpr_read_b32 v18, a34
v_accvgpr_read_b32 v19, a35
s_nop 1
s_or_b32 s28, s24, s25
 s_cmp_eq_u32 s28, 0
 s_cbranch_scc1 no_shadow_8
 v_mul_lo_u32 v13, s21, v10
 v_add_u32_e32 v13, v13, v11
 v_lshlrev_b32_e32 v13, 2, v13
 s_lshl_b32 s28, s21, 2
buffer_store_dword v16, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v17, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v18, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v19, v13, s[24:27], 0 offen
no_shadow_8:
v_trunc_f32_e32 v14, v16
 v_cmp_eq_f32_e32 vcc, v14, v16
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v16
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v16
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v32, v[28:29]
v_trunc_f32_e32 v14, v17
 v_cmp_eq_f32_e32 vcc, v14, v17
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v17
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v17
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v33, v[28:29]
v_trunc_f32_e32 v14, v18
 v_cmp_eq_f32_e32 vcc, v14, v18
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v18
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v18
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v34, v[28:29]
v_trunc_f32_e32 v14, v19
 v_cmp_eq_f32_e32 vcc, v14, v19
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v19
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v19
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v35, v[28:29]
s_waitcnt vmcnt(0)
v_bfe_u32 v14, v48, 0, 16
 v_add_u32_e32 v32, v32, v14
 v_sub_u32_e64 v15, v32, s16
 v_cmp_ge_u32_e64 vcc, v32, s16
 v_cndmask_b32_e32 v32, v32, v15, vcc
v_bfe_u32 v14, v48, 16, 16
 v_add_u32_e32 v33, v33, v14
 v_sub_u32_e64 v15, v33, s16
 v_cmp_ge_u32_e64 vcc, v33, s16
 v_cndmask_b32_e32 v33, v33, v15, vcc
v_bfe_u32 v14, v49, 0, 16
 v_add_u32_e32 v34, v34, v14
 v_sub_u32_e64 v15, v34, s16
 v_cmp_ge_u32_e64 vcc, v34, s16
 v_cndmask_b32_e32 v34, v34, v15, vcc
v_bfe_u32 v14, v49, 16, 16
 v_add_u32_e32 v35, v35, v14
 v_sub_u32_e64 v15, v35, s16
 v_cmp_ge_u32_e64 vcc, v35, s16
 v_cndmask_b32_e32 v35, v35, v15, vcc
v_lshlrev_b32_e32 v33, 16, v33
 v_or_b32_e32 v36, v32, v33
 v_lshlrev_b32_e32 v35, 16, v35
 v_or_b32_e32 v37, v34, v35
 buffer_store_dwordx2 v[36:37], v12, s[8:11], 0 offen
// group 9, AGPR 36..39, row+16, col+16
v_add_u32_e32 v10, 16, v8
 v_add_u32_e32 v11, 16, v9
 v_mul_lo_u32 v12, s21, v11
 v_add_u32_e32 v12, v12, v10
 v_lshlrev_b32_e32 v12, 1, v12
 v_mov_b32_e32 v48, 0
 v_mov_b32_e32 v49, 0
 s_cmp_eq_u32 s20, 0
 s_cbranch_scc1 no_old_9
 buffer_load_dwordx2 v[48:49], v12, s[8:11], 0 offen
no_old_9:
v_accvgpr_read_b32 v16, a36
v_accvgpr_read_b32 v17, a37
v_accvgpr_read_b32 v18, a38
v_accvgpr_read_b32 v19, a39
s_nop 1
s_or_b32 s28, s24, s25
 s_cmp_eq_u32 s28, 0
 s_cbranch_scc1 no_shadow_9
 v_mul_lo_u32 v13, s21, v10
 v_add_u32_e32 v13, v13, v11
 v_lshlrev_b32_e32 v13, 2, v13
 s_lshl_b32 s28, s21, 2
buffer_store_dword v16, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v17, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v18, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v19, v13, s[24:27], 0 offen
no_shadow_9:
v_trunc_f32_e32 v14, v16
 v_cmp_eq_f32_e32 vcc, v14, v16
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v16
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v16
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v32, v[28:29]
v_trunc_f32_e32 v14, v17
 v_cmp_eq_f32_e32 vcc, v14, v17
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v17
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v17
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v33, v[28:29]
v_trunc_f32_e32 v14, v18
 v_cmp_eq_f32_e32 vcc, v14, v18
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v18
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v18
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v34, v[28:29]
v_trunc_f32_e32 v14, v19
 v_cmp_eq_f32_e32 vcc, v14, v19
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v19
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v19
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v35, v[28:29]
s_waitcnt vmcnt(0)
v_bfe_u32 v14, v48, 0, 16
 v_add_u32_e32 v32, v32, v14
 v_sub_u32_e64 v15, v32, s16
 v_cmp_ge_u32_e64 vcc, v32, s16
 v_cndmask_b32_e32 v32, v32, v15, vcc
v_bfe_u32 v14, v48, 16, 16
 v_add_u32_e32 v33, v33, v14
 v_sub_u32_e64 v15, v33, s16
 v_cmp_ge_u32_e64 vcc, v33, s16
 v_cndmask_b32_e32 v33, v33, v15, vcc
v_bfe_u32 v14, v49, 0, 16
 v_add_u32_e32 v34, v34, v14
 v_sub_u32_e64 v15, v34, s16
 v_cmp_ge_u32_e64 vcc, v34, s16
 v_cndmask_b32_e32 v34, v34, v15, vcc
v_bfe_u32 v14, v49, 16, 16
 v_add_u32_e32 v35, v35, v14
 v_sub_u32_e64 v15, v35, s16
 v_cmp_ge_u32_e64 vcc, v35, s16
 v_cndmask_b32_e32 v35, v35, v15, vcc
v_lshlrev_b32_e32 v33, 16, v33
 v_or_b32_e32 v36, v32, v33
 v_lshlrev_b32_e32 v35, 16, v35
 v_or_b32_e32 v37, v34, v35
 buffer_store_dwordx2 v[36:37], v12, s[8:11], 0 offen
// group 10, AGPR 40..43, row+16, col+32
v_add_u32_e32 v10, 16, v8
 v_add_u32_e32 v11, 32, v9
 v_mul_lo_u32 v12, s21, v11
 v_add_u32_e32 v12, v12, v10
 v_lshlrev_b32_e32 v12, 1, v12
 v_mov_b32_e32 v48, 0
 v_mov_b32_e32 v49, 0
 s_cmp_eq_u32 s20, 0
 s_cbranch_scc1 no_old_10
 buffer_load_dwordx2 v[48:49], v12, s[8:11], 0 offen
no_old_10:
v_accvgpr_read_b32 v16, a40
v_accvgpr_read_b32 v17, a41
v_accvgpr_read_b32 v18, a42
v_accvgpr_read_b32 v19, a43
s_nop 1
s_or_b32 s28, s24, s25
 s_cmp_eq_u32 s28, 0
 s_cbranch_scc1 no_shadow_10
 v_mul_lo_u32 v13, s21, v10
 v_add_u32_e32 v13, v13, v11
 v_lshlrev_b32_e32 v13, 2, v13
 s_lshl_b32 s28, s21, 2
buffer_store_dword v16, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v17, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v18, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v19, v13, s[24:27], 0 offen
no_shadow_10:
v_trunc_f32_e32 v14, v16
 v_cmp_eq_f32_e32 vcc, v14, v16
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v16
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v16
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v32, v[28:29]
v_trunc_f32_e32 v14, v17
 v_cmp_eq_f32_e32 vcc, v14, v17
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v17
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v17
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v33, v[28:29]
v_trunc_f32_e32 v14, v18
 v_cmp_eq_f32_e32 vcc, v14, v18
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v18
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v18
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v34, v[28:29]
v_trunc_f32_e32 v14, v19
 v_cmp_eq_f32_e32 vcc, v14, v19
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v19
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v19
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v35, v[28:29]
s_waitcnt vmcnt(0)
v_bfe_u32 v14, v48, 0, 16
 v_add_u32_e32 v32, v32, v14
 v_sub_u32_e64 v15, v32, s16
 v_cmp_ge_u32_e64 vcc, v32, s16
 v_cndmask_b32_e32 v32, v32, v15, vcc
v_bfe_u32 v14, v48, 16, 16
 v_add_u32_e32 v33, v33, v14
 v_sub_u32_e64 v15, v33, s16
 v_cmp_ge_u32_e64 vcc, v33, s16
 v_cndmask_b32_e32 v33, v33, v15, vcc
v_bfe_u32 v14, v49, 0, 16
 v_add_u32_e32 v34, v34, v14
 v_sub_u32_e64 v15, v34, s16
 v_cmp_ge_u32_e64 vcc, v34, s16
 v_cndmask_b32_e32 v34, v34, v15, vcc
v_bfe_u32 v14, v49, 16, 16
 v_add_u32_e32 v35, v35, v14
 v_sub_u32_e64 v15, v35, s16
 v_cmp_ge_u32_e64 vcc, v35, s16
 v_cndmask_b32_e32 v35, v35, v15, vcc
v_lshlrev_b32_e32 v33, 16, v33
 v_or_b32_e32 v36, v32, v33
 v_lshlrev_b32_e32 v35, 16, v35
 v_or_b32_e32 v37, v34, v35
 buffer_store_dwordx2 v[36:37], v12, s[8:11], 0 offen
// group 11, AGPR 44..47, row+16, col+48
v_add_u32_e32 v10, 16, v8
 v_add_u32_e32 v11, 48, v9
 v_mul_lo_u32 v12, s21, v11
 v_add_u32_e32 v12, v12, v10
 v_lshlrev_b32_e32 v12, 1, v12
 v_mov_b32_e32 v48, 0
 v_mov_b32_e32 v49, 0
 s_cmp_eq_u32 s20, 0
 s_cbranch_scc1 no_old_11
 buffer_load_dwordx2 v[48:49], v12, s[8:11], 0 offen
no_old_11:
v_accvgpr_read_b32 v16, a44
v_accvgpr_read_b32 v17, a45
v_accvgpr_read_b32 v18, a46
v_accvgpr_read_b32 v19, a47
s_nop 1
s_or_b32 s28, s24, s25
 s_cmp_eq_u32 s28, 0
 s_cbranch_scc1 no_shadow_11
 v_mul_lo_u32 v13, s21, v10
 v_add_u32_e32 v13, v13, v11
 v_lshlrev_b32_e32 v13, 2, v13
 s_lshl_b32 s28, s21, 2
buffer_store_dword v16, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v17, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v18, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v19, v13, s[24:27], 0 offen
no_shadow_11:
v_trunc_f32_e32 v14, v16
 v_cmp_eq_f32_e32 vcc, v14, v16
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v16
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v16
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v32, v[28:29]
v_trunc_f32_e32 v14, v17
 v_cmp_eq_f32_e32 vcc, v14, v17
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v17
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v17
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v33, v[28:29]
v_trunc_f32_e32 v14, v18
 v_cmp_eq_f32_e32 vcc, v14, v18
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v18
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v18
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v34, v[28:29]
v_trunc_f32_e32 v14, v19
 v_cmp_eq_f32_e32 vcc, v14, v19
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v19
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v19
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v35, v[28:29]
s_waitcnt vmcnt(0)
v_bfe_u32 v14, v48, 0, 16
 v_add_u32_e32 v32, v32, v14
 v_sub_u32_e64 v15, v32, s16
 v_cmp_ge_u32_e64 vcc, v32, s16
 v_cndmask_b32_e32 v32, v32, v15, vcc
v_bfe_u32 v14, v48, 16, 16
 v_add_u32_e32 v33, v33, v14
 v_sub_u32_e64 v15, v33, s16
 v_cmp_ge_u32_e64 vcc, v33, s16
 v_cndmask_b32_e32 v33, v33, v15, vcc
v_bfe_u32 v14, v49, 0, 16
 v_add_u32_e32 v34, v34, v14
 v_sub_u32_e64 v15, v34, s16
 v_cmp_ge_u32_e64 vcc, v34, s16
 v_cndmask_b32_e32 v34, v34, v15, vcc
v_bfe_u32 v14, v49, 16, 16
 v_add_u32_e32 v35, v35, v14
 v_sub_u32_e64 v15, v35, s16
 v_cmp_ge_u32_e64 vcc, v35, s16
 v_cndmask_b32_e32 v35, v35, v15, vcc
v_lshlrev_b32_e32 v33, 16, v33
 v_or_b32_e32 v36, v32, v33
 v_lshlrev_b32_e32 v35, 16, v35
 v_or_b32_e32 v37, v34, v35
 buffer_store_dwordx2 v[36:37], v12, s[8:11], 0 offen
// group 12, AGPR 48..51, row+16, col+64
v_add_u32_e32 v10, 16, v8
 v_add_u32_e32 v11, 64, v9
 v_mul_lo_u32 v12, s21, v11
 v_add_u32_e32 v12, v12, v10
 v_lshlrev_b32_e32 v12, 1, v12
 v_mov_b32_e32 v48, 0
 v_mov_b32_e32 v49, 0
 s_cmp_eq_u32 s20, 0
 s_cbranch_scc1 no_old_12
 buffer_load_dwordx2 v[48:49], v12, s[8:11], 0 offen
no_old_12:
v_accvgpr_read_b32 v16, a48
v_accvgpr_read_b32 v17, a49
v_accvgpr_read_b32 v18, a50
v_accvgpr_read_b32 v19, a51
s_nop 1
s_or_b32 s28, s24, s25
 s_cmp_eq_u32 s28, 0
 s_cbranch_scc1 no_shadow_12
 v_mul_lo_u32 v13, s21, v10
 v_add_u32_e32 v13, v13, v11
 v_lshlrev_b32_e32 v13, 2, v13
 s_lshl_b32 s28, s21, 2
buffer_store_dword v16, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v17, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v18, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v19, v13, s[24:27], 0 offen
no_shadow_12:
v_trunc_f32_e32 v14, v16
 v_cmp_eq_f32_e32 vcc, v14, v16
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v16
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v16
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v32, v[28:29]
v_trunc_f32_e32 v14, v17
 v_cmp_eq_f32_e32 vcc, v14, v17
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v17
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v17
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v33, v[28:29]
v_trunc_f32_e32 v14, v18
 v_cmp_eq_f32_e32 vcc, v14, v18
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v18
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v18
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v34, v[28:29]
v_trunc_f32_e32 v14, v19
 v_cmp_eq_f32_e32 vcc, v14, v19
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v19
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v19
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v35, v[28:29]
s_waitcnt vmcnt(0)
v_bfe_u32 v14, v48, 0, 16
 v_add_u32_e32 v32, v32, v14
 v_sub_u32_e64 v15, v32, s16
 v_cmp_ge_u32_e64 vcc, v32, s16
 v_cndmask_b32_e32 v32, v32, v15, vcc
v_bfe_u32 v14, v48, 16, 16
 v_add_u32_e32 v33, v33, v14
 v_sub_u32_e64 v15, v33, s16
 v_cmp_ge_u32_e64 vcc, v33, s16
 v_cndmask_b32_e32 v33, v33, v15, vcc
v_bfe_u32 v14, v49, 0, 16
 v_add_u32_e32 v34, v34, v14
 v_sub_u32_e64 v15, v34, s16
 v_cmp_ge_u32_e64 vcc, v34, s16
 v_cndmask_b32_e32 v34, v34, v15, vcc
v_bfe_u32 v14, v49, 16, 16
 v_add_u32_e32 v35, v35, v14
 v_sub_u32_e64 v15, v35, s16
 v_cmp_ge_u32_e64 vcc, v35, s16
 v_cndmask_b32_e32 v35, v35, v15, vcc
v_lshlrev_b32_e32 v33, 16, v33
 v_or_b32_e32 v36, v32, v33
 v_lshlrev_b32_e32 v35, 16, v35
 v_or_b32_e32 v37, v34, v35
 buffer_store_dwordx2 v[36:37], v12, s[8:11], 0 offen
// group 13, AGPR 52..55, row+16, col+80
v_add_u32_e32 v10, 16, v8
 v_add_u32_e32 v11, 80, v9
 v_mul_lo_u32 v12, s21, v11
 v_add_u32_e32 v12, v12, v10
 v_lshlrev_b32_e32 v12, 1, v12
 v_mov_b32_e32 v48, 0
 v_mov_b32_e32 v49, 0
 s_cmp_eq_u32 s20, 0
 s_cbranch_scc1 no_old_13
 buffer_load_dwordx2 v[48:49], v12, s[8:11], 0 offen
no_old_13:
v_accvgpr_read_b32 v16, a52
v_accvgpr_read_b32 v17, a53
v_accvgpr_read_b32 v18, a54
v_accvgpr_read_b32 v19, a55
s_nop 1
s_or_b32 s28, s24, s25
 s_cmp_eq_u32 s28, 0
 s_cbranch_scc1 no_shadow_13
 v_mul_lo_u32 v13, s21, v10
 v_add_u32_e32 v13, v13, v11
 v_lshlrev_b32_e32 v13, 2, v13
 s_lshl_b32 s28, s21, 2
buffer_store_dword v16, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v17, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v18, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v19, v13, s[24:27], 0 offen
no_shadow_13:
v_trunc_f32_e32 v14, v16
 v_cmp_eq_f32_e32 vcc, v14, v16
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v16
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v16
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v32, v[28:29]
v_trunc_f32_e32 v14, v17
 v_cmp_eq_f32_e32 vcc, v14, v17
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v17
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v17
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v33, v[28:29]
v_trunc_f32_e32 v14, v18
 v_cmp_eq_f32_e32 vcc, v14, v18
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v18
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v18
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v34, v[28:29]
v_trunc_f32_e32 v14, v19
 v_cmp_eq_f32_e32 vcc, v14, v19
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v19
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v19
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v35, v[28:29]
s_waitcnt vmcnt(0)
v_bfe_u32 v14, v48, 0, 16
 v_add_u32_e32 v32, v32, v14
 v_sub_u32_e64 v15, v32, s16
 v_cmp_ge_u32_e64 vcc, v32, s16
 v_cndmask_b32_e32 v32, v32, v15, vcc
v_bfe_u32 v14, v48, 16, 16
 v_add_u32_e32 v33, v33, v14
 v_sub_u32_e64 v15, v33, s16
 v_cmp_ge_u32_e64 vcc, v33, s16
 v_cndmask_b32_e32 v33, v33, v15, vcc
v_bfe_u32 v14, v49, 0, 16
 v_add_u32_e32 v34, v34, v14
 v_sub_u32_e64 v15, v34, s16
 v_cmp_ge_u32_e64 vcc, v34, s16
 v_cndmask_b32_e32 v34, v34, v15, vcc
v_bfe_u32 v14, v49, 16, 16
 v_add_u32_e32 v35, v35, v14
 v_sub_u32_e64 v15, v35, s16
 v_cmp_ge_u32_e64 vcc, v35, s16
 v_cndmask_b32_e32 v35, v35, v15, vcc
v_lshlrev_b32_e32 v33, 16, v33
 v_or_b32_e32 v36, v32, v33
 v_lshlrev_b32_e32 v35, 16, v35
 v_or_b32_e32 v37, v34, v35
 buffer_store_dwordx2 v[36:37], v12, s[8:11], 0 offen
// group 14, AGPR 56..59, row+16, col+96
v_add_u32_e32 v10, 16, v8
 v_add_u32_e32 v11, 96, v9
 v_mul_lo_u32 v12, s21, v11
 v_add_u32_e32 v12, v12, v10
 v_lshlrev_b32_e32 v12, 1, v12
 v_mov_b32_e32 v48, 0
 v_mov_b32_e32 v49, 0
 s_cmp_eq_u32 s20, 0
 s_cbranch_scc1 no_old_14
 buffer_load_dwordx2 v[48:49], v12, s[8:11], 0 offen
no_old_14:
v_accvgpr_read_b32 v16, a56
v_accvgpr_read_b32 v17, a57
v_accvgpr_read_b32 v18, a58
v_accvgpr_read_b32 v19, a59
s_nop 1
s_or_b32 s28, s24, s25
 s_cmp_eq_u32 s28, 0
 s_cbranch_scc1 no_shadow_14
 v_mul_lo_u32 v13, s21, v10
 v_add_u32_e32 v13, v13, v11
 v_lshlrev_b32_e32 v13, 2, v13
 s_lshl_b32 s28, s21, 2
buffer_store_dword v16, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v17, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v18, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v19, v13, s[24:27], 0 offen
no_shadow_14:
v_trunc_f32_e32 v14, v16
 v_cmp_eq_f32_e32 vcc, v14, v16
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v16
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v16
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v32, v[28:29]
v_trunc_f32_e32 v14, v17
 v_cmp_eq_f32_e32 vcc, v14, v17
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v17
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v17
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v33, v[28:29]
v_trunc_f32_e32 v14, v18
 v_cmp_eq_f32_e32 vcc, v14, v18
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v18
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v18
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v34, v[28:29]
v_trunc_f32_e32 v14, v19
 v_cmp_eq_f32_e32 vcc, v14, v19
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v19
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v19
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v35, v[28:29]
s_waitcnt vmcnt(0)
v_bfe_u32 v14, v48, 0, 16
 v_add_u32_e32 v32, v32, v14
 v_sub_u32_e64 v15, v32, s16
 v_cmp_ge_u32_e64 vcc, v32, s16
 v_cndmask_b32_e32 v32, v32, v15, vcc
v_bfe_u32 v14, v48, 16, 16
 v_add_u32_e32 v33, v33, v14
 v_sub_u32_e64 v15, v33, s16
 v_cmp_ge_u32_e64 vcc, v33, s16
 v_cndmask_b32_e32 v33, v33, v15, vcc
v_bfe_u32 v14, v49, 0, 16
 v_add_u32_e32 v34, v34, v14
 v_sub_u32_e64 v15, v34, s16
 v_cmp_ge_u32_e64 vcc, v34, s16
 v_cndmask_b32_e32 v34, v34, v15, vcc
v_bfe_u32 v14, v49, 16, 16
 v_add_u32_e32 v35, v35, v14
 v_sub_u32_e64 v15, v35, s16
 v_cmp_ge_u32_e64 vcc, v35, s16
 v_cndmask_b32_e32 v35, v35, v15, vcc
v_lshlrev_b32_e32 v33, 16, v33
 v_or_b32_e32 v36, v32, v33
 v_lshlrev_b32_e32 v35, 16, v35
 v_or_b32_e32 v37, v34, v35
 buffer_store_dwordx2 v[36:37], v12, s[8:11], 0 offen
// group 15, AGPR 60..63, row+16, col+112
v_add_u32_e32 v10, 16, v8
 v_add_u32_e32 v11, 112, v9
 v_mul_lo_u32 v12, s21, v11
 v_add_u32_e32 v12, v12, v10
 v_lshlrev_b32_e32 v12, 1, v12
 v_mov_b32_e32 v48, 0
 v_mov_b32_e32 v49, 0
 s_cmp_eq_u32 s20, 0
 s_cbranch_scc1 no_old_15
 buffer_load_dwordx2 v[48:49], v12, s[8:11], 0 offen
no_old_15:
v_accvgpr_read_b32 v16, a60
v_accvgpr_read_b32 v17, a61
v_accvgpr_read_b32 v18, a62
v_accvgpr_read_b32 v19, a63
s_nop 1
s_or_b32 s28, s24, s25
 s_cmp_eq_u32 s28, 0
 s_cbranch_scc1 no_shadow_15
 v_mul_lo_u32 v13, s21, v10
 v_add_u32_e32 v13, v13, v11
 v_lshlrev_b32_e32 v13, 2, v13
 s_lshl_b32 s28, s21, 2
buffer_store_dword v16, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v17, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v18, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v19, v13, s[24:27], 0 offen
no_shadow_15:
v_trunc_f32_e32 v14, v16
 v_cmp_eq_f32_e32 vcc, v14, v16
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v16
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v16
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v32, v[28:29]
v_trunc_f32_e32 v14, v17
 v_cmp_eq_f32_e32 vcc, v14, v17
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v17
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v17
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v33, v[28:29]
v_trunc_f32_e32 v14, v18
 v_cmp_eq_f32_e32 vcc, v14, v18
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v18
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v18
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v34, v[28:29]
v_trunc_f32_e32 v14, v19
 v_cmp_eq_f32_e32 vcc, v14, v19
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v19
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v19
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v35, v[28:29]
s_waitcnt vmcnt(0)
v_bfe_u32 v14, v48, 0, 16
 v_add_u32_e32 v32, v32, v14
 v_sub_u32_e64 v15, v32, s16
 v_cmp_ge_u32_e64 vcc, v32, s16
 v_cndmask_b32_e32 v32, v32, v15, vcc
v_bfe_u32 v14, v48, 16, 16
 v_add_u32_e32 v33, v33, v14
 v_sub_u32_e64 v15, v33, s16
 v_cmp_ge_u32_e64 vcc, v33, s16
 v_cndmask_b32_e32 v33, v33, v15, vcc
v_bfe_u32 v14, v49, 0, 16
 v_add_u32_e32 v34, v34, v14
 v_sub_u32_e64 v15, v34, s16
 v_cmp_ge_u32_e64 vcc, v34, s16
 v_cndmask_b32_e32 v34, v34, v15, vcc
v_bfe_u32 v14, v49, 16, 16
 v_add_u32_e32 v35, v35, v14
 v_sub_u32_e64 v15, v35, s16
 v_cmp_ge_u32_e64 vcc, v35, s16
 v_cndmask_b32_e32 v35, v35, v15, vcc
v_lshlrev_b32_e32 v33, 16, v33
 v_or_b32_e32 v36, v32, v33
 v_lshlrev_b32_e32 v35, 16, v35
 v_or_b32_e32 v37, v34, v35
 buffer_store_dwordx2 v[36:37], v12, s[8:11], 0 offen
// group 16, AGPR 64..67, row+32, col+0
v_add_u32_e32 v10, 32, v8
 v_add_u32_e32 v11, 0, v9
 v_mul_lo_u32 v12, s21, v11
 v_add_u32_e32 v12, v12, v10
 v_lshlrev_b32_e32 v12, 1, v12
 v_mov_b32_e32 v48, 0
 v_mov_b32_e32 v49, 0
 s_cmp_eq_u32 s20, 0
 s_cbranch_scc1 no_old_16
 buffer_load_dwordx2 v[48:49], v12, s[8:11], 0 offen
no_old_16:
v_accvgpr_read_b32 v16, a64
v_accvgpr_read_b32 v17, a65
v_accvgpr_read_b32 v18, a66
v_accvgpr_read_b32 v19, a67
s_nop 1
s_or_b32 s28, s24, s25
 s_cmp_eq_u32 s28, 0
 s_cbranch_scc1 no_shadow_16
 v_mul_lo_u32 v13, s21, v10
 v_add_u32_e32 v13, v13, v11
 v_lshlrev_b32_e32 v13, 2, v13
 s_lshl_b32 s28, s21, 2
buffer_store_dword v16, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v17, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v18, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v19, v13, s[24:27], 0 offen
no_shadow_16:
v_trunc_f32_e32 v14, v16
 v_cmp_eq_f32_e32 vcc, v14, v16
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v16
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v16
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v32, v[28:29]
v_trunc_f32_e32 v14, v17
 v_cmp_eq_f32_e32 vcc, v14, v17
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v17
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v17
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v33, v[28:29]
v_trunc_f32_e32 v14, v18
 v_cmp_eq_f32_e32 vcc, v14, v18
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v18
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v18
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v34, v[28:29]
v_trunc_f32_e32 v14, v19
 v_cmp_eq_f32_e32 vcc, v14, v19
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v19
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v19
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v35, v[28:29]
s_waitcnt vmcnt(0)
v_bfe_u32 v14, v48, 0, 16
 v_add_u32_e32 v32, v32, v14
 v_sub_u32_e64 v15, v32, s16
 v_cmp_ge_u32_e64 vcc, v32, s16
 v_cndmask_b32_e32 v32, v32, v15, vcc
v_bfe_u32 v14, v48, 16, 16
 v_add_u32_e32 v33, v33, v14
 v_sub_u32_e64 v15, v33, s16
 v_cmp_ge_u32_e64 vcc, v33, s16
 v_cndmask_b32_e32 v33, v33, v15, vcc
v_bfe_u32 v14, v49, 0, 16
 v_add_u32_e32 v34, v34, v14
 v_sub_u32_e64 v15, v34, s16
 v_cmp_ge_u32_e64 vcc, v34, s16
 v_cndmask_b32_e32 v34, v34, v15, vcc
v_bfe_u32 v14, v49, 16, 16
 v_add_u32_e32 v35, v35, v14
 v_sub_u32_e64 v15, v35, s16
 v_cmp_ge_u32_e64 vcc, v35, s16
 v_cndmask_b32_e32 v35, v35, v15, vcc
v_lshlrev_b32_e32 v33, 16, v33
 v_or_b32_e32 v36, v32, v33
 v_lshlrev_b32_e32 v35, 16, v35
 v_or_b32_e32 v37, v34, v35
 buffer_store_dwordx2 v[36:37], v12, s[8:11], 0 offen
// group 17, AGPR 68..71, row+32, col+16
v_add_u32_e32 v10, 32, v8
 v_add_u32_e32 v11, 16, v9
 v_mul_lo_u32 v12, s21, v11
 v_add_u32_e32 v12, v12, v10
 v_lshlrev_b32_e32 v12, 1, v12
 v_mov_b32_e32 v48, 0
 v_mov_b32_e32 v49, 0
 s_cmp_eq_u32 s20, 0
 s_cbranch_scc1 no_old_17
 buffer_load_dwordx2 v[48:49], v12, s[8:11], 0 offen
no_old_17:
v_accvgpr_read_b32 v16, a68
v_accvgpr_read_b32 v17, a69
v_accvgpr_read_b32 v18, a70
v_accvgpr_read_b32 v19, a71
s_nop 1
s_or_b32 s28, s24, s25
 s_cmp_eq_u32 s28, 0
 s_cbranch_scc1 no_shadow_17
 v_mul_lo_u32 v13, s21, v10
 v_add_u32_e32 v13, v13, v11
 v_lshlrev_b32_e32 v13, 2, v13
 s_lshl_b32 s28, s21, 2
buffer_store_dword v16, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v17, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v18, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v19, v13, s[24:27], 0 offen
no_shadow_17:
v_trunc_f32_e32 v14, v16
 v_cmp_eq_f32_e32 vcc, v14, v16
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v16
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v16
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v32, v[28:29]
v_trunc_f32_e32 v14, v17
 v_cmp_eq_f32_e32 vcc, v14, v17
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v17
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v17
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v33, v[28:29]
v_trunc_f32_e32 v14, v18
 v_cmp_eq_f32_e32 vcc, v14, v18
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v18
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v18
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v34, v[28:29]
v_trunc_f32_e32 v14, v19
 v_cmp_eq_f32_e32 vcc, v14, v19
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v19
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v19
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v35, v[28:29]
s_waitcnt vmcnt(0)
v_bfe_u32 v14, v48, 0, 16
 v_add_u32_e32 v32, v32, v14
 v_sub_u32_e64 v15, v32, s16
 v_cmp_ge_u32_e64 vcc, v32, s16
 v_cndmask_b32_e32 v32, v32, v15, vcc
v_bfe_u32 v14, v48, 16, 16
 v_add_u32_e32 v33, v33, v14
 v_sub_u32_e64 v15, v33, s16
 v_cmp_ge_u32_e64 vcc, v33, s16
 v_cndmask_b32_e32 v33, v33, v15, vcc
v_bfe_u32 v14, v49, 0, 16
 v_add_u32_e32 v34, v34, v14
 v_sub_u32_e64 v15, v34, s16
 v_cmp_ge_u32_e64 vcc, v34, s16
 v_cndmask_b32_e32 v34, v34, v15, vcc
v_bfe_u32 v14, v49, 16, 16
 v_add_u32_e32 v35, v35, v14
 v_sub_u32_e64 v15, v35, s16
 v_cmp_ge_u32_e64 vcc, v35, s16
 v_cndmask_b32_e32 v35, v35, v15, vcc
v_lshlrev_b32_e32 v33, 16, v33
 v_or_b32_e32 v36, v32, v33
 v_lshlrev_b32_e32 v35, 16, v35
 v_or_b32_e32 v37, v34, v35
 buffer_store_dwordx2 v[36:37], v12, s[8:11], 0 offen
// group 18, AGPR 72..75, row+32, col+32
v_add_u32_e32 v10, 32, v8
 v_add_u32_e32 v11, 32, v9
 v_mul_lo_u32 v12, s21, v11
 v_add_u32_e32 v12, v12, v10
 v_lshlrev_b32_e32 v12, 1, v12
 v_mov_b32_e32 v48, 0
 v_mov_b32_e32 v49, 0
 s_cmp_eq_u32 s20, 0
 s_cbranch_scc1 no_old_18
 buffer_load_dwordx2 v[48:49], v12, s[8:11], 0 offen
no_old_18:
v_accvgpr_read_b32 v16, a72
v_accvgpr_read_b32 v17, a73
v_accvgpr_read_b32 v18, a74
v_accvgpr_read_b32 v19, a75
s_nop 1
s_or_b32 s28, s24, s25
 s_cmp_eq_u32 s28, 0
 s_cbranch_scc1 no_shadow_18
 v_mul_lo_u32 v13, s21, v10
 v_add_u32_e32 v13, v13, v11
 v_lshlrev_b32_e32 v13, 2, v13
 s_lshl_b32 s28, s21, 2
buffer_store_dword v16, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v17, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v18, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v19, v13, s[24:27], 0 offen
no_shadow_18:
v_trunc_f32_e32 v14, v16
 v_cmp_eq_f32_e32 vcc, v14, v16
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v16
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v16
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v32, v[28:29]
v_trunc_f32_e32 v14, v17
 v_cmp_eq_f32_e32 vcc, v14, v17
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v17
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v17
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v33, v[28:29]
v_trunc_f32_e32 v14, v18
 v_cmp_eq_f32_e32 vcc, v14, v18
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v18
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v18
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v34, v[28:29]
v_trunc_f32_e32 v14, v19
 v_cmp_eq_f32_e32 vcc, v14, v19
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v19
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v19
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v35, v[28:29]
s_waitcnt vmcnt(0)
v_bfe_u32 v14, v48, 0, 16
 v_add_u32_e32 v32, v32, v14
 v_sub_u32_e64 v15, v32, s16
 v_cmp_ge_u32_e64 vcc, v32, s16
 v_cndmask_b32_e32 v32, v32, v15, vcc
v_bfe_u32 v14, v48, 16, 16
 v_add_u32_e32 v33, v33, v14
 v_sub_u32_e64 v15, v33, s16
 v_cmp_ge_u32_e64 vcc, v33, s16
 v_cndmask_b32_e32 v33, v33, v15, vcc
v_bfe_u32 v14, v49, 0, 16
 v_add_u32_e32 v34, v34, v14
 v_sub_u32_e64 v15, v34, s16
 v_cmp_ge_u32_e64 vcc, v34, s16
 v_cndmask_b32_e32 v34, v34, v15, vcc
v_bfe_u32 v14, v49, 16, 16
 v_add_u32_e32 v35, v35, v14
 v_sub_u32_e64 v15, v35, s16
 v_cmp_ge_u32_e64 vcc, v35, s16
 v_cndmask_b32_e32 v35, v35, v15, vcc
v_lshlrev_b32_e32 v33, 16, v33
 v_or_b32_e32 v36, v32, v33
 v_lshlrev_b32_e32 v35, 16, v35
 v_or_b32_e32 v37, v34, v35
 buffer_store_dwordx2 v[36:37], v12, s[8:11], 0 offen
// group 19, AGPR 76..79, row+32, col+48
v_add_u32_e32 v10, 32, v8
 v_add_u32_e32 v11, 48, v9
 v_mul_lo_u32 v12, s21, v11
 v_add_u32_e32 v12, v12, v10
 v_lshlrev_b32_e32 v12, 1, v12
 v_mov_b32_e32 v48, 0
 v_mov_b32_e32 v49, 0
 s_cmp_eq_u32 s20, 0
 s_cbranch_scc1 no_old_19
 buffer_load_dwordx2 v[48:49], v12, s[8:11], 0 offen
no_old_19:
v_accvgpr_read_b32 v16, a76
v_accvgpr_read_b32 v17, a77
v_accvgpr_read_b32 v18, a78
v_accvgpr_read_b32 v19, a79
s_nop 1
s_or_b32 s28, s24, s25
 s_cmp_eq_u32 s28, 0
 s_cbranch_scc1 no_shadow_19
 v_mul_lo_u32 v13, s21, v10
 v_add_u32_e32 v13, v13, v11
 v_lshlrev_b32_e32 v13, 2, v13
 s_lshl_b32 s28, s21, 2
buffer_store_dword v16, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v17, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v18, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v19, v13, s[24:27], 0 offen
no_shadow_19:
v_trunc_f32_e32 v14, v16
 v_cmp_eq_f32_e32 vcc, v14, v16
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v16
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v16
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v32, v[28:29]
v_trunc_f32_e32 v14, v17
 v_cmp_eq_f32_e32 vcc, v14, v17
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v17
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v17
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v33, v[28:29]
v_trunc_f32_e32 v14, v18
 v_cmp_eq_f32_e32 vcc, v14, v18
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v18
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v18
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v34, v[28:29]
v_trunc_f32_e32 v14, v19
 v_cmp_eq_f32_e32 vcc, v14, v19
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v19
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v19
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v35, v[28:29]
s_waitcnt vmcnt(0)
v_bfe_u32 v14, v48, 0, 16
 v_add_u32_e32 v32, v32, v14
 v_sub_u32_e64 v15, v32, s16
 v_cmp_ge_u32_e64 vcc, v32, s16
 v_cndmask_b32_e32 v32, v32, v15, vcc
v_bfe_u32 v14, v48, 16, 16
 v_add_u32_e32 v33, v33, v14
 v_sub_u32_e64 v15, v33, s16
 v_cmp_ge_u32_e64 vcc, v33, s16
 v_cndmask_b32_e32 v33, v33, v15, vcc
v_bfe_u32 v14, v49, 0, 16
 v_add_u32_e32 v34, v34, v14
 v_sub_u32_e64 v15, v34, s16
 v_cmp_ge_u32_e64 vcc, v34, s16
 v_cndmask_b32_e32 v34, v34, v15, vcc
v_bfe_u32 v14, v49, 16, 16
 v_add_u32_e32 v35, v35, v14
 v_sub_u32_e64 v15, v35, s16
 v_cmp_ge_u32_e64 vcc, v35, s16
 v_cndmask_b32_e32 v35, v35, v15, vcc
v_lshlrev_b32_e32 v33, 16, v33
 v_or_b32_e32 v36, v32, v33
 v_lshlrev_b32_e32 v35, 16, v35
 v_or_b32_e32 v37, v34, v35
 buffer_store_dwordx2 v[36:37], v12, s[8:11], 0 offen
// group 20, AGPR 80..83, row+32, col+64
v_add_u32_e32 v10, 32, v8
 v_add_u32_e32 v11, 64, v9
 v_mul_lo_u32 v12, s21, v11
 v_add_u32_e32 v12, v12, v10
 v_lshlrev_b32_e32 v12, 1, v12
 v_mov_b32_e32 v48, 0
 v_mov_b32_e32 v49, 0
 s_cmp_eq_u32 s20, 0
 s_cbranch_scc1 no_old_20
 buffer_load_dwordx2 v[48:49], v12, s[8:11], 0 offen
no_old_20:
v_accvgpr_read_b32 v16, a80
v_accvgpr_read_b32 v17, a81
v_accvgpr_read_b32 v18, a82
v_accvgpr_read_b32 v19, a83
s_nop 1
s_or_b32 s28, s24, s25
 s_cmp_eq_u32 s28, 0
 s_cbranch_scc1 no_shadow_20
 v_mul_lo_u32 v13, s21, v10
 v_add_u32_e32 v13, v13, v11
 v_lshlrev_b32_e32 v13, 2, v13
 s_lshl_b32 s28, s21, 2
buffer_store_dword v16, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v17, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v18, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v19, v13, s[24:27], 0 offen
no_shadow_20:
v_trunc_f32_e32 v14, v16
 v_cmp_eq_f32_e32 vcc, v14, v16
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v16
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v16
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v32, v[28:29]
v_trunc_f32_e32 v14, v17
 v_cmp_eq_f32_e32 vcc, v14, v17
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v17
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v17
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v33, v[28:29]
v_trunc_f32_e32 v14, v18
 v_cmp_eq_f32_e32 vcc, v14, v18
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v18
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v18
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v34, v[28:29]
v_trunc_f32_e32 v14, v19
 v_cmp_eq_f32_e32 vcc, v14, v19
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v19
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v19
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v35, v[28:29]
s_waitcnt vmcnt(0)
v_bfe_u32 v14, v48, 0, 16
 v_add_u32_e32 v32, v32, v14
 v_sub_u32_e64 v15, v32, s16
 v_cmp_ge_u32_e64 vcc, v32, s16
 v_cndmask_b32_e32 v32, v32, v15, vcc
v_bfe_u32 v14, v48, 16, 16
 v_add_u32_e32 v33, v33, v14
 v_sub_u32_e64 v15, v33, s16
 v_cmp_ge_u32_e64 vcc, v33, s16
 v_cndmask_b32_e32 v33, v33, v15, vcc
v_bfe_u32 v14, v49, 0, 16
 v_add_u32_e32 v34, v34, v14
 v_sub_u32_e64 v15, v34, s16
 v_cmp_ge_u32_e64 vcc, v34, s16
 v_cndmask_b32_e32 v34, v34, v15, vcc
v_bfe_u32 v14, v49, 16, 16
 v_add_u32_e32 v35, v35, v14
 v_sub_u32_e64 v15, v35, s16
 v_cmp_ge_u32_e64 vcc, v35, s16
 v_cndmask_b32_e32 v35, v35, v15, vcc
v_lshlrev_b32_e32 v33, 16, v33
 v_or_b32_e32 v36, v32, v33
 v_lshlrev_b32_e32 v35, 16, v35
 v_or_b32_e32 v37, v34, v35
 buffer_store_dwordx2 v[36:37], v12, s[8:11], 0 offen
// group 21, AGPR 84..87, row+32, col+80
v_add_u32_e32 v10, 32, v8
 v_add_u32_e32 v11, 80, v9
 v_mul_lo_u32 v12, s21, v11
 v_add_u32_e32 v12, v12, v10
 v_lshlrev_b32_e32 v12, 1, v12
 v_mov_b32_e32 v48, 0
 v_mov_b32_e32 v49, 0
 s_cmp_eq_u32 s20, 0
 s_cbranch_scc1 no_old_21
 buffer_load_dwordx2 v[48:49], v12, s[8:11], 0 offen
no_old_21:
v_accvgpr_read_b32 v16, a84
v_accvgpr_read_b32 v17, a85
v_accvgpr_read_b32 v18, a86
v_accvgpr_read_b32 v19, a87
s_nop 1
s_or_b32 s28, s24, s25
 s_cmp_eq_u32 s28, 0
 s_cbranch_scc1 no_shadow_21
 v_mul_lo_u32 v13, s21, v10
 v_add_u32_e32 v13, v13, v11
 v_lshlrev_b32_e32 v13, 2, v13
 s_lshl_b32 s28, s21, 2
buffer_store_dword v16, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v17, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v18, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v19, v13, s[24:27], 0 offen
no_shadow_21:
v_trunc_f32_e32 v14, v16
 v_cmp_eq_f32_e32 vcc, v14, v16
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v16
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v16
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v32, v[28:29]
v_trunc_f32_e32 v14, v17
 v_cmp_eq_f32_e32 vcc, v14, v17
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v17
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v17
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v33, v[28:29]
v_trunc_f32_e32 v14, v18
 v_cmp_eq_f32_e32 vcc, v14, v18
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v18
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v18
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v34, v[28:29]
v_trunc_f32_e32 v14, v19
 v_cmp_eq_f32_e32 vcc, v14, v19
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v19
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v19
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v35, v[28:29]
s_waitcnt vmcnt(0)
v_bfe_u32 v14, v48, 0, 16
 v_add_u32_e32 v32, v32, v14
 v_sub_u32_e64 v15, v32, s16
 v_cmp_ge_u32_e64 vcc, v32, s16
 v_cndmask_b32_e32 v32, v32, v15, vcc
v_bfe_u32 v14, v48, 16, 16
 v_add_u32_e32 v33, v33, v14
 v_sub_u32_e64 v15, v33, s16
 v_cmp_ge_u32_e64 vcc, v33, s16
 v_cndmask_b32_e32 v33, v33, v15, vcc
v_bfe_u32 v14, v49, 0, 16
 v_add_u32_e32 v34, v34, v14
 v_sub_u32_e64 v15, v34, s16
 v_cmp_ge_u32_e64 vcc, v34, s16
 v_cndmask_b32_e32 v34, v34, v15, vcc
v_bfe_u32 v14, v49, 16, 16
 v_add_u32_e32 v35, v35, v14
 v_sub_u32_e64 v15, v35, s16
 v_cmp_ge_u32_e64 vcc, v35, s16
 v_cndmask_b32_e32 v35, v35, v15, vcc
v_lshlrev_b32_e32 v33, 16, v33
 v_or_b32_e32 v36, v32, v33
 v_lshlrev_b32_e32 v35, 16, v35
 v_or_b32_e32 v37, v34, v35
 buffer_store_dwordx2 v[36:37], v12, s[8:11], 0 offen
// group 22, AGPR 88..91, row+32, col+96
v_add_u32_e32 v10, 32, v8
 v_add_u32_e32 v11, 96, v9
 v_mul_lo_u32 v12, s21, v11
 v_add_u32_e32 v12, v12, v10
 v_lshlrev_b32_e32 v12, 1, v12
 v_mov_b32_e32 v48, 0
 v_mov_b32_e32 v49, 0
 s_cmp_eq_u32 s20, 0
 s_cbranch_scc1 no_old_22
 buffer_load_dwordx2 v[48:49], v12, s[8:11], 0 offen
no_old_22:
v_accvgpr_read_b32 v16, a88
v_accvgpr_read_b32 v17, a89
v_accvgpr_read_b32 v18, a90
v_accvgpr_read_b32 v19, a91
s_nop 1
s_or_b32 s28, s24, s25
 s_cmp_eq_u32 s28, 0
 s_cbranch_scc1 no_shadow_22
 v_mul_lo_u32 v13, s21, v10
 v_add_u32_e32 v13, v13, v11
 v_lshlrev_b32_e32 v13, 2, v13
 s_lshl_b32 s28, s21, 2
buffer_store_dword v16, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v17, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v18, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v19, v13, s[24:27], 0 offen
no_shadow_22:
v_trunc_f32_e32 v14, v16
 v_cmp_eq_f32_e32 vcc, v14, v16
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v16
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v16
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v32, v[28:29]
v_trunc_f32_e32 v14, v17
 v_cmp_eq_f32_e32 vcc, v14, v17
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v17
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v17
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v33, v[28:29]
v_trunc_f32_e32 v14, v18
 v_cmp_eq_f32_e32 vcc, v14, v18
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v18
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v18
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v34, v[28:29]
v_trunc_f32_e32 v14, v19
 v_cmp_eq_f32_e32 vcc, v14, v19
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v19
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v19
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v35, v[28:29]
s_waitcnt vmcnt(0)
v_bfe_u32 v14, v48, 0, 16
 v_add_u32_e32 v32, v32, v14
 v_sub_u32_e64 v15, v32, s16
 v_cmp_ge_u32_e64 vcc, v32, s16
 v_cndmask_b32_e32 v32, v32, v15, vcc
v_bfe_u32 v14, v48, 16, 16
 v_add_u32_e32 v33, v33, v14
 v_sub_u32_e64 v15, v33, s16
 v_cmp_ge_u32_e64 vcc, v33, s16
 v_cndmask_b32_e32 v33, v33, v15, vcc
v_bfe_u32 v14, v49, 0, 16
 v_add_u32_e32 v34, v34, v14
 v_sub_u32_e64 v15, v34, s16
 v_cmp_ge_u32_e64 vcc, v34, s16
 v_cndmask_b32_e32 v34, v34, v15, vcc
v_bfe_u32 v14, v49, 16, 16
 v_add_u32_e32 v35, v35, v14
 v_sub_u32_e64 v15, v35, s16
 v_cmp_ge_u32_e64 vcc, v35, s16
 v_cndmask_b32_e32 v35, v35, v15, vcc
v_lshlrev_b32_e32 v33, 16, v33
 v_or_b32_e32 v36, v32, v33
 v_lshlrev_b32_e32 v35, 16, v35
 v_or_b32_e32 v37, v34, v35
 buffer_store_dwordx2 v[36:37], v12, s[8:11], 0 offen
// group 23, AGPR 92..95, row+32, col+112
v_add_u32_e32 v10, 32, v8
 v_add_u32_e32 v11, 112, v9
 v_mul_lo_u32 v12, s21, v11
 v_add_u32_e32 v12, v12, v10
 v_lshlrev_b32_e32 v12, 1, v12
 v_mov_b32_e32 v48, 0
 v_mov_b32_e32 v49, 0
 s_cmp_eq_u32 s20, 0
 s_cbranch_scc1 no_old_23
 buffer_load_dwordx2 v[48:49], v12, s[8:11], 0 offen
no_old_23:
v_accvgpr_read_b32 v16, a92
v_accvgpr_read_b32 v17, a93
v_accvgpr_read_b32 v18, a94
v_accvgpr_read_b32 v19, a95
s_nop 1
s_or_b32 s28, s24, s25
 s_cmp_eq_u32 s28, 0
 s_cbranch_scc1 no_shadow_23
 v_mul_lo_u32 v13, s21, v10
 v_add_u32_e32 v13, v13, v11
 v_lshlrev_b32_e32 v13, 2, v13
 s_lshl_b32 s28, s21, 2
buffer_store_dword v16, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v17, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v18, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v19, v13, s[24:27], 0 offen
no_shadow_23:
v_trunc_f32_e32 v14, v16
 v_cmp_eq_f32_e32 vcc, v14, v16
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v16
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v16
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v32, v[28:29]
v_trunc_f32_e32 v14, v17
 v_cmp_eq_f32_e32 vcc, v14, v17
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v17
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v17
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v33, v[28:29]
v_trunc_f32_e32 v14, v18
 v_cmp_eq_f32_e32 vcc, v14, v18
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v18
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v18
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v34, v[28:29]
v_trunc_f32_e32 v14, v19
 v_cmp_eq_f32_e32 vcc, v14, v19
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v19
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v19
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v35, v[28:29]
s_waitcnt vmcnt(0)
v_bfe_u32 v14, v48, 0, 16
 v_add_u32_e32 v32, v32, v14
 v_sub_u32_e64 v15, v32, s16
 v_cmp_ge_u32_e64 vcc, v32, s16
 v_cndmask_b32_e32 v32, v32, v15, vcc
v_bfe_u32 v14, v48, 16, 16
 v_add_u32_e32 v33, v33, v14
 v_sub_u32_e64 v15, v33, s16
 v_cmp_ge_u32_e64 vcc, v33, s16
 v_cndmask_b32_e32 v33, v33, v15, vcc
v_bfe_u32 v14, v49, 0, 16
 v_add_u32_e32 v34, v34, v14
 v_sub_u32_e64 v15, v34, s16
 v_cmp_ge_u32_e64 vcc, v34, s16
 v_cndmask_b32_e32 v34, v34, v15, vcc
v_bfe_u32 v14, v49, 16, 16
 v_add_u32_e32 v35, v35, v14
 v_sub_u32_e64 v15, v35, s16
 v_cmp_ge_u32_e64 vcc, v35, s16
 v_cndmask_b32_e32 v35, v35, v15, vcc
v_lshlrev_b32_e32 v33, 16, v33
 v_or_b32_e32 v36, v32, v33
 v_lshlrev_b32_e32 v35, 16, v35
 v_or_b32_e32 v37, v34, v35
 buffer_store_dwordx2 v[36:37], v12, s[8:11], 0 offen
// group 24, AGPR 96..99, row+48, col+0
v_add_u32_e32 v10, 48, v8
 v_add_u32_e32 v11, 0, v9
 v_mul_lo_u32 v12, s21, v11
 v_add_u32_e32 v12, v12, v10
 v_lshlrev_b32_e32 v12, 1, v12
 v_mov_b32_e32 v48, 0
 v_mov_b32_e32 v49, 0
 s_cmp_eq_u32 s20, 0
 s_cbranch_scc1 no_old_24
 buffer_load_dwordx2 v[48:49], v12, s[8:11], 0 offen
no_old_24:
v_accvgpr_read_b32 v16, a96
v_accvgpr_read_b32 v17, a97
v_accvgpr_read_b32 v18, a98
v_accvgpr_read_b32 v19, a99
s_nop 1
s_or_b32 s28, s24, s25
 s_cmp_eq_u32 s28, 0
 s_cbranch_scc1 no_shadow_24
 v_mul_lo_u32 v13, s21, v10
 v_add_u32_e32 v13, v13, v11
 v_lshlrev_b32_e32 v13, 2, v13
 s_lshl_b32 s28, s21, 2
buffer_store_dword v16, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v17, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v18, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v19, v13, s[24:27], 0 offen
no_shadow_24:
v_trunc_f32_e32 v14, v16
 v_cmp_eq_f32_e32 vcc, v14, v16
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v16
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v16
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v32, v[28:29]
v_trunc_f32_e32 v14, v17
 v_cmp_eq_f32_e32 vcc, v14, v17
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v17
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v17
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v33, v[28:29]
v_trunc_f32_e32 v14, v18
 v_cmp_eq_f32_e32 vcc, v14, v18
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v18
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v18
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v34, v[28:29]
v_trunc_f32_e32 v14, v19
 v_cmp_eq_f32_e32 vcc, v14, v19
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v19
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v19
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v35, v[28:29]
s_waitcnt vmcnt(0)
v_bfe_u32 v14, v48, 0, 16
 v_add_u32_e32 v32, v32, v14
 v_sub_u32_e64 v15, v32, s16
 v_cmp_ge_u32_e64 vcc, v32, s16
 v_cndmask_b32_e32 v32, v32, v15, vcc
v_bfe_u32 v14, v48, 16, 16
 v_add_u32_e32 v33, v33, v14
 v_sub_u32_e64 v15, v33, s16
 v_cmp_ge_u32_e64 vcc, v33, s16
 v_cndmask_b32_e32 v33, v33, v15, vcc
v_bfe_u32 v14, v49, 0, 16
 v_add_u32_e32 v34, v34, v14
 v_sub_u32_e64 v15, v34, s16
 v_cmp_ge_u32_e64 vcc, v34, s16
 v_cndmask_b32_e32 v34, v34, v15, vcc
v_bfe_u32 v14, v49, 16, 16
 v_add_u32_e32 v35, v35, v14
 v_sub_u32_e64 v15, v35, s16
 v_cmp_ge_u32_e64 vcc, v35, s16
 v_cndmask_b32_e32 v35, v35, v15, vcc
v_lshlrev_b32_e32 v33, 16, v33
 v_or_b32_e32 v36, v32, v33
 v_lshlrev_b32_e32 v35, 16, v35
 v_or_b32_e32 v37, v34, v35
 buffer_store_dwordx2 v[36:37], v12, s[8:11], 0 offen
// group 25, AGPR 100..103, row+48, col+16
v_add_u32_e32 v10, 48, v8
 v_add_u32_e32 v11, 16, v9
 v_mul_lo_u32 v12, s21, v11
 v_add_u32_e32 v12, v12, v10
 v_lshlrev_b32_e32 v12, 1, v12
 v_mov_b32_e32 v48, 0
 v_mov_b32_e32 v49, 0
 s_cmp_eq_u32 s20, 0
 s_cbranch_scc1 no_old_25
 buffer_load_dwordx2 v[48:49], v12, s[8:11], 0 offen
no_old_25:
v_accvgpr_read_b32 v16, a100
v_accvgpr_read_b32 v17, a101
v_accvgpr_read_b32 v18, a102
v_accvgpr_read_b32 v19, a103
s_nop 1
s_or_b32 s28, s24, s25
 s_cmp_eq_u32 s28, 0
 s_cbranch_scc1 no_shadow_25
 v_mul_lo_u32 v13, s21, v10
 v_add_u32_e32 v13, v13, v11
 v_lshlrev_b32_e32 v13, 2, v13
 s_lshl_b32 s28, s21, 2
buffer_store_dword v16, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v17, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v18, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v19, v13, s[24:27], 0 offen
no_shadow_25:
v_trunc_f32_e32 v14, v16
 v_cmp_eq_f32_e32 vcc, v14, v16
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v16
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v16
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v32, v[28:29]
v_trunc_f32_e32 v14, v17
 v_cmp_eq_f32_e32 vcc, v14, v17
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v17
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v17
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v33, v[28:29]
v_trunc_f32_e32 v14, v18
 v_cmp_eq_f32_e32 vcc, v14, v18
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v18
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v18
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v34, v[28:29]
v_trunc_f32_e32 v14, v19
 v_cmp_eq_f32_e32 vcc, v14, v19
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v19
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v19
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v35, v[28:29]
s_waitcnt vmcnt(0)
v_bfe_u32 v14, v48, 0, 16
 v_add_u32_e32 v32, v32, v14
 v_sub_u32_e64 v15, v32, s16
 v_cmp_ge_u32_e64 vcc, v32, s16
 v_cndmask_b32_e32 v32, v32, v15, vcc
v_bfe_u32 v14, v48, 16, 16
 v_add_u32_e32 v33, v33, v14
 v_sub_u32_e64 v15, v33, s16
 v_cmp_ge_u32_e64 vcc, v33, s16
 v_cndmask_b32_e32 v33, v33, v15, vcc
v_bfe_u32 v14, v49, 0, 16
 v_add_u32_e32 v34, v34, v14
 v_sub_u32_e64 v15, v34, s16
 v_cmp_ge_u32_e64 vcc, v34, s16
 v_cndmask_b32_e32 v34, v34, v15, vcc
v_bfe_u32 v14, v49, 16, 16
 v_add_u32_e32 v35, v35, v14
 v_sub_u32_e64 v15, v35, s16
 v_cmp_ge_u32_e64 vcc, v35, s16
 v_cndmask_b32_e32 v35, v35, v15, vcc
v_lshlrev_b32_e32 v33, 16, v33
 v_or_b32_e32 v36, v32, v33
 v_lshlrev_b32_e32 v35, 16, v35
 v_or_b32_e32 v37, v34, v35
 buffer_store_dwordx2 v[36:37], v12, s[8:11], 0 offen
// group 26, AGPR 104..107, row+48, col+32
v_add_u32_e32 v10, 48, v8
 v_add_u32_e32 v11, 32, v9
 v_mul_lo_u32 v12, s21, v11
 v_add_u32_e32 v12, v12, v10
 v_lshlrev_b32_e32 v12, 1, v12
 v_mov_b32_e32 v48, 0
 v_mov_b32_e32 v49, 0
 s_cmp_eq_u32 s20, 0
 s_cbranch_scc1 no_old_26
 buffer_load_dwordx2 v[48:49], v12, s[8:11], 0 offen
no_old_26:
v_accvgpr_read_b32 v16, a104
v_accvgpr_read_b32 v17, a105
v_accvgpr_read_b32 v18, a106
v_accvgpr_read_b32 v19, a107
s_nop 1
s_or_b32 s28, s24, s25
 s_cmp_eq_u32 s28, 0
 s_cbranch_scc1 no_shadow_26
 v_mul_lo_u32 v13, s21, v10
 v_add_u32_e32 v13, v13, v11
 v_lshlrev_b32_e32 v13, 2, v13
 s_lshl_b32 s28, s21, 2
buffer_store_dword v16, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v17, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v18, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v19, v13, s[24:27], 0 offen
no_shadow_26:
v_trunc_f32_e32 v14, v16
 v_cmp_eq_f32_e32 vcc, v14, v16
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v16
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v16
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v32, v[28:29]
v_trunc_f32_e32 v14, v17
 v_cmp_eq_f32_e32 vcc, v14, v17
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v17
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v17
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v33, v[28:29]
v_trunc_f32_e32 v14, v18
 v_cmp_eq_f32_e32 vcc, v14, v18
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v18
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v18
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v34, v[28:29]
v_trunc_f32_e32 v14, v19
 v_cmp_eq_f32_e32 vcc, v14, v19
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v19
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v19
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v35, v[28:29]
s_waitcnt vmcnt(0)
v_bfe_u32 v14, v48, 0, 16
 v_add_u32_e32 v32, v32, v14
 v_sub_u32_e64 v15, v32, s16
 v_cmp_ge_u32_e64 vcc, v32, s16
 v_cndmask_b32_e32 v32, v32, v15, vcc
v_bfe_u32 v14, v48, 16, 16
 v_add_u32_e32 v33, v33, v14
 v_sub_u32_e64 v15, v33, s16
 v_cmp_ge_u32_e64 vcc, v33, s16
 v_cndmask_b32_e32 v33, v33, v15, vcc
v_bfe_u32 v14, v49, 0, 16
 v_add_u32_e32 v34, v34, v14
 v_sub_u32_e64 v15, v34, s16
 v_cmp_ge_u32_e64 vcc, v34, s16
 v_cndmask_b32_e32 v34, v34, v15, vcc
v_bfe_u32 v14, v49, 16, 16
 v_add_u32_e32 v35, v35, v14
 v_sub_u32_e64 v15, v35, s16
 v_cmp_ge_u32_e64 vcc, v35, s16
 v_cndmask_b32_e32 v35, v35, v15, vcc
v_lshlrev_b32_e32 v33, 16, v33
 v_or_b32_e32 v36, v32, v33
 v_lshlrev_b32_e32 v35, 16, v35
 v_or_b32_e32 v37, v34, v35
 buffer_store_dwordx2 v[36:37], v12, s[8:11], 0 offen
// group 27, AGPR 108..111, row+48, col+48
v_add_u32_e32 v10, 48, v8
 v_add_u32_e32 v11, 48, v9
 v_mul_lo_u32 v12, s21, v11
 v_add_u32_e32 v12, v12, v10
 v_lshlrev_b32_e32 v12, 1, v12
 v_mov_b32_e32 v48, 0
 v_mov_b32_e32 v49, 0
 s_cmp_eq_u32 s20, 0
 s_cbranch_scc1 no_old_27
 buffer_load_dwordx2 v[48:49], v12, s[8:11], 0 offen
no_old_27:
v_accvgpr_read_b32 v16, a108
v_accvgpr_read_b32 v17, a109
v_accvgpr_read_b32 v18, a110
v_accvgpr_read_b32 v19, a111
s_nop 1
s_or_b32 s28, s24, s25
 s_cmp_eq_u32 s28, 0
 s_cbranch_scc1 no_shadow_27
 v_mul_lo_u32 v13, s21, v10
 v_add_u32_e32 v13, v13, v11
 v_lshlrev_b32_e32 v13, 2, v13
 s_lshl_b32 s28, s21, 2
buffer_store_dword v16, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v17, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v18, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v19, v13, s[24:27], 0 offen
no_shadow_27:
v_trunc_f32_e32 v14, v16
 v_cmp_eq_f32_e32 vcc, v14, v16
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v16
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v16
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v32, v[28:29]
v_trunc_f32_e32 v14, v17
 v_cmp_eq_f32_e32 vcc, v14, v17
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v17
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v17
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v33, v[28:29]
v_trunc_f32_e32 v14, v18
 v_cmp_eq_f32_e32 vcc, v14, v18
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v18
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v18
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v34, v[28:29]
v_trunc_f32_e32 v14, v19
 v_cmp_eq_f32_e32 vcc, v14, v19
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v19
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v19
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v35, v[28:29]
s_waitcnt vmcnt(0)
v_bfe_u32 v14, v48, 0, 16
 v_add_u32_e32 v32, v32, v14
 v_sub_u32_e64 v15, v32, s16
 v_cmp_ge_u32_e64 vcc, v32, s16
 v_cndmask_b32_e32 v32, v32, v15, vcc
v_bfe_u32 v14, v48, 16, 16
 v_add_u32_e32 v33, v33, v14
 v_sub_u32_e64 v15, v33, s16
 v_cmp_ge_u32_e64 vcc, v33, s16
 v_cndmask_b32_e32 v33, v33, v15, vcc
v_bfe_u32 v14, v49, 0, 16
 v_add_u32_e32 v34, v34, v14
 v_sub_u32_e64 v15, v34, s16
 v_cmp_ge_u32_e64 vcc, v34, s16
 v_cndmask_b32_e32 v34, v34, v15, vcc
v_bfe_u32 v14, v49, 16, 16
 v_add_u32_e32 v35, v35, v14
 v_sub_u32_e64 v15, v35, s16
 v_cmp_ge_u32_e64 vcc, v35, s16
 v_cndmask_b32_e32 v35, v35, v15, vcc
v_lshlrev_b32_e32 v33, 16, v33
 v_or_b32_e32 v36, v32, v33
 v_lshlrev_b32_e32 v35, 16, v35
 v_or_b32_e32 v37, v34, v35
 buffer_store_dwordx2 v[36:37], v12, s[8:11], 0 offen
// group 28, AGPR 112..115, row+48, col+64
v_add_u32_e32 v10, 48, v8
 v_add_u32_e32 v11, 64, v9
 v_mul_lo_u32 v12, s21, v11
 v_add_u32_e32 v12, v12, v10
 v_lshlrev_b32_e32 v12, 1, v12
 v_mov_b32_e32 v48, 0
 v_mov_b32_e32 v49, 0
 s_cmp_eq_u32 s20, 0
 s_cbranch_scc1 no_old_28
 buffer_load_dwordx2 v[48:49], v12, s[8:11], 0 offen
no_old_28:
v_accvgpr_read_b32 v16, a112
v_accvgpr_read_b32 v17, a113
v_accvgpr_read_b32 v18, a114
v_accvgpr_read_b32 v19, a115
s_nop 1
s_or_b32 s28, s24, s25
 s_cmp_eq_u32 s28, 0
 s_cbranch_scc1 no_shadow_28
 v_mul_lo_u32 v13, s21, v10
 v_add_u32_e32 v13, v13, v11
 v_lshlrev_b32_e32 v13, 2, v13
 s_lshl_b32 s28, s21, 2
buffer_store_dword v16, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v17, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v18, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v19, v13, s[24:27], 0 offen
no_shadow_28:
v_trunc_f32_e32 v14, v16
 v_cmp_eq_f32_e32 vcc, v14, v16
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v16
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v16
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v32, v[28:29]
v_trunc_f32_e32 v14, v17
 v_cmp_eq_f32_e32 vcc, v14, v17
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v17
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v17
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v33, v[28:29]
v_trunc_f32_e32 v14, v18
 v_cmp_eq_f32_e32 vcc, v14, v18
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v18
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v18
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v34, v[28:29]
v_trunc_f32_e32 v14, v19
 v_cmp_eq_f32_e32 vcc, v14, v19
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v19
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v19
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v35, v[28:29]
s_waitcnt vmcnt(0)
v_bfe_u32 v14, v48, 0, 16
 v_add_u32_e32 v32, v32, v14
 v_sub_u32_e64 v15, v32, s16
 v_cmp_ge_u32_e64 vcc, v32, s16
 v_cndmask_b32_e32 v32, v32, v15, vcc
v_bfe_u32 v14, v48, 16, 16
 v_add_u32_e32 v33, v33, v14
 v_sub_u32_e64 v15, v33, s16
 v_cmp_ge_u32_e64 vcc, v33, s16
 v_cndmask_b32_e32 v33, v33, v15, vcc
v_bfe_u32 v14, v49, 0, 16
 v_add_u32_e32 v34, v34, v14
 v_sub_u32_e64 v15, v34, s16
 v_cmp_ge_u32_e64 vcc, v34, s16
 v_cndmask_b32_e32 v34, v34, v15, vcc
v_bfe_u32 v14, v49, 16, 16
 v_add_u32_e32 v35, v35, v14
 v_sub_u32_e64 v15, v35, s16
 v_cmp_ge_u32_e64 vcc, v35, s16
 v_cndmask_b32_e32 v35, v35, v15, vcc
v_lshlrev_b32_e32 v33, 16, v33
 v_or_b32_e32 v36, v32, v33
 v_lshlrev_b32_e32 v35, 16, v35
 v_or_b32_e32 v37, v34, v35
 buffer_store_dwordx2 v[36:37], v12, s[8:11], 0 offen
// group 29, AGPR 116..119, row+48, col+80
v_add_u32_e32 v10, 48, v8
 v_add_u32_e32 v11, 80, v9
 v_mul_lo_u32 v12, s21, v11
 v_add_u32_e32 v12, v12, v10
 v_lshlrev_b32_e32 v12, 1, v12
 v_mov_b32_e32 v48, 0
 v_mov_b32_e32 v49, 0
 s_cmp_eq_u32 s20, 0
 s_cbranch_scc1 no_old_29
 buffer_load_dwordx2 v[48:49], v12, s[8:11], 0 offen
no_old_29:
v_accvgpr_read_b32 v16, a116
v_accvgpr_read_b32 v17, a117
v_accvgpr_read_b32 v18, a118
v_accvgpr_read_b32 v19, a119
s_nop 1
s_or_b32 s28, s24, s25
 s_cmp_eq_u32 s28, 0
 s_cbranch_scc1 no_shadow_29
 v_mul_lo_u32 v13, s21, v10
 v_add_u32_e32 v13, v13, v11
 v_lshlrev_b32_e32 v13, 2, v13
 s_lshl_b32 s28, s21, 2
buffer_store_dword v16, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v17, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v18, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v19, v13, s[24:27], 0 offen
no_shadow_29:
v_trunc_f32_e32 v14, v16
 v_cmp_eq_f32_e32 vcc, v14, v16
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v16
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v16
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v32, v[28:29]
v_trunc_f32_e32 v14, v17
 v_cmp_eq_f32_e32 vcc, v14, v17
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v17
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v17
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v33, v[28:29]
v_trunc_f32_e32 v14, v18
 v_cmp_eq_f32_e32 vcc, v14, v18
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v18
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v18
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v34, v[28:29]
v_trunc_f32_e32 v14, v19
 v_cmp_eq_f32_e32 vcc, v14, v19
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v19
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v19
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v35, v[28:29]
s_waitcnt vmcnt(0)
v_bfe_u32 v14, v48, 0, 16
 v_add_u32_e32 v32, v32, v14
 v_sub_u32_e64 v15, v32, s16
 v_cmp_ge_u32_e64 vcc, v32, s16
 v_cndmask_b32_e32 v32, v32, v15, vcc
v_bfe_u32 v14, v48, 16, 16
 v_add_u32_e32 v33, v33, v14
 v_sub_u32_e64 v15, v33, s16
 v_cmp_ge_u32_e64 vcc, v33, s16
 v_cndmask_b32_e32 v33, v33, v15, vcc
v_bfe_u32 v14, v49, 0, 16
 v_add_u32_e32 v34, v34, v14
 v_sub_u32_e64 v15, v34, s16
 v_cmp_ge_u32_e64 vcc, v34, s16
 v_cndmask_b32_e32 v34, v34, v15, vcc
v_bfe_u32 v14, v49, 16, 16
 v_add_u32_e32 v35, v35, v14
 v_sub_u32_e64 v15, v35, s16
 v_cmp_ge_u32_e64 vcc, v35, s16
 v_cndmask_b32_e32 v35, v35, v15, vcc
v_lshlrev_b32_e32 v33, 16, v33
 v_or_b32_e32 v36, v32, v33
 v_lshlrev_b32_e32 v35, 16, v35
 v_or_b32_e32 v37, v34, v35
 buffer_store_dwordx2 v[36:37], v12, s[8:11], 0 offen
// group 30, AGPR 120..123, row+48, col+96
v_add_u32_e32 v10, 48, v8
 v_add_u32_e32 v11, 96, v9
 v_mul_lo_u32 v12, s21, v11
 v_add_u32_e32 v12, v12, v10
 v_lshlrev_b32_e32 v12, 1, v12
 v_mov_b32_e32 v48, 0
 v_mov_b32_e32 v49, 0
 s_cmp_eq_u32 s20, 0
 s_cbranch_scc1 no_old_30
 buffer_load_dwordx2 v[48:49], v12, s[8:11], 0 offen
no_old_30:
v_accvgpr_read_b32 v16, a120
v_accvgpr_read_b32 v17, a121
v_accvgpr_read_b32 v18, a122
v_accvgpr_read_b32 v19, a123
s_nop 1
s_or_b32 s28, s24, s25
 s_cmp_eq_u32 s28, 0
 s_cbranch_scc1 no_shadow_30
 v_mul_lo_u32 v13, s21, v10
 v_add_u32_e32 v13, v13, v11
 v_lshlrev_b32_e32 v13, 2, v13
 s_lshl_b32 s28, s21, 2
buffer_store_dword v16, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v17, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v18, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v19, v13, s[24:27], 0 offen
no_shadow_30:
v_trunc_f32_e32 v14, v16
 v_cmp_eq_f32_e32 vcc, v14, v16
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v16
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v16
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v32, v[28:29]
v_trunc_f32_e32 v14, v17
 v_cmp_eq_f32_e32 vcc, v14, v17
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v17
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v17
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v33, v[28:29]
v_trunc_f32_e32 v14, v18
 v_cmp_eq_f32_e32 vcc, v14, v18
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v18
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v18
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v34, v[28:29]
v_trunc_f32_e32 v14, v19
 v_cmp_eq_f32_e32 vcc, v14, v19
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v19
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v19
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v35, v[28:29]
s_waitcnt vmcnt(0)
v_bfe_u32 v14, v48, 0, 16
 v_add_u32_e32 v32, v32, v14
 v_sub_u32_e64 v15, v32, s16
 v_cmp_ge_u32_e64 vcc, v32, s16
 v_cndmask_b32_e32 v32, v32, v15, vcc
v_bfe_u32 v14, v48, 16, 16
 v_add_u32_e32 v33, v33, v14
 v_sub_u32_e64 v15, v33, s16
 v_cmp_ge_u32_e64 vcc, v33, s16
 v_cndmask_b32_e32 v33, v33, v15, vcc
v_bfe_u32 v14, v49, 0, 16
 v_add_u32_e32 v34, v34, v14
 v_sub_u32_e64 v15, v34, s16
 v_cmp_ge_u32_e64 vcc, v34, s16
 v_cndmask_b32_e32 v34, v34, v15, vcc
v_bfe_u32 v14, v49, 16, 16
 v_add_u32_e32 v35, v35, v14
 v_sub_u32_e64 v15, v35, s16
 v_cmp_ge_u32_e64 vcc, v35, s16
 v_cndmask_b32_e32 v35, v35, v15, vcc
v_lshlrev_b32_e32 v33, 16, v33
 v_or_b32_e32 v36, v32, v33
 v_lshlrev_b32_e32 v35, 16, v35
 v_or_b32_e32 v37, v34, v35
 buffer_store_dwordx2 v[36:37], v12, s[8:11], 0 offen
// group 31, AGPR 124..127, row+48, col+112
v_add_u32_e32 v10, 48, v8
 v_add_u32_e32 v11, 112, v9
 v_mul_lo_u32 v12, s21, v11
 v_add_u32_e32 v12, v12, v10
 v_lshlrev_b32_e32 v12, 1, v12
 v_mov_b32_e32 v48, 0
 v_mov_b32_e32 v49, 0
 s_cmp_eq_u32 s20, 0
 s_cbranch_scc1 no_old_31
 buffer_load_dwordx2 v[48:49], v12, s[8:11], 0 offen
no_old_31:
v_accvgpr_read_b32 v16, a124
v_accvgpr_read_b32 v17, a125
v_accvgpr_read_b32 v18, a126
v_accvgpr_read_b32 v19, a127
s_nop 1
s_or_b32 s28, s24, s25
 s_cmp_eq_u32 s28, 0
 s_cbranch_scc1 no_shadow_31
 v_mul_lo_u32 v13, s21, v10
 v_add_u32_e32 v13, v13, v11
 v_lshlrev_b32_e32 v13, 2, v13
 s_lshl_b32 s28, s21, 2
buffer_store_dword v16, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v17, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v18, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v19, v13, s[24:27], 0 offen
no_shadow_31:
v_trunc_f32_e32 v14, v16
 v_cmp_eq_f32_e32 vcc, v14, v16
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v16
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v16
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v32, v[28:29]
v_trunc_f32_e32 v14, v17
 v_cmp_eq_f32_e32 vcc, v14, v17
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v17
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v17
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v33, v[28:29]
v_trunc_f32_e32 v14, v18
 v_cmp_eq_f32_e32 vcc, v14, v18
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v18
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v18
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v34, v[28:29]
v_trunc_f32_e32 v14, v19
 v_cmp_eq_f32_e32 vcc, v14, v19
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v19
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v19
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v35, v[28:29]
s_waitcnt vmcnt(0)
v_bfe_u32 v14, v48, 0, 16
 v_add_u32_e32 v32, v32, v14
 v_sub_u32_e64 v15, v32, s16
 v_cmp_ge_u32_e64 vcc, v32, s16
 v_cndmask_b32_e32 v32, v32, v15, vcc
v_bfe_u32 v14, v48, 16, 16
 v_add_u32_e32 v33, v33, v14
 v_sub_u32_e64 v15, v33, s16
 v_cmp_ge_u32_e64 vcc, v33, s16
 v_cndmask_b32_e32 v33, v33, v15, vcc
v_bfe_u32 v14, v49, 0, 16
 v_add_u32_e32 v34, v34, v14
 v_sub_u32_e64 v15, v34, s16
 v_cmp_ge_u32_e64 vcc, v34, s16
 v_cndmask_b32_e32 v34, v34, v15, vcc
v_bfe_u32 v14, v49, 16, 16
 v_add_u32_e32 v35, v35, v14
 v_sub_u32_e64 v15, v35, s16
 v_cmp_ge_u32_e64 vcc, v35, s16
 v_cndmask_b32_e32 v35, v35, v15, vcc
v_lshlrev_b32_e32 v33, 16, v33
 v_or_b32_e32 v36, v32, v33
 v_lshlrev_b32_e32 v35, 16, v35
 v_or_b32_e32 v37, v34, v35
 buffer_store_dwordx2 v[36:37], v12, s[8:11], 0 offen
// group 32, AGPR 128..131, row+64, col+0
v_add_u32_e32 v10, 64, v8
 v_add_u32_e32 v11, 0, v9
 v_mul_lo_u32 v12, s21, v11
 v_add_u32_e32 v12, v12, v10
 v_lshlrev_b32_e32 v12, 1, v12
 v_mov_b32_e32 v48, 0
 v_mov_b32_e32 v49, 0
 s_cmp_eq_u32 s20, 0
 s_cbranch_scc1 no_old_32
 buffer_load_dwordx2 v[48:49], v12, s[8:11], 0 offen
no_old_32:
v_accvgpr_read_b32 v16, a128
v_accvgpr_read_b32 v17, a129
v_accvgpr_read_b32 v18, a130
v_accvgpr_read_b32 v19, a131
s_nop 1
s_or_b32 s28, s24, s25
 s_cmp_eq_u32 s28, 0
 s_cbranch_scc1 no_shadow_32
 v_mul_lo_u32 v13, s21, v10
 v_add_u32_e32 v13, v13, v11
 v_lshlrev_b32_e32 v13, 2, v13
 s_lshl_b32 s28, s21, 2
buffer_store_dword v16, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v17, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v18, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v19, v13, s[24:27], 0 offen
no_shadow_32:
v_trunc_f32_e32 v14, v16
 v_cmp_eq_f32_e32 vcc, v14, v16
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v16
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v16
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v32, v[28:29]
v_trunc_f32_e32 v14, v17
 v_cmp_eq_f32_e32 vcc, v14, v17
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v17
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v17
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v33, v[28:29]
v_trunc_f32_e32 v14, v18
 v_cmp_eq_f32_e32 vcc, v14, v18
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v18
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v18
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v34, v[28:29]
v_trunc_f32_e32 v14, v19
 v_cmp_eq_f32_e32 vcc, v14, v19
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v19
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v19
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v35, v[28:29]
s_waitcnt vmcnt(0)
v_bfe_u32 v14, v48, 0, 16
 v_add_u32_e32 v32, v32, v14
 v_sub_u32_e64 v15, v32, s16
 v_cmp_ge_u32_e64 vcc, v32, s16
 v_cndmask_b32_e32 v32, v32, v15, vcc
v_bfe_u32 v14, v48, 16, 16
 v_add_u32_e32 v33, v33, v14
 v_sub_u32_e64 v15, v33, s16
 v_cmp_ge_u32_e64 vcc, v33, s16
 v_cndmask_b32_e32 v33, v33, v15, vcc
v_bfe_u32 v14, v49, 0, 16
 v_add_u32_e32 v34, v34, v14
 v_sub_u32_e64 v15, v34, s16
 v_cmp_ge_u32_e64 vcc, v34, s16
 v_cndmask_b32_e32 v34, v34, v15, vcc
v_bfe_u32 v14, v49, 16, 16
 v_add_u32_e32 v35, v35, v14
 v_sub_u32_e64 v15, v35, s16
 v_cmp_ge_u32_e64 vcc, v35, s16
 v_cndmask_b32_e32 v35, v35, v15, vcc
v_lshlrev_b32_e32 v33, 16, v33
 v_or_b32_e32 v36, v32, v33
 v_lshlrev_b32_e32 v35, 16, v35
 v_or_b32_e32 v37, v34, v35
 buffer_store_dwordx2 v[36:37], v12, s[8:11], 0 offen
// group 33, AGPR 132..135, row+64, col+16
v_add_u32_e32 v10, 64, v8
 v_add_u32_e32 v11, 16, v9
 v_mul_lo_u32 v12, s21, v11
 v_add_u32_e32 v12, v12, v10
 v_lshlrev_b32_e32 v12, 1, v12
 v_mov_b32_e32 v48, 0
 v_mov_b32_e32 v49, 0
 s_cmp_eq_u32 s20, 0
 s_cbranch_scc1 no_old_33
 buffer_load_dwordx2 v[48:49], v12, s[8:11], 0 offen
no_old_33:
v_accvgpr_read_b32 v16, a132
v_accvgpr_read_b32 v17, a133
v_accvgpr_read_b32 v18, a134
v_accvgpr_read_b32 v19, a135
s_nop 1
s_or_b32 s28, s24, s25
 s_cmp_eq_u32 s28, 0
 s_cbranch_scc1 no_shadow_33
 v_mul_lo_u32 v13, s21, v10
 v_add_u32_e32 v13, v13, v11
 v_lshlrev_b32_e32 v13, 2, v13
 s_lshl_b32 s28, s21, 2
buffer_store_dword v16, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v17, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v18, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v19, v13, s[24:27], 0 offen
no_shadow_33:
v_trunc_f32_e32 v14, v16
 v_cmp_eq_f32_e32 vcc, v14, v16
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v16
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v16
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v32, v[28:29]
v_trunc_f32_e32 v14, v17
 v_cmp_eq_f32_e32 vcc, v14, v17
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v17
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v17
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v33, v[28:29]
v_trunc_f32_e32 v14, v18
 v_cmp_eq_f32_e32 vcc, v14, v18
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v18
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v18
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v34, v[28:29]
v_trunc_f32_e32 v14, v19
 v_cmp_eq_f32_e32 vcc, v14, v19
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v19
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v19
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v35, v[28:29]
s_waitcnt vmcnt(0)
v_bfe_u32 v14, v48, 0, 16
 v_add_u32_e32 v32, v32, v14
 v_sub_u32_e64 v15, v32, s16
 v_cmp_ge_u32_e64 vcc, v32, s16
 v_cndmask_b32_e32 v32, v32, v15, vcc
v_bfe_u32 v14, v48, 16, 16
 v_add_u32_e32 v33, v33, v14
 v_sub_u32_e64 v15, v33, s16
 v_cmp_ge_u32_e64 vcc, v33, s16
 v_cndmask_b32_e32 v33, v33, v15, vcc
v_bfe_u32 v14, v49, 0, 16
 v_add_u32_e32 v34, v34, v14
 v_sub_u32_e64 v15, v34, s16
 v_cmp_ge_u32_e64 vcc, v34, s16
 v_cndmask_b32_e32 v34, v34, v15, vcc
v_bfe_u32 v14, v49, 16, 16
 v_add_u32_e32 v35, v35, v14
 v_sub_u32_e64 v15, v35, s16
 v_cmp_ge_u32_e64 vcc, v35, s16
 v_cndmask_b32_e32 v35, v35, v15, vcc
v_lshlrev_b32_e32 v33, 16, v33
 v_or_b32_e32 v36, v32, v33
 v_lshlrev_b32_e32 v35, 16, v35
 v_or_b32_e32 v37, v34, v35
 buffer_store_dwordx2 v[36:37], v12, s[8:11], 0 offen
// group 34, AGPR 136..139, row+64, col+32
v_add_u32_e32 v10, 64, v8
 v_add_u32_e32 v11, 32, v9
 v_mul_lo_u32 v12, s21, v11
 v_add_u32_e32 v12, v12, v10
 v_lshlrev_b32_e32 v12, 1, v12
 v_mov_b32_e32 v48, 0
 v_mov_b32_e32 v49, 0
 s_cmp_eq_u32 s20, 0
 s_cbranch_scc1 no_old_34
 buffer_load_dwordx2 v[48:49], v12, s[8:11], 0 offen
no_old_34:
v_accvgpr_read_b32 v16, a136
v_accvgpr_read_b32 v17, a137
v_accvgpr_read_b32 v18, a138
v_accvgpr_read_b32 v19, a139
s_nop 1
s_or_b32 s28, s24, s25
 s_cmp_eq_u32 s28, 0
 s_cbranch_scc1 no_shadow_34
 v_mul_lo_u32 v13, s21, v10
 v_add_u32_e32 v13, v13, v11
 v_lshlrev_b32_e32 v13, 2, v13
 s_lshl_b32 s28, s21, 2
buffer_store_dword v16, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v17, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v18, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v19, v13, s[24:27], 0 offen
no_shadow_34:
v_trunc_f32_e32 v14, v16
 v_cmp_eq_f32_e32 vcc, v14, v16
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v16
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v16
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v32, v[28:29]
v_trunc_f32_e32 v14, v17
 v_cmp_eq_f32_e32 vcc, v14, v17
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v17
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v17
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v33, v[28:29]
v_trunc_f32_e32 v14, v18
 v_cmp_eq_f32_e32 vcc, v14, v18
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v18
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v18
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v34, v[28:29]
v_trunc_f32_e32 v14, v19
 v_cmp_eq_f32_e32 vcc, v14, v19
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v19
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v19
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v35, v[28:29]
s_waitcnt vmcnt(0)
v_bfe_u32 v14, v48, 0, 16
 v_add_u32_e32 v32, v32, v14
 v_sub_u32_e64 v15, v32, s16
 v_cmp_ge_u32_e64 vcc, v32, s16
 v_cndmask_b32_e32 v32, v32, v15, vcc
v_bfe_u32 v14, v48, 16, 16
 v_add_u32_e32 v33, v33, v14
 v_sub_u32_e64 v15, v33, s16
 v_cmp_ge_u32_e64 vcc, v33, s16
 v_cndmask_b32_e32 v33, v33, v15, vcc
v_bfe_u32 v14, v49, 0, 16
 v_add_u32_e32 v34, v34, v14
 v_sub_u32_e64 v15, v34, s16
 v_cmp_ge_u32_e64 vcc, v34, s16
 v_cndmask_b32_e32 v34, v34, v15, vcc
v_bfe_u32 v14, v49, 16, 16
 v_add_u32_e32 v35, v35, v14
 v_sub_u32_e64 v15, v35, s16
 v_cmp_ge_u32_e64 vcc, v35, s16
 v_cndmask_b32_e32 v35, v35, v15, vcc
v_lshlrev_b32_e32 v33, 16, v33
 v_or_b32_e32 v36, v32, v33
 v_lshlrev_b32_e32 v35, 16, v35
 v_or_b32_e32 v37, v34, v35
 buffer_store_dwordx2 v[36:37], v12, s[8:11], 0 offen
// group 35, AGPR 140..143, row+64, col+48
v_add_u32_e32 v10, 64, v8
 v_add_u32_e32 v11, 48, v9
 v_mul_lo_u32 v12, s21, v11
 v_add_u32_e32 v12, v12, v10
 v_lshlrev_b32_e32 v12, 1, v12
 v_mov_b32_e32 v48, 0
 v_mov_b32_e32 v49, 0
 s_cmp_eq_u32 s20, 0
 s_cbranch_scc1 no_old_35
 buffer_load_dwordx2 v[48:49], v12, s[8:11], 0 offen
no_old_35:
v_accvgpr_read_b32 v16, a140
v_accvgpr_read_b32 v17, a141
v_accvgpr_read_b32 v18, a142
v_accvgpr_read_b32 v19, a143
s_nop 1
s_or_b32 s28, s24, s25
 s_cmp_eq_u32 s28, 0
 s_cbranch_scc1 no_shadow_35
 v_mul_lo_u32 v13, s21, v10
 v_add_u32_e32 v13, v13, v11
 v_lshlrev_b32_e32 v13, 2, v13
 s_lshl_b32 s28, s21, 2
buffer_store_dword v16, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v17, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v18, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v19, v13, s[24:27], 0 offen
no_shadow_35:
v_trunc_f32_e32 v14, v16
 v_cmp_eq_f32_e32 vcc, v14, v16
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v16
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v16
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v32, v[28:29]
v_trunc_f32_e32 v14, v17
 v_cmp_eq_f32_e32 vcc, v14, v17
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v17
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v17
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v33, v[28:29]
v_trunc_f32_e32 v14, v18
 v_cmp_eq_f32_e32 vcc, v14, v18
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v18
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v18
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v34, v[28:29]
v_trunc_f32_e32 v14, v19
 v_cmp_eq_f32_e32 vcc, v14, v19
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v19
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v19
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v35, v[28:29]
s_waitcnt vmcnt(0)
v_bfe_u32 v14, v48, 0, 16
 v_add_u32_e32 v32, v32, v14
 v_sub_u32_e64 v15, v32, s16
 v_cmp_ge_u32_e64 vcc, v32, s16
 v_cndmask_b32_e32 v32, v32, v15, vcc
v_bfe_u32 v14, v48, 16, 16
 v_add_u32_e32 v33, v33, v14
 v_sub_u32_e64 v15, v33, s16
 v_cmp_ge_u32_e64 vcc, v33, s16
 v_cndmask_b32_e32 v33, v33, v15, vcc
v_bfe_u32 v14, v49, 0, 16
 v_add_u32_e32 v34, v34, v14
 v_sub_u32_e64 v15, v34, s16
 v_cmp_ge_u32_e64 vcc, v34, s16
 v_cndmask_b32_e32 v34, v34, v15, vcc
v_bfe_u32 v14, v49, 16, 16
 v_add_u32_e32 v35, v35, v14
 v_sub_u32_e64 v15, v35, s16
 v_cmp_ge_u32_e64 vcc, v35, s16
 v_cndmask_b32_e32 v35, v35, v15, vcc
v_lshlrev_b32_e32 v33, 16, v33
 v_or_b32_e32 v36, v32, v33
 v_lshlrev_b32_e32 v35, 16, v35
 v_or_b32_e32 v37, v34, v35
 buffer_store_dwordx2 v[36:37], v12, s[8:11], 0 offen
// group 36, AGPR 144..147, row+64, col+64
v_add_u32_e32 v10, 64, v8
 v_add_u32_e32 v11, 64, v9
 v_mul_lo_u32 v12, s21, v11
 v_add_u32_e32 v12, v12, v10
 v_lshlrev_b32_e32 v12, 1, v12
 v_mov_b32_e32 v48, 0
 v_mov_b32_e32 v49, 0
 s_cmp_eq_u32 s20, 0
 s_cbranch_scc1 no_old_36
 buffer_load_dwordx2 v[48:49], v12, s[8:11], 0 offen
no_old_36:
v_accvgpr_read_b32 v16, a144
v_accvgpr_read_b32 v17, a145
v_accvgpr_read_b32 v18, a146
v_accvgpr_read_b32 v19, a147
s_nop 1
s_or_b32 s28, s24, s25
 s_cmp_eq_u32 s28, 0
 s_cbranch_scc1 no_shadow_36
 v_mul_lo_u32 v13, s21, v10
 v_add_u32_e32 v13, v13, v11
 v_lshlrev_b32_e32 v13, 2, v13
 s_lshl_b32 s28, s21, 2
buffer_store_dword v16, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v17, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v18, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v19, v13, s[24:27], 0 offen
no_shadow_36:
v_trunc_f32_e32 v14, v16
 v_cmp_eq_f32_e32 vcc, v14, v16
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v16
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v16
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v32, v[28:29]
v_trunc_f32_e32 v14, v17
 v_cmp_eq_f32_e32 vcc, v14, v17
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v17
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v17
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v33, v[28:29]
v_trunc_f32_e32 v14, v18
 v_cmp_eq_f32_e32 vcc, v14, v18
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v18
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v18
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v34, v[28:29]
v_trunc_f32_e32 v14, v19
 v_cmp_eq_f32_e32 vcc, v14, v19
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v19
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v19
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v35, v[28:29]
s_waitcnt vmcnt(0)
v_bfe_u32 v14, v48, 0, 16
 v_add_u32_e32 v32, v32, v14
 v_sub_u32_e64 v15, v32, s16
 v_cmp_ge_u32_e64 vcc, v32, s16
 v_cndmask_b32_e32 v32, v32, v15, vcc
v_bfe_u32 v14, v48, 16, 16
 v_add_u32_e32 v33, v33, v14
 v_sub_u32_e64 v15, v33, s16
 v_cmp_ge_u32_e64 vcc, v33, s16
 v_cndmask_b32_e32 v33, v33, v15, vcc
v_bfe_u32 v14, v49, 0, 16
 v_add_u32_e32 v34, v34, v14
 v_sub_u32_e64 v15, v34, s16
 v_cmp_ge_u32_e64 vcc, v34, s16
 v_cndmask_b32_e32 v34, v34, v15, vcc
v_bfe_u32 v14, v49, 16, 16
 v_add_u32_e32 v35, v35, v14
 v_sub_u32_e64 v15, v35, s16
 v_cmp_ge_u32_e64 vcc, v35, s16
 v_cndmask_b32_e32 v35, v35, v15, vcc
v_lshlrev_b32_e32 v33, 16, v33
 v_or_b32_e32 v36, v32, v33
 v_lshlrev_b32_e32 v35, 16, v35
 v_or_b32_e32 v37, v34, v35
 buffer_store_dwordx2 v[36:37], v12, s[8:11], 0 offen
// group 37, AGPR 148..151, row+64, col+80
v_add_u32_e32 v10, 64, v8
 v_add_u32_e32 v11, 80, v9
 v_mul_lo_u32 v12, s21, v11
 v_add_u32_e32 v12, v12, v10
 v_lshlrev_b32_e32 v12, 1, v12
 v_mov_b32_e32 v48, 0
 v_mov_b32_e32 v49, 0
 s_cmp_eq_u32 s20, 0
 s_cbranch_scc1 no_old_37
 buffer_load_dwordx2 v[48:49], v12, s[8:11], 0 offen
no_old_37:
v_accvgpr_read_b32 v16, a148
v_accvgpr_read_b32 v17, a149
v_accvgpr_read_b32 v18, a150
v_accvgpr_read_b32 v19, a151
s_nop 1
s_or_b32 s28, s24, s25
 s_cmp_eq_u32 s28, 0
 s_cbranch_scc1 no_shadow_37
 v_mul_lo_u32 v13, s21, v10
 v_add_u32_e32 v13, v13, v11
 v_lshlrev_b32_e32 v13, 2, v13
 s_lshl_b32 s28, s21, 2
buffer_store_dword v16, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v17, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v18, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v19, v13, s[24:27], 0 offen
no_shadow_37:
v_trunc_f32_e32 v14, v16
 v_cmp_eq_f32_e32 vcc, v14, v16
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v16
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v16
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v32, v[28:29]
v_trunc_f32_e32 v14, v17
 v_cmp_eq_f32_e32 vcc, v14, v17
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v17
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v17
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v33, v[28:29]
v_trunc_f32_e32 v14, v18
 v_cmp_eq_f32_e32 vcc, v14, v18
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v18
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v18
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v34, v[28:29]
v_trunc_f32_e32 v14, v19
 v_cmp_eq_f32_e32 vcc, v14, v19
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v19
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v19
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v35, v[28:29]
s_waitcnt vmcnt(0)
v_bfe_u32 v14, v48, 0, 16
 v_add_u32_e32 v32, v32, v14
 v_sub_u32_e64 v15, v32, s16
 v_cmp_ge_u32_e64 vcc, v32, s16
 v_cndmask_b32_e32 v32, v32, v15, vcc
v_bfe_u32 v14, v48, 16, 16
 v_add_u32_e32 v33, v33, v14
 v_sub_u32_e64 v15, v33, s16
 v_cmp_ge_u32_e64 vcc, v33, s16
 v_cndmask_b32_e32 v33, v33, v15, vcc
v_bfe_u32 v14, v49, 0, 16
 v_add_u32_e32 v34, v34, v14
 v_sub_u32_e64 v15, v34, s16
 v_cmp_ge_u32_e64 vcc, v34, s16
 v_cndmask_b32_e32 v34, v34, v15, vcc
v_bfe_u32 v14, v49, 16, 16
 v_add_u32_e32 v35, v35, v14
 v_sub_u32_e64 v15, v35, s16
 v_cmp_ge_u32_e64 vcc, v35, s16
 v_cndmask_b32_e32 v35, v35, v15, vcc
v_lshlrev_b32_e32 v33, 16, v33
 v_or_b32_e32 v36, v32, v33
 v_lshlrev_b32_e32 v35, 16, v35
 v_or_b32_e32 v37, v34, v35
 buffer_store_dwordx2 v[36:37], v12, s[8:11], 0 offen
// group 38, AGPR 152..155, row+64, col+96
v_add_u32_e32 v10, 64, v8
 v_add_u32_e32 v11, 96, v9
 v_mul_lo_u32 v12, s21, v11
 v_add_u32_e32 v12, v12, v10
 v_lshlrev_b32_e32 v12, 1, v12
 v_mov_b32_e32 v48, 0
 v_mov_b32_e32 v49, 0
 s_cmp_eq_u32 s20, 0
 s_cbranch_scc1 no_old_38
 buffer_load_dwordx2 v[48:49], v12, s[8:11], 0 offen
no_old_38:
v_accvgpr_read_b32 v16, a152
v_accvgpr_read_b32 v17, a153
v_accvgpr_read_b32 v18, a154
v_accvgpr_read_b32 v19, a155
s_nop 1
s_or_b32 s28, s24, s25
 s_cmp_eq_u32 s28, 0
 s_cbranch_scc1 no_shadow_38
 v_mul_lo_u32 v13, s21, v10
 v_add_u32_e32 v13, v13, v11
 v_lshlrev_b32_e32 v13, 2, v13
 s_lshl_b32 s28, s21, 2
buffer_store_dword v16, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v17, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v18, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v19, v13, s[24:27], 0 offen
no_shadow_38:
v_trunc_f32_e32 v14, v16
 v_cmp_eq_f32_e32 vcc, v14, v16
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v16
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v16
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v32, v[28:29]
v_trunc_f32_e32 v14, v17
 v_cmp_eq_f32_e32 vcc, v14, v17
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v17
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v17
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v33, v[28:29]
v_trunc_f32_e32 v14, v18
 v_cmp_eq_f32_e32 vcc, v14, v18
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v18
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v18
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v34, v[28:29]
v_trunc_f32_e32 v14, v19
 v_cmp_eq_f32_e32 vcc, v14, v19
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v19
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v19
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v35, v[28:29]
s_waitcnt vmcnt(0)
v_bfe_u32 v14, v48, 0, 16
 v_add_u32_e32 v32, v32, v14
 v_sub_u32_e64 v15, v32, s16
 v_cmp_ge_u32_e64 vcc, v32, s16
 v_cndmask_b32_e32 v32, v32, v15, vcc
v_bfe_u32 v14, v48, 16, 16
 v_add_u32_e32 v33, v33, v14
 v_sub_u32_e64 v15, v33, s16
 v_cmp_ge_u32_e64 vcc, v33, s16
 v_cndmask_b32_e32 v33, v33, v15, vcc
v_bfe_u32 v14, v49, 0, 16
 v_add_u32_e32 v34, v34, v14
 v_sub_u32_e64 v15, v34, s16
 v_cmp_ge_u32_e64 vcc, v34, s16
 v_cndmask_b32_e32 v34, v34, v15, vcc
v_bfe_u32 v14, v49, 16, 16
 v_add_u32_e32 v35, v35, v14
 v_sub_u32_e64 v15, v35, s16
 v_cmp_ge_u32_e64 vcc, v35, s16
 v_cndmask_b32_e32 v35, v35, v15, vcc
v_lshlrev_b32_e32 v33, 16, v33
 v_or_b32_e32 v36, v32, v33
 v_lshlrev_b32_e32 v35, 16, v35
 v_or_b32_e32 v37, v34, v35
 buffer_store_dwordx2 v[36:37], v12, s[8:11], 0 offen
// group 39, AGPR 156..159, row+64, col+112
v_add_u32_e32 v10, 64, v8
 v_add_u32_e32 v11, 112, v9
 v_mul_lo_u32 v12, s21, v11
 v_add_u32_e32 v12, v12, v10
 v_lshlrev_b32_e32 v12, 1, v12
 v_mov_b32_e32 v48, 0
 v_mov_b32_e32 v49, 0
 s_cmp_eq_u32 s20, 0
 s_cbranch_scc1 no_old_39
 buffer_load_dwordx2 v[48:49], v12, s[8:11], 0 offen
no_old_39:
v_accvgpr_read_b32 v16, a156
v_accvgpr_read_b32 v17, a157
v_accvgpr_read_b32 v18, a158
v_accvgpr_read_b32 v19, a159
s_nop 1
s_or_b32 s28, s24, s25
 s_cmp_eq_u32 s28, 0
 s_cbranch_scc1 no_shadow_39
 v_mul_lo_u32 v13, s21, v10
 v_add_u32_e32 v13, v13, v11
 v_lshlrev_b32_e32 v13, 2, v13
 s_lshl_b32 s28, s21, 2
buffer_store_dword v16, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v17, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v18, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v19, v13, s[24:27], 0 offen
no_shadow_39:
v_trunc_f32_e32 v14, v16
 v_cmp_eq_f32_e32 vcc, v14, v16
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v16
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v16
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v32, v[28:29]
v_trunc_f32_e32 v14, v17
 v_cmp_eq_f32_e32 vcc, v14, v17
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v17
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v17
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v33, v[28:29]
v_trunc_f32_e32 v14, v18
 v_cmp_eq_f32_e32 vcc, v14, v18
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v18
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v18
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v34, v[28:29]
v_trunc_f32_e32 v14, v19
 v_cmp_eq_f32_e32 vcc, v14, v19
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v19
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v19
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v35, v[28:29]
s_waitcnt vmcnt(0)
v_bfe_u32 v14, v48, 0, 16
 v_add_u32_e32 v32, v32, v14
 v_sub_u32_e64 v15, v32, s16
 v_cmp_ge_u32_e64 vcc, v32, s16
 v_cndmask_b32_e32 v32, v32, v15, vcc
v_bfe_u32 v14, v48, 16, 16
 v_add_u32_e32 v33, v33, v14
 v_sub_u32_e64 v15, v33, s16
 v_cmp_ge_u32_e64 vcc, v33, s16
 v_cndmask_b32_e32 v33, v33, v15, vcc
v_bfe_u32 v14, v49, 0, 16
 v_add_u32_e32 v34, v34, v14
 v_sub_u32_e64 v15, v34, s16
 v_cmp_ge_u32_e64 vcc, v34, s16
 v_cndmask_b32_e32 v34, v34, v15, vcc
v_bfe_u32 v14, v49, 16, 16
 v_add_u32_e32 v35, v35, v14
 v_sub_u32_e64 v15, v35, s16
 v_cmp_ge_u32_e64 vcc, v35, s16
 v_cndmask_b32_e32 v35, v35, v15, vcc
v_lshlrev_b32_e32 v33, 16, v33
 v_or_b32_e32 v36, v32, v33
 v_lshlrev_b32_e32 v35, 16, v35
 v_or_b32_e32 v37, v34, v35
 buffer_store_dwordx2 v[36:37], v12, s[8:11], 0 offen
// group 40, AGPR 160..163, row+80, col+0
v_add_u32_e32 v10, 80, v8
 v_add_u32_e32 v11, 0, v9
 v_mul_lo_u32 v12, s21, v11
 v_add_u32_e32 v12, v12, v10
 v_lshlrev_b32_e32 v12, 1, v12
 v_mov_b32_e32 v48, 0
 v_mov_b32_e32 v49, 0
 s_cmp_eq_u32 s20, 0
 s_cbranch_scc1 no_old_40
 buffer_load_dwordx2 v[48:49], v12, s[8:11], 0 offen
no_old_40:
v_accvgpr_read_b32 v16, a160
v_accvgpr_read_b32 v17, a161
v_accvgpr_read_b32 v18, a162
v_accvgpr_read_b32 v19, a163
s_nop 1
s_or_b32 s28, s24, s25
 s_cmp_eq_u32 s28, 0
 s_cbranch_scc1 no_shadow_40
 v_mul_lo_u32 v13, s21, v10
 v_add_u32_e32 v13, v13, v11
 v_lshlrev_b32_e32 v13, 2, v13
 s_lshl_b32 s28, s21, 2
buffer_store_dword v16, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v17, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v18, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v19, v13, s[24:27], 0 offen
no_shadow_40:
v_trunc_f32_e32 v14, v16
 v_cmp_eq_f32_e32 vcc, v14, v16
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v16
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v16
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v32, v[28:29]
v_trunc_f32_e32 v14, v17
 v_cmp_eq_f32_e32 vcc, v14, v17
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v17
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v17
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v33, v[28:29]
v_trunc_f32_e32 v14, v18
 v_cmp_eq_f32_e32 vcc, v14, v18
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v18
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v18
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v34, v[28:29]
v_trunc_f32_e32 v14, v19
 v_cmp_eq_f32_e32 vcc, v14, v19
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v19
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v19
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v35, v[28:29]
s_waitcnt vmcnt(0)
v_bfe_u32 v14, v48, 0, 16
 v_add_u32_e32 v32, v32, v14
 v_sub_u32_e64 v15, v32, s16
 v_cmp_ge_u32_e64 vcc, v32, s16
 v_cndmask_b32_e32 v32, v32, v15, vcc
v_bfe_u32 v14, v48, 16, 16
 v_add_u32_e32 v33, v33, v14
 v_sub_u32_e64 v15, v33, s16
 v_cmp_ge_u32_e64 vcc, v33, s16
 v_cndmask_b32_e32 v33, v33, v15, vcc
v_bfe_u32 v14, v49, 0, 16
 v_add_u32_e32 v34, v34, v14
 v_sub_u32_e64 v15, v34, s16
 v_cmp_ge_u32_e64 vcc, v34, s16
 v_cndmask_b32_e32 v34, v34, v15, vcc
v_bfe_u32 v14, v49, 16, 16
 v_add_u32_e32 v35, v35, v14
 v_sub_u32_e64 v15, v35, s16
 v_cmp_ge_u32_e64 vcc, v35, s16
 v_cndmask_b32_e32 v35, v35, v15, vcc
v_lshlrev_b32_e32 v33, 16, v33
 v_or_b32_e32 v36, v32, v33
 v_lshlrev_b32_e32 v35, 16, v35
 v_or_b32_e32 v37, v34, v35
 buffer_store_dwordx2 v[36:37], v12, s[8:11], 0 offen
// group 41, AGPR 164..167, row+80, col+16
v_add_u32_e32 v10, 80, v8
 v_add_u32_e32 v11, 16, v9
 v_mul_lo_u32 v12, s21, v11
 v_add_u32_e32 v12, v12, v10
 v_lshlrev_b32_e32 v12, 1, v12
 v_mov_b32_e32 v48, 0
 v_mov_b32_e32 v49, 0
 s_cmp_eq_u32 s20, 0
 s_cbranch_scc1 no_old_41
 buffer_load_dwordx2 v[48:49], v12, s[8:11], 0 offen
no_old_41:
v_accvgpr_read_b32 v16, a164
v_accvgpr_read_b32 v17, a165
v_accvgpr_read_b32 v18, a166
v_accvgpr_read_b32 v19, a167
s_nop 1
s_or_b32 s28, s24, s25
 s_cmp_eq_u32 s28, 0
 s_cbranch_scc1 no_shadow_41
 v_mul_lo_u32 v13, s21, v10
 v_add_u32_e32 v13, v13, v11
 v_lshlrev_b32_e32 v13, 2, v13
 s_lshl_b32 s28, s21, 2
buffer_store_dword v16, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v17, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v18, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v19, v13, s[24:27], 0 offen
no_shadow_41:
v_trunc_f32_e32 v14, v16
 v_cmp_eq_f32_e32 vcc, v14, v16
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v16
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v16
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v32, v[28:29]
v_trunc_f32_e32 v14, v17
 v_cmp_eq_f32_e32 vcc, v14, v17
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v17
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v17
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v33, v[28:29]
v_trunc_f32_e32 v14, v18
 v_cmp_eq_f32_e32 vcc, v14, v18
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v18
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v18
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v34, v[28:29]
v_trunc_f32_e32 v14, v19
 v_cmp_eq_f32_e32 vcc, v14, v19
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v19
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v19
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v35, v[28:29]
s_waitcnt vmcnt(0)
v_bfe_u32 v14, v48, 0, 16
 v_add_u32_e32 v32, v32, v14
 v_sub_u32_e64 v15, v32, s16
 v_cmp_ge_u32_e64 vcc, v32, s16
 v_cndmask_b32_e32 v32, v32, v15, vcc
v_bfe_u32 v14, v48, 16, 16
 v_add_u32_e32 v33, v33, v14
 v_sub_u32_e64 v15, v33, s16
 v_cmp_ge_u32_e64 vcc, v33, s16
 v_cndmask_b32_e32 v33, v33, v15, vcc
v_bfe_u32 v14, v49, 0, 16
 v_add_u32_e32 v34, v34, v14
 v_sub_u32_e64 v15, v34, s16
 v_cmp_ge_u32_e64 vcc, v34, s16
 v_cndmask_b32_e32 v34, v34, v15, vcc
v_bfe_u32 v14, v49, 16, 16
 v_add_u32_e32 v35, v35, v14
 v_sub_u32_e64 v15, v35, s16
 v_cmp_ge_u32_e64 vcc, v35, s16
 v_cndmask_b32_e32 v35, v35, v15, vcc
v_lshlrev_b32_e32 v33, 16, v33
 v_or_b32_e32 v36, v32, v33
 v_lshlrev_b32_e32 v35, 16, v35
 v_or_b32_e32 v37, v34, v35
 buffer_store_dwordx2 v[36:37], v12, s[8:11], 0 offen
// group 42, AGPR 168..171, row+80, col+32
v_add_u32_e32 v10, 80, v8
 v_add_u32_e32 v11, 32, v9
 v_mul_lo_u32 v12, s21, v11
 v_add_u32_e32 v12, v12, v10
 v_lshlrev_b32_e32 v12, 1, v12
 v_mov_b32_e32 v48, 0
 v_mov_b32_e32 v49, 0
 s_cmp_eq_u32 s20, 0
 s_cbranch_scc1 no_old_42
 buffer_load_dwordx2 v[48:49], v12, s[8:11], 0 offen
no_old_42:
v_accvgpr_read_b32 v16, a168
v_accvgpr_read_b32 v17, a169
v_accvgpr_read_b32 v18, a170
v_accvgpr_read_b32 v19, a171
s_nop 1
s_or_b32 s28, s24, s25
 s_cmp_eq_u32 s28, 0
 s_cbranch_scc1 no_shadow_42
 v_mul_lo_u32 v13, s21, v10
 v_add_u32_e32 v13, v13, v11
 v_lshlrev_b32_e32 v13, 2, v13
 s_lshl_b32 s28, s21, 2
buffer_store_dword v16, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v17, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v18, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v19, v13, s[24:27], 0 offen
no_shadow_42:
v_trunc_f32_e32 v14, v16
 v_cmp_eq_f32_e32 vcc, v14, v16
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v16
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v16
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v32, v[28:29]
v_trunc_f32_e32 v14, v17
 v_cmp_eq_f32_e32 vcc, v14, v17
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v17
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v17
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v33, v[28:29]
v_trunc_f32_e32 v14, v18
 v_cmp_eq_f32_e32 vcc, v14, v18
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v18
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v18
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v34, v[28:29]
v_trunc_f32_e32 v14, v19
 v_cmp_eq_f32_e32 vcc, v14, v19
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v19
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v19
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v35, v[28:29]
s_waitcnt vmcnt(0)
v_bfe_u32 v14, v48, 0, 16
 v_add_u32_e32 v32, v32, v14
 v_sub_u32_e64 v15, v32, s16
 v_cmp_ge_u32_e64 vcc, v32, s16
 v_cndmask_b32_e32 v32, v32, v15, vcc
v_bfe_u32 v14, v48, 16, 16
 v_add_u32_e32 v33, v33, v14
 v_sub_u32_e64 v15, v33, s16
 v_cmp_ge_u32_e64 vcc, v33, s16
 v_cndmask_b32_e32 v33, v33, v15, vcc
v_bfe_u32 v14, v49, 0, 16
 v_add_u32_e32 v34, v34, v14
 v_sub_u32_e64 v15, v34, s16
 v_cmp_ge_u32_e64 vcc, v34, s16
 v_cndmask_b32_e32 v34, v34, v15, vcc
v_bfe_u32 v14, v49, 16, 16
 v_add_u32_e32 v35, v35, v14
 v_sub_u32_e64 v15, v35, s16
 v_cmp_ge_u32_e64 vcc, v35, s16
 v_cndmask_b32_e32 v35, v35, v15, vcc
v_lshlrev_b32_e32 v33, 16, v33
 v_or_b32_e32 v36, v32, v33
 v_lshlrev_b32_e32 v35, 16, v35
 v_or_b32_e32 v37, v34, v35
 buffer_store_dwordx2 v[36:37], v12, s[8:11], 0 offen
// group 43, AGPR 172..175, row+80, col+48
v_add_u32_e32 v10, 80, v8
 v_add_u32_e32 v11, 48, v9
 v_mul_lo_u32 v12, s21, v11
 v_add_u32_e32 v12, v12, v10
 v_lshlrev_b32_e32 v12, 1, v12
 v_mov_b32_e32 v48, 0
 v_mov_b32_e32 v49, 0
 s_cmp_eq_u32 s20, 0
 s_cbranch_scc1 no_old_43
 buffer_load_dwordx2 v[48:49], v12, s[8:11], 0 offen
no_old_43:
v_accvgpr_read_b32 v16, a172
v_accvgpr_read_b32 v17, a173
v_accvgpr_read_b32 v18, a174
v_accvgpr_read_b32 v19, a175
s_nop 1
s_or_b32 s28, s24, s25
 s_cmp_eq_u32 s28, 0
 s_cbranch_scc1 no_shadow_43
 v_mul_lo_u32 v13, s21, v10
 v_add_u32_e32 v13, v13, v11
 v_lshlrev_b32_e32 v13, 2, v13
 s_lshl_b32 s28, s21, 2
buffer_store_dword v16, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v17, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v18, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v19, v13, s[24:27], 0 offen
no_shadow_43:
v_trunc_f32_e32 v14, v16
 v_cmp_eq_f32_e32 vcc, v14, v16
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v16
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v16
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v32, v[28:29]
v_trunc_f32_e32 v14, v17
 v_cmp_eq_f32_e32 vcc, v14, v17
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v17
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v17
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v33, v[28:29]
v_trunc_f32_e32 v14, v18
 v_cmp_eq_f32_e32 vcc, v14, v18
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v18
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v18
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v34, v[28:29]
v_trunc_f32_e32 v14, v19
 v_cmp_eq_f32_e32 vcc, v14, v19
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v19
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v19
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v35, v[28:29]
s_waitcnt vmcnt(0)
v_bfe_u32 v14, v48, 0, 16
 v_add_u32_e32 v32, v32, v14
 v_sub_u32_e64 v15, v32, s16
 v_cmp_ge_u32_e64 vcc, v32, s16
 v_cndmask_b32_e32 v32, v32, v15, vcc
v_bfe_u32 v14, v48, 16, 16
 v_add_u32_e32 v33, v33, v14
 v_sub_u32_e64 v15, v33, s16
 v_cmp_ge_u32_e64 vcc, v33, s16
 v_cndmask_b32_e32 v33, v33, v15, vcc
v_bfe_u32 v14, v49, 0, 16
 v_add_u32_e32 v34, v34, v14
 v_sub_u32_e64 v15, v34, s16
 v_cmp_ge_u32_e64 vcc, v34, s16
 v_cndmask_b32_e32 v34, v34, v15, vcc
v_bfe_u32 v14, v49, 16, 16
 v_add_u32_e32 v35, v35, v14
 v_sub_u32_e64 v15, v35, s16
 v_cmp_ge_u32_e64 vcc, v35, s16
 v_cndmask_b32_e32 v35, v35, v15, vcc
v_lshlrev_b32_e32 v33, 16, v33
 v_or_b32_e32 v36, v32, v33
 v_lshlrev_b32_e32 v35, 16, v35
 v_or_b32_e32 v37, v34, v35
 buffer_store_dwordx2 v[36:37], v12, s[8:11], 0 offen
// group 44, AGPR 176..179, row+80, col+64
v_add_u32_e32 v10, 80, v8
 v_add_u32_e32 v11, 64, v9
 v_mul_lo_u32 v12, s21, v11
 v_add_u32_e32 v12, v12, v10
 v_lshlrev_b32_e32 v12, 1, v12
 v_mov_b32_e32 v48, 0
 v_mov_b32_e32 v49, 0
 s_cmp_eq_u32 s20, 0
 s_cbranch_scc1 no_old_44
 buffer_load_dwordx2 v[48:49], v12, s[8:11], 0 offen
no_old_44:
v_accvgpr_read_b32 v16, a176
v_accvgpr_read_b32 v17, a177
v_accvgpr_read_b32 v18, a178
v_accvgpr_read_b32 v19, a179
s_nop 1
s_or_b32 s28, s24, s25
 s_cmp_eq_u32 s28, 0
 s_cbranch_scc1 no_shadow_44
 v_mul_lo_u32 v13, s21, v10
 v_add_u32_e32 v13, v13, v11
 v_lshlrev_b32_e32 v13, 2, v13
 s_lshl_b32 s28, s21, 2
buffer_store_dword v16, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v17, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v18, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v19, v13, s[24:27], 0 offen
no_shadow_44:
v_trunc_f32_e32 v14, v16
 v_cmp_eq_f32_e32 vcc, v14, v16
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v16
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v16
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v32, v[28:29]
v_trunc_f32_e32 v14, v17
 v_cmp_eq_f32_e32 vcc, v14, v17
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v17
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v17
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v33, v[28:29]
v_trunc_f32_e32 v14, v18
 v_cmp_eq_f32_e32 vcc, v14, v18
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v18
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v18
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v34, v[28:29]
v_trunc_f32_e32 v14, v19
 v_cmp_eq_f32_e32 vcc, v14, v19
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v19
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v19
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v35, v[28:29]
s_waitcnt vmcnt(0)
v_bfe_u32 v14, v48, 0, 16
 v_add_u32_e32 v32, v32, v14
 v_sub_u32_e64 v15, v32, s16
 v_cmp_ge_u32_e64 vcc, v32, s16
 v_cndmask_b32_e32 v32, v32, v15, vcc
v_bfe_u32 v14, v48, 16, 16
 v_add_u32_e32 v33, v33, v14
 v_sub_u32_e64 v15, v33, s16
 v_cmp_ge_u32_e64 vcc, v33, s16
 v_cndmask_b32_e32 v33, v33, v15, vcc
v_bfe_u32 v14, v49, 0, 16
 v_add_u32_e32 v34, v34, v14
 v_sub_u32_e64 v15, v34, s16
 v_cmp_ge_u32_e64 vcc, v34, s16
 v_cndmask_b32_e32 v34, v34, v15, vcc
v_bfe_u32 v14, v49, 16, 16
 v_add_u32_e32 v35, v35, v14
 v_sub_u32_e64 v15, v35, s16
 v_cmp_ge_u32_e64 vcc, v35, s16
 v_cndmask_b32_e32 v35, v35, v15, vcc
v_lshlrev_b32_e32 v33, 16, v33
 v_or_b32_e32 v36, v32, v33
 v_lshlrev_b32_e32 v35, 16, v35
 v_or_b32_e32 v37, v34, v35
 buffer_store_dwordx2 v[36:37], v12, s[8:11], 0 offen
// group 45, AGPR 180..183, row+80, col+80
v_add_u32_e32 v10, 80, v8
 v_add_u32_e32 v11, 80, v9
 v_mul_lo_u32 v12, s21, v11
 v_add_u32_e32 v12, v12, v10
 v_lshlrev_b32_e32 v12, 1, v12
 v_mov_b32_e32 v48, 0
 v_mov_b32_e32 v49, 0
 s_cmp_eq_u32 s20, 0
 s_cbranch_scc1 no_old_45
 buffer_load_dwordx2 v[48:49], v12, s[8:11], 0 offen
no_old_45:
v_accvgpr_read_b32 v16, a180
v_accvgpr_read_b32 v17, a181
v_accvgpr_read_b32 v18, a182
v_accvgpr_read_b32 v19, a183
s_nop 1
s_or_b32 s28, s24, s25
 s_cmp_eq_u32 s28, 0
 s_cbranch_scc1 no_shadow_45
 v_mul_lo_u32 v13, s21, v10
 v_add_u32_e32 v13, v13, v11
 v_lshlrev_b32_e32 v13, 2, v13
 s_lshl_b32 s28, s21, 2
buffer_store_dword v16, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v17, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v18, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v19, v13, s[24:27], 0 offen
no_shadow_45:
v_trunc_f32_e32 v14, v16
 v_cmp_eq_f32_e32 vcc, v14, v16
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v16
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v16
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v32, v[28:29]
v_trunc_f32_e32 v14, v17
 v_cmp_eq_f32_e32 vcc, v14, v17
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v17
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v17
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v33, v[28:29]
v_trunc_f32_e32 v14, v18
 v_cmp_eq_f32_e32 vcc, v14, v18
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v18
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v18
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v34, v[28:29]
v_trunc_f32_e32 v14, v19
 v_cmp_eq_f32_e32 vcc, v14, v19
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v19
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v19
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v35, v[28:29]
s_waitcnt vmcnt(0)
v_bfe_u32 v14, v48, 0, 16
 v_add_u32_e32 v32, v32, v14
 v_sub_u32_e64 v15, v32, s16
 v_cmp_ge_u32_e64 vcc, v32, s16
 v_cndmask_b32_e32 v32, v32, v15, vcc
v_bfe_u32 v14, v48, 16, 16
 v_add_u32_e32 v33, v33, v14
 v_sub_u32_e64 v15, v33, s16
 v_cmp_ge_u32_e64 vcc, v33, s16
 v_cndmask_b32_e32 v33, v33, v15, vcc
v_bfe_u32 v14, v49, 0, 16
 v_add_u32_e32 v34, v34, v14
 v_sub_u32_e64 v15, v34, s16
 v_cmp_ge_u32_e64 vcc, v34, s16
 v_cndmask_b32_e32 v34, v34, v15, vcc
v_bfe_u32 v14, v49, 16, 16
 v_add_u32_e32 v35, v35, v14
 v_sub_u32_e64 v15, v35, s16
 v_cmp_ge_u32_e64 vcc, v35, s16
 v_cndmask_b32_e32 v35, v35, v15, vcc
v_lshlrev_b32_e32 v33, 16, v33
 v_or_b32_e32 v36, v32, v33
 v_lshlrev_b32_e32 v35, 16, v35
 v_or_b32_e32 v37, v34, v35
 buffer_store_dwordx2 v[36:37], v12, s[8:11], 0 offen
// group 46, AGPR 184..187, row+80, col+96
v_add_u32_e32 v10, 80, v8
 v_add_u32_e32 v11, 96, v9
 v_mul_lo_u32 v12, s21, v11
 v_add_u32_e32 v12, v12, v10
 v_lshlrev_b32_e32 v12, 1, v12
 v_mov_b32_e32 v48, 0
 v_mov_b32_e32 v49, 0
 s_cmp_eq_u32 s20, 0
 s_cbranch_scc1 no_old_46
 buffer_load_dwordx2 v[48:49], v12, s[8:11], 0 offen
no_old_46:
v_accvgpr_read_b32 v16, a184
v_accvgpr_read_b32 v17, a185
v_accvgpr_read_b32 v18, a186
v_accvgpr_read_b32 v19, a187
s_nop 1
s_or_b32 s28, s24, s25
 s_cmp_eq_u32 s28, 0
 s_cbranch_scc1 no_shadow_46
 v_mul_lo_u32 v13, s21, v10
 v_add_u32_e32 v13, v13, v11
 v_lshlrev_b32_e32 v13, 2, v13
 s_lshl_b32 s28, s21, 2
buffer_store_dword v16, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v17, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v18, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v19, v13, s[24:27], 0 offen
no_shadow_46:
v_trunc_f32_e32 v14, v16
 v_cmp_eq_f32_e32 vcc, v14, v16
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v16
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v16
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v32, v[28:29]
v_trunc_f32_e32 v14, v17
 v_cmp_eq_f32_e32 vcc, v14, v17
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v17
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v17
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v33, v[28:29]
v_trunc_f32_e32 v14, v18
 v_cmp_eq_f32_e32 vcc, v14, v18
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v18
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v18
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v34, v[28:29]
v_trunc_f32_e32 v14, v19
 v_cmp_eq_f32_e32 vcc, v14, v19
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v19
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v19
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v35, v[28:29]
s_waitcnt vmcnt(0)
v_bfe_u32 v14, v48, 0, 16
 v_add_u32_e32 v32, v32, v14
 v_sub_u32_e64 v15, v32, s16
 v_cmp_ge_u32_e64 vcc, v32, s16
 v_cndmask_b32_e32 v32, v32, v15, vcc
v_bfe_u32 v14, v48, 16, 16
 v_add_u32_e32 v33, v33, v14
 v_sub_u32_e64 v15, v33, s16
 v_cmp_ge_u32_e64 vcc, v33, s16
 v_cndmask_b32_e32 v33, v33, v15, vcc
v_bfe_u32 v14, v49, 0, 16
 v_add_u32_e32 v34, v34, v14
 v_sub_u32_e64 v15, v34, s16
 v_cmp_ge_u32_e64 vcc, v34, s16
 v_cndmask_b32_e32 v34, v34, v15, vcc
v_bfe_u32 v14, v49, 16, 16
 v_add_u32_e32 v35, v35, v14
 v_sub_u32_e64 v15, v35, s16
 v_cmp_ge_u32_e64 vcc, v35, s16
 v_cndmask_b32_e32 v35, v35, v15, vcc
v_lshlrev_b32_e32 v33, 16, v33
 v_or_b32_e32 v36, v32, v33
 v_lshlrev_b32_e32 v35, 16, v35
 v_or_b32_e32 v37, v34, v35
 buffer_store_dwordx2 v[36:37], v12, s[8:11], 0 offen
// group 47, AGPR 188..191, row+80, col+112
v_add_u32_e32 v10, 80, v8
 v_add_u32_e32 v11, 112, v9
 v_mul_lo_u32 v12, s21, v11
 v_add_u32_e32 v12, v12, v10
 v_lshlrev_b32_e32 v12, 1, v12
 v_mov_b32_e32 v48, 0
 v_mov_b32_e32 v49, 0
 s_cmp_eq_u32 s20, 0
 s_cbranch_scc1 no_old_47
 buffer_load_dwordx2 v[48:49], v12, s[8:11], 0 offen
no_old_47:
v_accvgpr_read_b32 v16, a188
v_accvgpr_read_b32 v17, a189
v_accvgpr_read_b32 v18, a190
v_accvgpr_read_b32 v19, a191
s_nop 1
s_or_b32 s28, s24, s25
 s_cmp_eq_u32 s28, 0
 s_cbranch_scc1 no_shadow_47
 v_mul_lo_u32 v13, s21, v10
 v_add_u32_e32 v13, v13, v11
 v_lshlrev_b32_e32 v13, 2, v13
 s_lshl_b32 s28, s21, 2
buffer_store_dword v16, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v17, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v18, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v19, v13, s[24:27], 0 offen
no_shadow_47:
v_trunc_f32_e32 v14, v16
 v_cmp_eq_f32_e32 vcc, v14, v16
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v16
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v16
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v32, v[28:29]
v_trunc_f32_e32 v14, v17
 v_cmp_eq_f32_e32 vcc, v14, v17
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v17
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v17
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v33, v[28:29]
v_trunc_f32_e32 v14, v18
 v_cmp_eq_f32_e32 vcc, v14, v18
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v18
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v18
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v34, v[28:29]
v_trunc_f32_e32 v14, v19
 v_cmp_eq_f32_e32 vcc, v14, v19
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v19
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v19
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v35, v[28:29]
s_waitcnt vmcnt(0)
v_bfe_u32 v14, v48, 0, 16
 v_add_u32_e32 v32, v32, v14
 v_sub_u32_e64 v15, v32, s16
 v_cmp_ge_u32_e64 vcc, v32, s16
 v_cndmask_b32_e32 v32, v32, v15, vcc
v_bfe_u32 v14, v48, 16, 16
 v_add_u32_e32 v33, v33, v14
 v_sub_u32_e64 v15, v33, s16
 v_cmp_ge_u32_e64 vcc, v33, s16
 v_cndmask_b32_e32 v33, v33, v15, vcc
v_bfe_u32 v14, v49, 0, 16
 v_add_u32_e32 v34, v34, v14
 v_sub_u32_e64 v15, v34, s16
 v_cmp_ge_u32_e64 vcc, v34, s16
 v_cndmask_b32_e32 v34, v34, v15, vcc
v_bfe_u32 v14, v49, 16, 16
 v_add_u32_e32 v35, v35, v14
 v_sub_u32_e64 v15, v35, s16
 v_cmp_ge_u32_e64 vcc, v35, s16
 v_cndmask_b32_e32 v35, v35, v15, vcc
v_lshlrev_b32_e32 v33, 16, v33
 v_or_b32_e32 v36, v32, v33
 v_lshlrev_b32_e32 v35, 16, v35
 v_or_b32_e32 v37, v34, v35
 buffer_store_dwordx2 v[36:37], v12, s[8:11], 0 offen
// group 48, AGPR 192..195, row+96, col+0
v_add_u32_e32 v10, 96, v8
 v_add_u32_e32 v11, 0, v9
 v_mul_lo_u32 v12, s21, v11
 v_add_u32_e32 v12, v12, v10
 v_lshlrev_b32_e32 v12, 1, v12
 v_mov_b32_e32 v48, 0
 v_mov_b32_e32 v49, 0
 s_cmp_eq_u32 s20, 0
 s_cbranch_scc1 no_old_48
 buffer_load_dwordx2 v[48:49], v12, s[8:11], 0 offen
no_old_48:
v_accvgpr_read_b32 v16, a192
v_accvgpr_read_b32 v17, a193
v_accvgpr_read_b32 v18, a194
v_accvgpr_read_b32 v19, a195
s_nop 1
s_or_b32 s28, s24, s25
 s_cmp_eq_u32 s28, 0
 s_cbranch_scc1 no_shadow_48
 v_mul_lo_u32 v13, s21, v10
 v_add_u32_e32 v13, v13, v11
 v_lshlrev_b32_e32 v13, 2, v13
 s_lshl_b32 s28, s21, 2
buffer_store_dword v16, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v17, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v18, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v19, v13, s[24:27], 0 offen
no_shadow_48:
v_trunc_f32_e32 v14, v16
 v_cmp_eq_f32_e32 vcc, v14, v16
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v16
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v16
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v32, v[28:29]
v_trunc_f32_e32 v14, v17
 v_cmp_eq_f32_e32 vcc, v14, v17
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v17
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v17
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v33, v[28:29]
v_trunc_f32_e32 v14, v18
 v_cmp_eq_f32_e32 vcc, v14, v18
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v18
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v18
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v34, v[28:29]
v_trunc_f32_e32 v14, v19
 v_cmp_eq_f32_e32 vcc, v14, v19
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v19
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v19
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v35, v[28:29]
s_waitcnt vmcnt(0)
v_bfe_u32 v14, v48, 0, 16
 v_add_u32_e32 v32, v32, v14
 v_sub_u32_e64 v15, v32, s16
 v_cmp_ge_u32_e64 vcc, v32, s16
 v_cndmask_b32_e32 v32, v32, v15, vcc
v_bfe_u32 v14, v48, 16, 16
 v_add_u32_e32 v33, v33, v14
 v_sub_u32_e64 v15, v33, s16
 v_cmp_ge_u32_e64 vcc, v33, s16
 v_cndmask_b32_e32 v33, v33, v15, vcc
v_bfe_u32 v14, v49, 0, 16
 v_add_u32_e32 v34, v34, v14
 v_sub_u32_e64 v15, v34, s16
 v_cmp_ge_u32_e64 vcc, v34, s16
 v_cndmask_b32_e32 v34, v34, v15, vcc
v_bfe_u32 v14, v49, 16, 16
 v_add_u32_e32 v35, v35, v14
 v_sub_u32_e64 v15, v35, s16
 v_cmp_ge_u32_e64 vcc, v35, s16
 v_cndmask_b32_e32 v35, v35, v15, vcc
v_lshlrev_b32_e32 v33, 16, v33
 v_or_b32_e32 v36, v32, v33
 v_lshlrev_b32_e32 v35, 16, v35
 v_or_b32_e32 v37, v34, v35
 buffer_store_dwordx2 v[36:37], v12, s[8:11], 0 offen
// group 49, AGPR 196..199, row+96, col+16
v_add_u32_e32 v10, 96, v8
 v_add_u32_e32 v11, 16, v9
 v_mul_lo_u32 v12, s21, v11
 v_add_u32_e32 v12, v12, v10
 v_lshlrev_b32_e32 v12, 1, v12
 v_mov_b32_e32 v48, 0
 v_mov_b32_e32 v49, 0
 s_cmp_eq_u32 s20, 0
 s_cbranch_scc1 no_old_49
 buffer_load_dwordx2 v[48:49], v12, s[8:11], 0 offen
no_old_49:
v_accvgpr_read_b32 v16, a196
v_accvgpr_read_b32 v17, a197
v_accvgpr_read_b32 v18, a198
v_accvgpr_read_b32 v19, a199
s_nop 1
s_or_b32 s28, s24, s25
 s_cmp_eq_u32 s28, 0
 s_cbranch_scc1 no_shadow_49
 v_mul_lo_u32 v13, s21, v10
 v_add_u32_e32 v13, v13, v11
 v_lshlrev_b32_e32 v13, 2, v13
 s_lshl_b32 s28, s21, 2
buffer_store_dword v16, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v17, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v18, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v19, v13, s[24:27], 0 offen
no_shadow_49:
v_trunc_f32_e32 v14, v16
 v_cmp_eq_f32_e32 vcc, v14, v16
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v16
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v16
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v32, v[28:29]
v_trunc_f32_e32 v14, v17
 v_cmp_eq_f32_e32 vcc, v14, v17
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v17
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v17
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v33, v[28:29]
v_trunc_f32_e32 v14, v18
 v_cmp_eq_f32_e32 vcc, v14, v18
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v18
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v18
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v34, v[28:29]
v_trunc_f32_e32 v14, v19
 v_cmp_eq_f32_e32 vcc, v14, v19
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v19
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v19
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v35, v[28:29]
s_waitcnt vmcnt(0)
v_bfe_u32 v14, v48, 0, 16
 v_add_u32_e32 v32, v32, v14
 v_sub_u32_e64 v15, v32, s16
 v_cmp_ge_u32_e64 vcc, v32, s16
 v_cndmask_b32_e32 v32, v32, v15, vcc
v_bfe_u32 v14, v48, 16, 16
 v_add_u32_e32 v33, v33, v14
 v_sub_u32_e64 v15, v33, s16
 v_cmp_ge_u32_e64 vcc, v33, s16
 v_cndmask_b32_e32 v33, v33, v15, vcc
v_bfe_u32 v14, v49, 0, 16
 v_add_u32_e32 v34, v34, v14
 v_sub_u32_e64 v15, v34, s16
 v_cmp_ge_u32_e64 vcc, v34, s16
 v_cndmask_b32_e32 v34, v34, v15, vcc
v_bfe_u32 v14, v49, 16, 16
 v_add_u32_e32 v35, v35, v14
 v_sub_u32_e64 v15, v35, s16
 v_cmp_ge_u32_e64 vcc, v35, s16
 v_cndmask_b32_e32 v35, v35, v15, vcc
v_lshlrev_b32_e32 v33, 16, v33
 v_or_b32_e32 v36, v32, v33
 v_lshlrev_b32_e32 v35, 16, v35
 v_or_b32_e32 v37, v34, v35
 buffer_store_dwordx2 v[36:37], v12, s[8:11], 0 offen
// group 50, AGPR 200..203, row+96, col+32
v_add_u32_e32 v10, 96, v8
 v_add_u32_e32 v11, 32, v9
 v_mul_lo_u32 v12, s21, v11
 v_add_u32_e32 v12, v12, v10
 v_lshlrev_b32_e32 v12, 1, v12
 v_mov_b32_e32 v48, 0
 v_mov_b32_e32 v49, 0
 s_cmp_eq_u32 s20, 0
 s_cbranch_scc1 no_old_50
 buffer_load_dwordx2 v[48:49], v12, s[8:11], 0 offen
no_old_50:
v_accvgpr_read_b32 v16, a200
v_accvgpr_read_b32 v17, a201
v_accvgpr_read_b32 v18, a202
v_accvgpr_read_b32 v19, a203
s_nop 1
s_or_b32 s28, s24, s25
 s_cmp_eq_u32 s28, 0
 s_cbranch_scc1 no_shadow_50
 v_mul_lo_u32 v13, s21, v10
 v_add_u32_e32 v13, v13, v11
 v_lshlrev_b32_e32 v13, 2, v13
 s_lshl_b32 s28, s21, 2
buffer_store_dword v16, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v17, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v18, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v19, v13, s[24:27], 0 offen
no_shadow_50:
v_trunc_f32_e32 v14, v16
 v_cmp_eq_f32_e32 vcc, v14, v16
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v16
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v16
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v32, v[28:29]
v_trunc_f32_e32 v14, v17
 v_cmp_eq_f32_e32 vcc, v14, v17
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v17
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v17
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v33, v[28:29]
v_trunc_f32_e32 v14, v18
 v_cmp_eq_f32_e32 vcc, v14, v18
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v18
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v18
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v34, v[28:29]
v_trunc_f32_e32 v14, v19
 v_cmp_eq_f32_e32 vcc, v14, v19
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v19
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v19
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v35, v[28:29]
s_waitcnt vmcnt(0)
v_bfe_u32 v14, v48, 0, 16
 v_add_u32_e32 v32, v32, v14
 v_sub_u32_e64 v15, v32, s16
 v_cmp_ge_u32_e64 vcc, v32, s16
 v_cndmask_b32_e32 v32, v32, v15, vcc
v_bfe_u32 v14, v48, 16, 16
 v_add_u32_e32 v33, v33, v14
 v_sub_u32_e64 v15, v33, s16
 v_cmp_ge_u32_e64 vcc, v33, s16
 v_cndmask_b32_e32 v33, v33, v15, vcc
v_bfe_u32 v14, v49, 0, 16
 v_add_u32_e32 v34, v34, v14
 v_sub_u32_e64 v15, v34, s16
 v_cmp_ge_u32_e64 vcc, v34, s16
 v_cndmask_b32_e32 v34, v34, v15, vcc
v_bfe_u32 v14, v49, 16, 16
 v_add_u32_e32 v35, v35, v14
 v_sub_u32_e64 v15, v35, s16
 v_cmp_ge_u32_e64 vcc, v35, s16
 v_cndmask_b32_e32 v35, v35, v15, vcc
v_lshlrev_b32_e32 v33, 16, v33
 v_or_b32_e32 v36, v32, v33
 v_lshlrev_b32_e32 v35, 16, v35
 v_or_b32_e32 v37, v34, v35
 buffer_store_dwordx2 v[36:37], v12, s[8:11], 0 offen
// group 51, AGPR 204..207, row+96, col+48
v_add_u32_e32 v10, 96, v8
 v_add_u32_e32 v11, 48, v9
 v_mul_lo_u32 v12, s21, v11
 v_add_u32_e32 v12, v12, v10
 v_lshlrev_b32_e32 v12, 1, v12
 v_mov_b32_e32 v48, 0
 v_mov_b32_e32 v49, 0
 s_cmp_eq_u32 s20, 0
 s_cbranch_scc1 no_old_51
 buffer_load_dwordx2 v[48:49], v12, s[8:11], 0 offen
no_old_51:
v_accvgpr_read_b32 v16, a204
v_accvgpr_read_b32 v17, a205
v_accvgpr_read_b32 v18, a206
v_accvgpr_read_b32 v19, a207
s_nop 1
s_or_b32 s28, s24, s25
 s_cmp_eq_u32 s28, 0
 s_cbranch_scc1 no_shadow_51
 v_mul_lo_u32 v13, s21, v10
 v_add_u32_e32 v13, v13, v11
 v_lshlrev_b32_e32 v13, 2, v13
 s_lshl_b32 s28, s21, 2
buffer_store_dword v16, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v17, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v18, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v19, v13, s[24:27], 0 offen
no_shadow_51:
v_trunc_f32_e32 v14, v16
 v_cmp_eq_f32_e32 vcc, v14, v16
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v16
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v16
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v32, v[28:29]
v_trunc_f32_e32 v14, v17
 v_cmp_eq_f32_e32 vcc, v14, v17
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v17
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v17
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v33, v[28:29]
v_trunc_f32_e32 v14, v18
 v_cmp_eq_f32_e32 vcc, v14, v18
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v18
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v18
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v34, v[28:29]
v_trunc_f32_e32 v14, v19
 v_cmp_eq_f32_e32 vcc, v14, v19
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v19
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v19
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v35, v[28:29]
s_waitcnt vmcnt(0)
v_bfe_u32 v14, v48, 0, 16
 v_add_u32_e32 v32, v32, v14
 v_sub_u32_e64 v15, v32, s16
 v_cmp_ge_u32_e64 vcc, v32, s16
 v_cndmask_b32_e32 v32, v32, v15, vcc
v_bfe_u32 v14, v48, 16, 16
 v_add_u32_e32 v33, v33, v14
 v_sub_u32_e64 v15, v33, s16
 v_cmp_ge_u32_e64 vcc, v33, s16
 v_cndmask_b32_e32 v33, v33, v15, vcc
v_bfe_u32 v14, v49, 0, 16
 v_add_u32_e32 v34, v34, v14
 v_sub_u32_e64 v15, v34, s16
 v_cmp_ge_u32_e64 vcc, v34, s16
 v_cndmask_b32_e32 v34, v34, v15, vcc
v_bfe_u32 v14, v49, 16, 16
 v_add_u32_e32 v35, v35, v14
 v_sub_u32_e64 v15, v35, s16
 v_cmp_ge_u32_e64 vcc, v35, s16
 v_cndmask_b32_e32 v35, v35, v15, vcc
v_lshlrev_b32_e32 v33, 16, v33
 v_or_b32_e32 v36, v32, v33
 v_lshlrev_b32_e32 v35, 16, v35
 v_or_b32_e32 v37, v34, v35
 buffer_store_dwordx2 v[36:37], v12, s[8:11], 0 offen
// group 52, AGPR 208..211, row+96, col+64
v_add_u32_e32 v10, 96, v8
 v_add_u32_e32 v11, 64, v9
 v_mul_lo_u32 v12, s21, v11
 v_add_u32_e32 v12, v12, v10
 v_lshlrev_b32_e32 v12, 1, v12
 v_mov_b32_e32 v48, 0
 v_mov_b32_e32 v49, 0
 s_cmp_eq_u32 s20, 0
 s_cbranch_scc1 no_old_52
 buffer_load_dwordx2 v[48:49], v12, s[8:11], 0 offen
no_old_52:
v_accvgpr_read_b32 v16, a208
v_accvgpr_read_b32 v17, a209
v_accvgpr_read_b32 v18, a210
v_accvgpr_read_b32 v19, a211
s_nop 1
s_or_b32 s28, s24, s25
 s_cmp_eq_u32 s28, 0
 s_cbranch_scc1 no_shadow_52
 v_mul_lo_u32 v13, s21, v10
 v_add_u32_e32 v13, v13, v11
 v_lshlrev_b32_e32 v13, 2, v13
 s_lshl_b32 s28, s21, 2
buffer_store_dword v16, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v17, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v18, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v19, v13, s[24:27], 0 offen
no_shadow_52:
v_trunc_f32_e32 v14, v16
 v_cmp_eq_f32_e32 vcc, v14, v16
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v16
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v16
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v32, v[28:29]
v_trunc_f32_e32 v14, v17
 v_cmp_eq_f32_e32 vcc, v14, v17
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v17
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v17
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v33, v[28:29]
v_trunc_f32_e32 v14, v18
 v_cmp_eq_f32_e32 vcc, v14, v18
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v18
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v18
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v34, v[28:29]
v_trunc_f32_e32 v14, v19
 v_cmp_eq_f32_e32 vcc, v14, v19
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v19
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v19
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v35, v[28:29]
s_waitcnt vmcnt(0)
v_bfe_u32 v14, v48, 0, 16
 v_add_u32_e32 v32, v32, v14
 v_sub_u32_e64 v15, v32, s16
 v_cmp_ge_u32_e64 vcc, v32, s16
 v_cndmask_b32_e32 v32, v32, v15, vcc
v_bfe_u32 v14, v48, 16, 16
 v_add_u32_e32 v33, v33, v14
 v_sub_u32_e64 v15, v33, s16
 v_cmp_ge_u32_e64 vcc, v33, s16
 v_cndmask_b32_e32 v33, v33, v15, vcc
v_bfe_u32 v14, v49, 0, 16
 v_add_u32_e32 v34, v34, v14
 v_sub_u32_e64 v15, v34, s16
 v_cmp_ge_u32_e64 vcc, v34, s16
 v_cndmask_b32_e32 v34, v34, v15, vcc
v_bfe_u32 v14, v49, 16, 16
 v_add_u32_e32 v35, v35, v14
 v_sub_u32_e64 v15, v35, s16
 v_cmp_ge_u32_e64 vcc, v35, s16
 v_cndmask_b32_e32 v35, v35, v15, vcc
v_lshlrev_b32_e32 v33, 16, v33
 v_or_b32_e32 v36, v32, v33
 v_lshlrev_b32_e32 v35, 16, v35
 v_or_b32_e32 v37, v34, v35
 buffer_store_dwordx2 v[36:37], v12, s[8:11], 0 offen
// group 53, AGPR 212..215, row+96, col+80
v_add_u32_e32 v10, 96, v8
 v_add_u32_e32 v11, 80, v9
 v_mul_lo_u32 v12, s21, v11
 v_add_u32_e32 v12, v12, v10
 v_lshlrev_b32_e32 v12, 1, v12
 v_mov_b32_e32 v48, 0
 v_mov_b32_e32 v49, 0
 s_cmp_eq_u32 s20, 0
 s_cbranch_scc1 no_old_53
 buffer_load_dwordx2 v[48:49], v12, s[8:11], 0 offen
no_old_53:
v_accvgpr_read_b32 v16, a212
v_accvgpr_read_b32 v17, a213
v_accvgpr_read_b32 v18, a214
v_accvgpr_read_b32 v19, a215
s_nop 1
s_or_b32 s28, s24, s25
 s_cmp_eq_u32 s28, 0
 s_cbranch_scc1 no_shadow_53
 v_mul_lo_u32 v13, s21, v10
 v_add_u32_e32 v13, v13, v11
 v_lshlrev_b32_e32 v13, 2, v13
 s_lshl_b32 s28, s21, 2
buffer_store_dword v16, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v17, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v18, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v19, v13, s[24:27], 0 offen
no_shadow_53:
v_trunc_f32_e32 v14, v16
 v_cmp_eq_f32_e32 vcc, v14, v16
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v16
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v16
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v32, v[28:29]
v_trunc_f32_e32 v14, v17
 v_cmp_eq_f32_e32 vcc, v14, v17
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v17
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v17
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v33, v[28:29]
v_trunc_f32_e32 v14, v18
 v_cmp_eq_f32_e32 vcc, v14, v18
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v18
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v18
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v34, v[28:29]
v_trunc_f32_e32 v14, v19
 v_cmp_eq_f32_e32 vcc, v14, v19
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v19
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v19
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v35, v[28:29]
s_waitcnt vmcnt(0)
v_bfe_u32 v14, v48, 0, 16
 v_add_u32_e32 v32, v32, v14
 v_sub_u32_e64 v15, v32, s16
 v_cmp_ge_u32_e64 vcc, v32, s16
 v_cndmask_b32_e32 v32, v32, v15, vcc
v_bfe_u32 v14, v48, 16, 16
 v_add_u32_e32 v33, v33, v14
 v_sub_u32_e64 v15, v33, s16
 v_cmp_ge_u32_e64 vcc, v33, s16
 v_cndmask_b32_e32 v33, v33, v15, vcc
v_bfe_u32 v14, v49, 0, 16
 v_add_u32_e32 v34, v34, v14
 v_sub_u32_e64 v15, v34, s16
 v_cmp_ge_u32_e64 vcc, v34, s16
 v_cndmask_b32_e32 v34, v34, v15, vcc
v_bfe_u32 v14, v49, 16, 16
 v_add_u32_e32 v35, v35, v14
 v_sub_u32_e64 v15, v35, s16
 v_cmp_ge_u32_e64 vcc, v35, s16
 v_cndmask_b32_e32 v35, v35, v15, vcc
v_lshlrev_b32_e32 v33, 16, v33
 v_or_b32_e32 v36, v32, v33
 v_lshlrev_b32_e32 v35, 16, v35
 v_or_b32_e32 v37, v34, v35
 buffer_store_dwordx2 v[36:37], v12, s[8:11], 0 offen
// group 54, AGPR 216..219, row+96, col+96
v_add_u32_e32 v10, 96, v8
 v_add_u32_e32 v11, 96, v9
 v_mul_lo_u32 v12, s21, v11
 v_add_u32_e32 v12, v12, v10
 v_lshlrev_b32_e32 v12, 1, v12
 v_mov_b32_e32 v48, 0
 v_mov_b32_e32 v49, 0
 s_cmp_eq_u32 s20, 0
 s_cbranch_scc1 no_old_54
 buffer_load_dwordx2 v[48:49], v12, s[8:11], 0 offen
no_old_54:
v_accvgpr_read_b32 v16, a216
v_accvgpr_read_b32 v17, a217
v_accvgpr_read_b32 v18, a218
v_accvgpr_read_b32 v19, a219
s_nop 1
s_or_b32 s28, s24, s25
 s_cmp_eq_u32 s28, 0
 s_cbranch_scc1 no_shadow_54
 v_mul_lo_u32 v13, s21, v10
 v_add_u32_e32 v13, v13, v11
 v_lshlrev_b32_e32 v13, 2, v13
 s_lshl_b32 s28, s21, 2
buffer_store_dword v16, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v17, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v18, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v19, v13, s[24:27], 0 offen
no_shadow_54:
v_trunc_f32_e32 v14, v16
 v_cmp_eq_f32_e32 vcc, v14, v16
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v16
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v16
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v32, v[28:29]
v_trunc_f32_e32 v14, v17
 v_cmp_eq_f32_e32 vcc, v14, v17
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v17
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v17
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v33, v[28:29]
v_trunc_f32_e32 v14, v18
 v_cmp_eq_f32_e32 vcc, v14, v18
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v18
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v18
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v34, v[28:29]
v_trunc_f32_e32 v14, v19
 v_cmp_eq_f32_e32 vcc, v14, v19
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v19
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v19
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v35, v[28:29]
s_waitcnt vmcnt(0)
v_bfe_u32 v14, v48, 0, 16
 v_add_u32_e32 v32, v32, v14
 v_sub_u32_e64 v15, v32, s16
 v_cmp_ge_u32_e64 vcc, v32, s16
 v_cndmask_b32_e32 v32, v32, v15, vcc
v_bfe_u32 v14, v48, 16, 16
 v_add_u32_e32 v33, v33, v14
 v_sub_u32_e64 v15, v33, s16
 v_cmp_ge_u32_e64 vcc, v33, s16
 v_cndmask_b32_e32 v33, v33, v15, vcc
v_bfe_u32 v14, v49, 0, 16
 v_add_u32_e32 v34, v34, v14
 v_sub_u32_e64 v15, v34, s16
 v_cmp_ge_u32_e64 vcc, v34, s16
 v_cndmask_b32_e32 v34, v34, v15, vcc
v_bfe_u32 v14, v49, 16, 16
 v_add_u32_e32 v35, v35, v14
 v_sub_u32_e64 v15, v35, s16
 v_cmp_ge_u32_e64 vcc, v35, s16
 v_cndmask_b32_e32 v35, v35, v15, vcc
v_lshlrev_b32_e32 v33, 16, v33
 v_or_b32_e32 v36, v32, v33
 v_lshlrev_b32_e32 v35, 16, v35
 v_or_b32_e32 v37, v34, v35
 buffer_store_dwordx2 v[36:37], v12, s[8:11], 0 offen
// group 55, AGPR 220..223, row+96, col+112
v_add_u32_e32 v10, 96, v8
 v_add_u32_e32 v11, 112, v9
 v_mul_lo_u32 v12, s21, v11
 v_add_u32_e32 v12, v12, v10
 v_lshlrev_b32_e32 v12, 1, v12
 v_mov_b32_e32 v48, 0
 v_mov_b32_e32 v49, 0
 s_cmp_eq_u32 s20, 0
 s_cbranch_scc1 no_old_55
 buffer_load_dwordx2 v[48:49], v12, s[8:11], 0 offen
no_old_55:
v_accvgpr_read_b32 v16, a220
v_accvgpr_read_b32 v17, a221
v_accvgpr_read_b32 v18, a222
v_accvgpr_read_b32 v19, a223
s_nop 1
s_or_b32 s28, s24, s25
 s_cmp_eq_u32 s28, 0
 s_cbranch_scc1 no_shadow_55
 v_mul_lo_u32 v13, s21, v10
 v_add_u32_e32 v13, v13, v11
 v_lshlrev_b32_e32 v13, 2, v13
 s_lshl_b32 s28, s21, 2
buffer_store_dword v16, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v17, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v18, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v19, v13, s[24:27], 0 offen
no_shadow_55:
v_trunc_f32_e32 v14, v16
 v_cmp_eq_f32_e32 vcc, v14, v16
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v16
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v16
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v32, v[28:29]
v_trunc_f32_e32 v14, v17
 v_cmp_eq_f32_e32 vcc, v14, v17
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v17
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v17
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v33, v[28:29]
v_trunc_f32_e32 v14, v18
 v_cmp_eq_f32_e32 vcc, v14, v18
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v18
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v18
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v34, v[28:29]
v_trunc_f32_e32 v14, v19
 v_cmp_eq_f32_e32 vcc, v14, v19
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v19
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v19
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v35, v[28:29]
s_waitcnt vmcnt(0)
v_bfe_u32 v14, v48, 0, 16
 v_add_u32_e32 v32, v32, v14
 v_sub_u32_e64 v15, v32, s16
 v_cmp_ge_u32_e64 vcc, v32, s16
 v_cndmask_b32_e32 v32, v32, v15, vcc
v_bfe_u32 v14, v48, 16, 16
 v_add_u32_e32 v33, v33, v14
 v_sub_u32_e64 v15, v33, s16
 v_cmp_ge_u32_e64 vcc, v33, s16
 v_cndmask_b32_e32 v33, v33, v15, vcc
v_bfe_u32 v14, v49, 0, 16
 v_add_u32_e32 v34, v34, v14
 v_sub_u32_e64 v15, v34, s16
 v_cmp_ge_u32_e64 vcc, v34, s16
 v_cndmask_b32_e32 v34, v34, v15, vcc
v_bfe_u32 v14, v49, 16, 16
 v_add_u32_e32 v35, v35, v14
 v_sub_u32_e64 v15, v35, s16
 v_cmp_ge_u32_e64 vcc, v35, s16
 v_cndmask_b32_e32 v35, v35, v15, vcc
v_lshlrev_b32_e32 v33, 16, v33
 v_or_b32_e32 v36, v32, v33
 v_lshlrev_b32_e32 v35, 16, v35
 v_or_b32_e32 v37, v34, v35
 buffer_store_dwordx2 v[36:37], v12, s[8:11], 0 offen
// group 56, AGPR 224..227, row+112, col+0
v_add_u32_e32 v10, 112, v8
 v_add_u32_e32 v11, 0, v9
 v_mul_lo_u32 v12, s21, v11
 v_add_u32_e32 v12, v12, v10
 v_lshlrev_b32_e32 v12, 1, v12
 v_mov_b32_e32 v48, 0
 v_mov_b32_e32 v49, 0
 s_cmp_eq_u32 s20, 0
 s_cbranch_scc1 no_old_56
 buffer_load_dwordx2 v[48:49], v12, s[8:11], 0 offen
no_old_56:
v_accvgpr_read_b32 v16, a224
v_accvgpr_read_b32 v17, a225
v_accvgpr_read_b32 v18, a226
v_accvgpr_read_b32 v19, a227
s_nop 1
s_or_b32 s28, s24, s25
 s_cmp_eq_u32 s28, 0
 s_cbranch_scc1 no_shadow_56
 v_mul_lo_u32 v13, s21, v10
 v_add_u32_e32 v13, v13, v11
 v_lshlrev_b32_e32 v13, 2, v13
 s_lshl_b32 s28, s21, 2
buffer_store_dword v16, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v17, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v18, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v19, v13, s[24:27], 0 offen
no_shadow_56:
v_trunc_f32_e32 v14, v16
 v_cmp_eq_f32_e32 vcc, v14, v16
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v16
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v16
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v32, v[28:29]
v_trunc_f32_e32 v14, v17
 v_cmp_eq_f32_e32 vcc, v14, v17
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v17
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v17
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v33, v[28:29]
v_trunc_f32_e32 v14, v18
 v_cmp_eq_f32_e32 vcc, v14, v18
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v18
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v18
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v34, v[28:29]
v_trunc_f32_e32 v14, v19
 v_cmp_eq_f32_e32 vcc, v14, v19
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v19
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v19
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v35, v[28:29]
s_waitcnt vmcnt(0)
v_bfe_u32 v14, v48, 0, 16
 v_add_u32_e32 v32, v32, v14
 v_sub_u32_e64 v15, v32, s16
 v_cmp_ge_u32_e64 vcc, v32, s16
 v_cndmask_b32_e32 v32, v32, v15, vcc
v_bfe_u32 v14, v48, 16, 16
 v_add_u32_e32 v33, v33, v14
 v_sub_u32_e64 v15, v33, s16
 v_cmp_ge_u32_e64 vcc, v33, s16
 v_cndmask_b32_e32 v33, v33, v15, vcc
v_bfe_u32 v14, v49, 0, 16
 v_add_u32_e32 v34, v34, v14
 v_sub_u32_e64 v15, v34, s16
 v_cmp_ge_u32_e64 vcc, v34, s16
 v_cndmask_b32_e32 v34, v34, v15, vcc
v_bfe_u32 v14, v49, 16, 16
 v_add_u32_e32 v35, v35, v14
 v_sub_u32_e64 v15, v35, s16
 v_cmp_ge_u32_e64 vcc, v35, s16
 v_cndmask_b32_e32 v35, v35, v15, vcc
v_lshlrev_b32_e32 v33, 16, v33
 v_or_b32_e32 v36, v32, v33
 v_lshlrev_b32_e32 v35, 16, v35
 v_or_b32_e32 v37, v34, v35
 buffer_store_dwordx2 v[36:37], v12, s[8:11], 0 offen
// group 57, AGPR 228..231, row+112, col+16
v_add_u32_e32 v10, 112, v8
 v_add_u32_e32 v11, 16, v9
 v_mul_lo_u32 v12, s21, v11
 v_add_u32_e32 v12, v12, v10
 v_lshlrev_b32_e32 v12, 1, v12
 v_mov_b32_e32 v48, 0
 v_mov_b32_e32 v49, 0
 s_cmp_eq_u32 s20, 0
 s_cbranch_scc1 no_old_57
 buffer_load_dwordx2 v[48:49], v12, s[8:11], 0 offen
no_old_57:
v_accvgpr_read_b32 v16, a228
v_accvgpr_read_b32 v17, a229
v_accvgpr_read_b32 v18, a230
v_accvgpr_read_b32 v19, a231
s_nop 1
s_or_b32 s28, s24, s25
 s_cmp_eq_u32 s28, 0
 s_cbranch_scc1 no_shadow_57
 v_mul_lo_u32 v13, s21, v10
 v_add_u32_e32 v13, v13, v11
 v_lshlrev_b32_e32 v13, 2, v13
 s_lshl_b32 s28, s21, 2
buffer_store_dword v16, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v17, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v18, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v19, v13, s[24:27], 0 offen
no_shadow_57:
v_trunc_f32_e32 v14, v16
 v_cmp_eq_f32_e32 vcc, v14, v16
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v16
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v16
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v32, v[28:29]
v_trunc_f32_e32 v14, v17
 v_cmp_eq_f32_e32 vcc, v14, v17
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v17
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v17
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v33, v[28:29]
v_trunc_f32_e32 v14, v18
 v_cmp_eq_f32_e32 vcc, v14, v18
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v18
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v18
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v34, v[28:29]
v_trunc_f32_e32 v14, v19
 v_cmp_eq_f32_e32 vcc, v14, v19
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v19
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v19
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v35, v[28:29]
s_waitcnt vmcnt(0)
v_bfe_u32 v14, v48, 0, 16
 v_add_u32_e32 v32, v32, v14
 v_sub_u32_e64 v15, v32, s16
 v_cmp_ge_u32_e64 vcc, v32, s16
 v_cndmask_b32_e32 v32, v32, v15, vcc
v_bfe_u32 v14, v48, 16, 16
 v_add_u32_e32 v33, v33, v14
 v_sub_u32_e64 v15, v33, s16
 v_cmp_ge_u32_e64 vcc, v33, s16
 v_cndmask_b32_e32 v33, v33, v15, vcc
v_bfe_u32 v14, v49, 0, 16
 v_add_u32_e32 v34, v34, v14
 v_sub_u32_e64 v15, v34, s16
 v_cmp_ge_u32_e64 vcc, v34, s16
 v_cndmask_b32_e32 v34, v34, v15, vcc
v_bfe_u32 v14, v49, 16, 16
 v_add_u32_e32 v35, v35, v14
 v_sub_u32_e64 v15, v35, s16
 v_cmp_ge_u32_e64 vcc, v35, s16
 v_cndmask_b32_e32 v35, v35, v15, vcc
v_lshlrev_b32_e32 v33, 16, v33
 v_or_b32_e32 v36, v32, v33
 v_lshlrev_b32_e32 v35, 16, v35
 v_or_b32_e32 v37, v34, v35
 buffer_store_dwordx2 v[36:37], v12, s[8:11], 0 offen
// group 58, AGPR 232..235, row+112, col+32
v_add_u32_e32 v10, 112, v8
 v_add_u32_e32 v11, 32, v9
 v_mul_lo_u32 v12, s21, v11
 v_add_u32_e32 v12, v12, v10
 v_lshlrev_b32_e32 v12, 1, v12
 v_mov_b32_e32 v48, 0
 v_mov_b32_e32 v49, 0
 s_cmp_eq_u32 s20, 0
 s_cbranch_scc1 no_old_58
 buffer_load_dwordx2 v[48:49], v12, s[8:11], 0 offen
no_old_58:
v_accvgpr_read_b32 v16, a232
v_accvgpr_read_b32 v17, a233
v_accvgpr_read_b32 v18, a234
v_accvgpr_read_b32 v19, a235
s_nop 1
s_or_b32 s28, s24, s25
 s_cmp_eq_u32 s28, 0
 s_cbranch_scc1 no_shadow_58
 v_mul_lo_u32 v13, s21, v10
 v_add_u32_e32 v13, v13, v11
 v_lshlrev_b32_e32 v13, 2, v13
 s_lshl_b32 s28, s21, 2
buffer_store_dword v16, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v17, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v18, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v19, v13, s[24:27], 0 offen
no_shadow_58:
v_trunc_f32_e32 v14, v16
 v_cmp_eq_f32_e32 vcc, v14, v16
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v16
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v16
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v32, v[28:29]
v_trunc_f32_e32 v14, v17
 v_cmp_eq_f32_e32 vcc, v14, v17
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v17
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v17
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v33, v[28:29]
v_trunc_f32_e32 v14, v18
 v_cmp_eq_f32_e32 vcc, v14, v18
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v18
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v18
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v34, v[28:29]
v_trunc_f32_e32 v14, v19
 v_cmp_eq_f32_e32 vcc, v14, v19
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v19
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v19
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v35, v[28:29]
s_waitcnt vmcnt(0)
v_bfe_u32 v14, v48, 0, 16
 v_add_u32_e32 v32, v32, v14
 v_sub_u32_e64 v15, v32, s16
 v_cmp_ge_u32_e64 vcc, v32, s16
 v_cndmask_b32_e32 v32, v32, v15, vcc
v_bfe_u32 v14, v48, 16, 16
 v_add_u32_e32 v33, v33, v14
 v_sub_u32_e64 v15, v33, s16
 v_cmp_ge_u32_e64 vcc, v33, s16
 v_cndmask_b32_e32 v33, v33, v15, vcc
v_bfe_u32 v14, v49, 0, 16
 v_add_u32_e32 v34, v34, v14
 v_sub_u32_e64 v15, v34, s16
 v_cmp_ge_u32_e64 vcc, v34, s16
 v_cndmask_b32_e32 v34, v34, v15, vcc
v_bfe_u32 v14, v49, 16, 16
 v_add_u32_e32 v35, v35, v14
 v_sub_u32_e64 v15, v35, s16
 v_cmp_ge_u32_e64 vcc, v35, s16
 v_cndmask_b32_e32 v35, v35, v15, vcc
v_lshlrev_b32_e32 v33, 16, v33
 v_or_b32_e32 v36, v32, v33
 v_lshlrev_b32_e32 v35, 16, v35
 v_or_b32_e32 v37, v34, v35
 buffer_store_dwordx2 v[36:37], v12, s[8:11], 0 offen
// group 59, AGPR 236..239, row+112, col+48
v_add_u32_e32 v10, 112, v8
 v_add_u32_e32 v11, 48, v9
 v_mul_lo_u32 v12, s21, v11
 v_add_u32_e32 v12, v12, v10
 v_lshlrev_b32_e32 v12, 1, v12
 v_mov_b32_e32 v48, 0
 v_mov_b32_e32 v49, 0
 s_cmp_eq_u32 s20, 0
 s_cbranch_scc1 no_old_59
 buffer_load_dwordx2 v[48:49], v12, s[8:11], 0 offen
no_old_59:
v_accvgpr_read_b32 v16, a236
v_accvgpr_read_b32 v17, a237
v_accvgpr_read_b32 v18, a238
v_accvgpr_read_b32 v19, a239
s_nop 1
s_or_b32 s28, s24, s25
 s_cmp_eq_u32 s28, 0
 s_cbranch_scc1 no_shadow_59
 v_mul_lo_u32 v13, s21, v10
 v_add_u32_e32 v13, v13, v11
 v_lshlrev_b32_e32 v13, 2, v13
 s_lshl_b32 s28, s21, 2
buffer_store_dword v16, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v17, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v18, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v19, v13, s[24:27], 0 offen
no_shadow_59:
v_trunc_f32_e32 v14, v16
 v_cmp_eq_f32_e32 vcc, v14, v16
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v16
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v16
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v32, v[28:29]
v_trunc_f32_e32 v14, v17
 v_cmp_eq_f32_e32 vcc, v14, v17
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v17
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v17
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v33, v[28:29]
v_trunc_f32_e32 v14, v18
 v_cmp_eq_f32_e32 vcc, v14, v18
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v18
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v18
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v34, v[28:29]
v_trunc_f32_e32 v14, v19
 v_cmp_eq_f32_e32 vcc, v14, v19
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v19
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v19
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v35, v[28:29]
s_waitcnt vmcnt(0)
v_bfe_u32 v14, v48, 0, 16
 v_add_u32_e32 v32, v32, v14
 v_sub_u32_e64 v15, v32, s16
 v_cmp_ge_u32_e64 vcc, v32, s16
 v_cndmask_b32_e32 v32, v32, v15, vcc
v_bfe_u32 v14, v48, 16, 16
 v_add_u32_e32 v33, v33, v14
 v_sub_u32_e64 v15, v33, s16
 v_cmp_ge_u32_e64 vcc, v33, s16
 v_cndmask_b32_e32 v33, v33, v15, vcc
v_bfe_u32 v14, v49, 0, 16
 v_add_u32_e32 v34, v34, v14
 v_sub_u32_e64 v15, v34, s16
 v_cmp_ge_u32_e64 vcc, v34, s16
 v_cndmask_b32_e32 v34, v34, v15, vcc
v_bfe_u32 v14, v49, 16, 16
 v_add_u32_e32 v35, v35, v14
 v_sub_u32_e64 v15, v35, s16
 v_cmp_ge_u32_e64 vcc, v35, s16
 v_cndmask_b32_e32 v35, v35, v15, vcc
v_lshlrev_b32_e32 v33, 16, v33
 v_or_b32_e32 v36, v32, v33
 v_lshlrev_b32_e32 v35, 16, v35
 v_or_b32_e32 v37, v34, v35
 buffer_store_dwordx2 v[36:37], v12, s[8:11], 0 offen
// group 60, AGPR 240..243, row+112, col+64
v_add_u32_e32 v10, 112, v8
 v_add_u32_e32 v11, 64, v9
 v_mul_lo_u32 v12, s21, v11
 v_add_u32_e32 v12, v12, v10
 v_lshlrev_b32_e32 v12, 1, v12
 v_mov_b32_e32 v48, 0
 v_mov_b32_e32 v49, 0
 s_cmp_eq_u32 s20, 0
 s_cbranch_scc1 no_old_60
 buffer_load_dwordx2 v[48:49], v12, s[8:11], 0 offen
no_old_60:
v_accvgpr_read_b32 v16, a240
v_accvgpr_read_b32 v17, a241
v_accvgpr_read_b32 v18, a242
v_accvgpr_read_b32 v19, a243
s_nop 1
s_or_b32 s28, s24, s25
 s_cmp_eq_u32 s28, 0
 s_cbranch_scc1 no_shadow_60
 v_mul_lo_u32 v13, s21, v10
 v_add_u32_e32 v13, v13, v11
 v_lshlrev_b32_e32 v13, 2, v13
 s_lshl_b32 s28, s21, 2
buffer_store_dword v16, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v17, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v18, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v19, v13, s[24:27], 0 offen
no_shadow_60:
v_trunc_f32_e32 v14, v16
 v_cmp_eq_f32_e32 vcc, v14, v16
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v16
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v16
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v32, v[28:29]
v_trunc_f32_e32 v14, v17
 v_cmp_eq_f32_e32 vcc, v14, v17
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v17
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v17
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v33, v[28:29]
v_trunc_f32_e32 v14, v18
 v_cmp_eq_f32_e32 vcc, v14, v18
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v18
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v18
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v34, v[28:29]
v_trunc_f32_e32 v14, v19
 v_cmp_eq_f32_e32 vcc, v14, v19
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v19
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v19
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v35, v[28:29]
s_waitcnt vmcnt(0)
v_bfe_u32 v14, v48, 0, 16
 v_add_u32_e32 v32, v32, v14
 v_sub_u32_e64 v15, v32, s16
 v_cmp_ge_u32_e64 vcc, v32, s16
 v_cndmask_b32_e32 v32, v32, v15, vcc
v_bfe_u32 v14, v48, 16, 16
 v_add_u32_e32 v33, v33, v14
 v_sub_u32_e64 v15, v33, s16
 v_cmp_ge_u32_e64 vcc, v33, s16
 v_cndmask_b32_e32 v33, v33, v15, vcc
v_bfe_u32 v14, v49, 0, 16
 v_add_u32_e32 v34, v34, v14
 v_sub_u32_e64 v15, v34, s16
 v_cmp_ge_u32_e64 vcc, v34, s16
 v_cndmask_b32_e32 v34, v34, v15, vcc
v_bfe_u32 v14, v49, 16, 16
 v_add_u32_e32 v35, v35, v14
 v_sub_u32_e64 v15, v35, s16
 v_cmp_ge_u32_e64 vcc, v35, s16
 v_cndmask_b32_e32 v35, v35, v15, vcc
v_lshlrev_b32_e32 v33, 16, v33
 v_or_b32_e32 v36, v32, v33
 v_lshlrev_b32_e32 v35, 16, v35
 v_or_b32_e32 v37, v34, v35
 buffer_store_dwordx2 v[36:37], v12, s[8:11], 0 offen
// group 61, AGPR 244..247, row+112, col+80
v_add_u32_e32 v10, 112, v8
 v_add_u32_e32 v11, 80, v9
 v_mul_lo_u32 v12, s21, v11
 v_add_u32_e32 v12, v12, v10
 v_lshlrev_b32_e32 v12, 1, v12
 v_mov_b32_e32 v48, 0
 v_mov_b32_e32 v49, 0
 s_cmp_eq_u32 s20, 0
 s_cbranch_scc1 no_old_61
 buffer_load_dwordx2 v[48:49], v12, s[8:11], 0 offen
no_old_61:
v_accvgpr_read_b32 v16, a244
v_accvgpr_read_b32 v17, a245
v_accvgpr_read_b32 v18, a246
v_accvgpr_read_b32 v19, a247
s_nop 1
s_or_b32 s28, s24, s25
 s_cmp_eq_u32 s28, 0
 s_cbranch_scc1 no_shadow_61
 v_mul_lo_u32 v13, s21, v10
 v_add_u32_e32 v13, v13, v11
 v_lshlrev_b32_e32 v13, 2, v13
 s_lshl_b32 s28, s21, 2
buffer_store_dword v16, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v17, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v18, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v19, v13, s[24:27], 0 offen
no_shadow_61:
v_trunc_f32_e32 v14, v16
 v_cmp_eq_f32_e32 vcc, v14, v16
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v16
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v16
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v32, v[28:29]
v_trunc_f32_e32 v14, v17
 v_cmp_eq_f32_e32 vcc, v14, v17
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v17
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v17
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v33, v[28:29]
v_trunc_f32_e32 v14, v18
 v_cmp_eq_f32_e32 vcc, v14, v18
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v18
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v18
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v34, v[28:29]
v_trunc_f32_e32 v14, v19
 v_cmp_eq_f32_e32 vcc, v14, v19
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v19
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v19
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v35, v[28:29]
s_waitcnt vmcnt(0)
v_bfe_u32 v14, v48, 0, 16
 v_add_u32_e32 v32, v32, v14
 v_sub_u32_e64 v15, v32, s16
 v_cmp_ge_u32_e64 vcc, v32, s16
 v_cndmask_b32_e32 v32, v32, v15, vcc
v_bfe_u32 v14, v48, 16, 16
 v_add_u32_e32 v33, v33, v14
 v_sub_u32_e64 v15, v33, s16
 v_cmp_ge_u32_e64 vcc, v33, s16
 v_cndmask_b32_e32 v33, v33, v15, vcc
v_bfe_u32 v14, v49, 0, 16
 v_add_u32_e32 v34, v34, v14
 v_sub_u32_e64 v15, v34, s16
 v_cmp_ge_u32_e64 vcc, v34, s16
 v_cndmask_b32_e32 v34, v34, v15, vcc
v_bfe_u32 v14, v49, 16, 16
 v_add_u32_e32 v35, v35, v14
 v_sub_u32_e64 v15, v35, s16
 v_cmp_ge_u32_e64 vcc, v35, s16
 v_cndmask_b32_e32 v35, v35, v15, vcc
v_lshlrev_b32_e32 v33, 16, v33
 v_or_b32_e32 v36, v32, v33
 v_lshlrev_b32_e32 v35, 16, v35
 v_or_b32_e32 v37, v34, v35
 buffer_store_dwordx2 v[36:37], v12, s[8:11], 0 offen
// group 62, AGPR 248..251, row+112, col+96
v_add_u32_e32 v10, 112, v8
 v_add_u32_e32 v11, 96, v9
 v_mul_lo_u32 v12, s21, v11
 v_add_u32_e32 v12, v12, v10
 v_lshlrev_b32_e32 v12, 1, v12
 v_mov_b32_e32 v48, 0
 v_mov_b32_e32 v49, 0
 s_cmp_eq_u32 s20, 0
 s_cbranch_scc1 no_old_62
 buffer_load_dwordx2 v[48:49], v12, s[8:11], 0 offen
no_old_62:
v_accvgpr_read_b32 v16, a248
v_accvgpr_read_b32 v17, a249
v_accvgpr_read_b32 v18, a250
v_accvgpr_read_b32 v19, a251
s_nop 1
s_or_b32 s28, s24, s25
 s_cmp_eq_u32 s28, 0
 s_cbranch_scc1 no_shadow_62
 v_mul_lo_u32 v13, s21, v10
 v_add_u32_e32 v13, v13, v11
 v_lshlrev_b32_e32 v13, 2, v13
 s_lshl_b32 s28, s21, 2
buffer_store_dword v16, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v17, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v18, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v19, v13, s[24:27], 0 offen
no_shadow_62:
v_trunc_f32_e32 v14, v16
 v_cmp_eq_f32_e32 vcc, v14, v16
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v16
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v16
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v32, v[28:29]
v_trunc_f32_e32 v14, v17
 v_cmp_eq_f32_e32 vcc, v14, v17
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v17
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v17
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v33, v[28:29]
v_trunc_f32_e32 v14, v18
 v_cmp_eq_f32_e32 vcc, v14, v18
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v18
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v18
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v34, v[28:29]
v_trunc_f32_e32 v14, v19
 v_cmp_eq_f32_e32 vcc, v14, v19
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v19
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v19
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v35, v[28:29]
s_waitcnt vmcnt(0)
v_bfe_u32 v14, v48, 0, 16
 v_add_u32_e32 v32, v32, v14
 v_sub_u32_e64 v15, v32, s16
 v_cmp_ge_u32_e64 vcc, v32, s16
 v_cndmask_b32_e32 v32, v32, v15, vcc
v_bfe_u32 v14, v48, 16, 16
 v_add_u32_e32 v33, v33, v14
 v_sub_u32_e64 v15, v33, s16
 v_cmp_ge_u32_e64 vcc, v33, s16
 v_cndmask_b32_e32 v33, v33, v15, vcc
v_bfe_u32 v14, v49, 0, 16
 v_add_u32_e32 v34, v34, v14
 v_sub_u32_e64 v15, v34, s16
 v_cmp_ge_u32_e64 vcc, v34, s16
 v_cndmask_b32_e32 v34, v34, v15, vcc
v_bfe_u32 v14, v49, 16, 16
 v_add_u32_e32 v35, v35, v14
 v_sub_u32_e64 v15, v35, s16
 v_cmp_ge_u32_e64 vcc, v35, s16
 v_cndmask_b32_e32 v35, v35, v15, vcc
v_lshlrev_b32_e32 v33, 16, v33
 v_or_b32_e32 v36, v32, v33
 v_lshlrev_b32_e32 v35, 16, v35
 v_or_b32_e32 v37, v34, v35
 buffer_store_dwordx2 v[36:37], v12, s[8:11], 0 offen
// group 63, AGPR 252..255, row+112, col+112
v_add_u32_e32 v10, 112, v8
 v_add_u32_e32 v11, 112, v9
 v_mul_lo_u32 v12, s21, v11
 v_add_u32_e32 v12, v12, v10
 v_lshlrev_b32_e32 v12, 1, v12
 v_mov_b32_e32 v48, 0
 v_mov_b32_e32 v49, 0
 s_cmp_eq_u32 s20, 0
 s_cbranch_scc1 no_old_63
 buffer_load_dwordx2 v[48:49], v12, s[8:11], 0 offen
no_old_63:
v_accvgpr_read_b32 v16, a252
v_accvgpr_read_b32 v17, a253
v_accvgpr_read_b32 v18, a254
v_accvgpr_read_b32 v19, a255
s_nop 1
s_or_b32 s28, s24, s25
 s_cmp_eq_u32 s28, 0
 s_cbranch_scc1 no_shadow_63
 v_mul_lo_u32 v13, s21, v10
 v_add_u32_e32 v13, v13, v11
 v_lshlrev_b32_e32 v13, 2, v13
 s_lshl_b32 s28, s21, 2
buffer_store_dword v16, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v17, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v18, v13, s[24:27], 0 offen
v_add_u32_e32 v13, s28, v13
buffer_store_dword v19, v13, s[24:27], 0 offen
no_shadow_63:
v_trunc_f32_e32 v14, v16
 v_cmp_eq_f32_e32 vcc, v14, v16
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v16
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v16
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v32, v[28:29]
v_trunc_f32_e32 v14, v17
 v_cmp_eq_f32_e32 vcc, v14, v17
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v17
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v17
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v33, v[28:29]
v_trunc_f32_e32 v14, v18
 v_cmp_eq_f32_e32 vcc, v14, v18
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v18
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v18
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v34, v[28:29]
v_trunc_f32_e32 v14, v19
 v_cmp_eq_f32_e32 vcc, v14, v19
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_and_b32_e32 v14, 0x7fffffff, v19
 v_cmp_lt_f32_e64 vcc, v14, s22
 v_cndmask_b32_e64 v15, 1, 0, vcc
 v_or_b32_e32 v60, v60, v15
 v_cvt_f64_f32 v[24:25], v19
 v_mul_f64 v[24:25], v[24:25], v[40:41]
 v_mul_f64 v[26:27], v[24:25], v[44:45]
 v_floor_f64 v[26:27], v[26:27]
 v_fma_f64 v[28:29], -v[26:27], v[42:43], v[24:25]
 v_add_f64 v[30:31], v[28:29], v[42:43]
 v_cmp_lt_f64_e64 vcc, v[28:29], 0
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_add_f64 v[30:31], v[28:29], -v[42:43]
 v_cmp_ge_f64_e32 vcc, v[28:29], v[42:43]
 v_cndmask_b32_e32 v28, v28, v30, vcc
 v_cndmask_b32_e32 v29, v29, v31, vcc
 v_cvt_u32_f64 v35, v[28:29]
s_waitcnt vmcnt(0)
v_bfe_u32 v14, v48, 0, 16
 v_add_u32_e32 v32, v32, v14
 v_sub_u32_e64 v15, v32, s16
 v_cmp_ge_u32_e64 vcc, v32, s16
 v_cndmask_b32_e32 v32, v32, v15, vcc
v_bfe_u32 v14, v48, 16, 16
 v_add_u32_e32 v33, v33, v14
 v_sub_u32_e64 v15, v33, s16
 v_cmp_ge_u32_e64 vcc, v33, s16
 v_cndmask_b32_e32 v33, v33, v15, vcc
v_bfe_u32 v14, v49, 0, 16
 v_add_u32_e32 v34, v34, v14
 v_sub_u32_e64 v15, v34, s16
 v_cmp_ge_u32_e64 vcc, v34, s16
 v_cndmask_b32_e32 v34, v34, v15, vcc
v_bfe_u32 v14, v49, 16, 16
 v_add_u32_e32 v35, v35, v14
 v_sub_u32_e64 v15, v35, s16
 v_cmp_ge_u32_e64 vcc, v35, s16
 v_cndmask_b32_e32 v35, v35, v15, vcc
v_lshlrev_b32_e32 v33, 16, v33
 v_or_b32_e32 v36, v32, v33
 v_lshlrev_b32_e32 v35, 16, v35
 v_or_b32_e32 v37, v34, v35
 buffer_store_dwordx2 v[36:37], v12, s[8:11], 0 offen
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
 s_endpgm
.size f6gemm_awe_modular_v1, .-f6gemm_awe_modular_v1
.section .rodata,"a",@progbits
.p2align 6
.globl f6gemm_awe_modular_v1.kd
.type f6gemm_awe_modular_v1.kd,@object
f6gemm_awe_modular_v1.kd:
 .incbin "modular_descriptor.bin",0,16
 .quad f6gemm_awe_modular_v1-f6gemm_awe_modular_v1.kd
 .incbin "modular_descriptor.bin",24,40
.size f6gemm_awe_modular_v1.kd,64
.section .note,"a",@note
.p2align 2
 .incbin "modular_note.bin"
