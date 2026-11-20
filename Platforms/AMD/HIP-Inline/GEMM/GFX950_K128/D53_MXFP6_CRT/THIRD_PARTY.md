# Third-party code in this directory

## AITER (MIT): `AITER/modular.s`, `AITER/modular_descriptor.bin`, `AITER/modular_note.bin`

The MXFP6 GEMM kernel `f6gemm_awe_modular_v1` is derived from the prebuilt code object
`gemm_mxfp6_dmabig_grp16.co` (kernel `f6gemm_dmabig_grp16_kernel_func`, gfx950) shipped in the
[AITER repository](https://github.com/ROCm/aiter) at commit `22c7cb461abd52a7ed1bd87709e53d1c5e46bcc5`.
The code object was disassembled instruction by instruction, re-assembled byte-identically, and then
changed as listed below. The kernel-argument layout follows AITER's `asm_gemm_a6w6.cu` and
`gemm_op_a6w6.py`. `modular_descriptor.bin` and `modular_note.bin` are the kernel descriptor and the
code-object metadata note of the original, with the kernel name, symbol and argument size updated.
Each instruction line up to the epilogue boundary (original PC 0x2df8) ends with its address in the
original code object (`// original PC 0x...`), so it can be compared with a disassembly of AITER's
code object; the epilogue after the boundary is new code.

Changes to the original kernel:

- The epilogue stores FP32 instead of BF16: the two `v_cvt_pk_bf16_f32` + `buffer_store_short`
  pairs per accumulator became `v_mov_b32` + `buffer_store_dword`, with the output stride doubled
  (byte address `4 (m stride + n)` instead of `2 (m stride + n)`). Integer products summed in FP32 are
  exact below 2^24; the BF16 conversion was not.
- The grid swizzle for "grp16" tiles is bypassed (`s_cbranch_scc1 swz_done` at the original PC
  0x15d4 became `s_branch swz_done`): both K = 4096 and K = 8192 launches use full 256 x 256 tiles
  and the plain tile order.
- The FP32 epilogue was replaced by the modular residue update: each accumulator is checked to be an
  integer of magnitude below 2^24, reduced modulo p (FP64 floor-based), multiplied by the product weight, added to
  the stored uint16 residue (optionally) and stored as uint16. The residue pointer, invalid flag,
  modulus, weight, 1/p, accumulate flag, residue stride and limit are a 64-byte tail appended to the
  original 384-byte argument block (bytes 384 to 447). Any invalid lane sets the shared flag with one
  atomic OR.
- Nothing before the epilogue boundary (original PC 0x2df8) was changed except the branch above; the
  main loop, LDS layout and operand layout are AITER's.

MIT License text of AITER:

```
Copyright © Advanced Micro Devices, Inc. All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## HipKittens (MIT)

The encoder's packets of 16 FP6 values in 3 dwords were first designed for the `fp6_dwordx3` kernels
of [HipKittens](https://github.com/HazyResearch/HipKittens) (commit
`9703faea3758f6fd6e47aed671454a437433f637`); their placement in memory follows the AITER layout
above. No HipKittens source is included. Its notice:

```
MIT License

Copyright (c) 2024 HazyResearch

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Everything else in this directory is covered by the repository [LICENSE](../../../../../../LICENSE).
