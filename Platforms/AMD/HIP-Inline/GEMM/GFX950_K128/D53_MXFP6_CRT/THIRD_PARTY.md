# Third-party code in this directory

## AITER (MIT): `AITER/modular.s`, `AITER/modular_descriptor.bin`, `AITER/modular_note.bin`

The MXFP6 GEMM kernel `f6gemm_awe_modular_v2` is derived from the prebuilt code object
`gemm_mxfp6_dmabig_grp16.co` (kernel `f6gemm_dmabig_grp16_kernel_func`, gfx950) shipped in the
[AITER repository](https://github.com/ROCm/aiter) at commit `22c7cb461abd52a7ed1bd87709e53d1c5e46bcc5`.
The code object was disassembled instruction by instruction, re-assembled byte-identically, and then
changed as listed below. The kernel-argument layout follows AITER's `asm_gemm_a6w6.cu` and
`gemm_op_a6w6.py`. `modular_descriptor.bin` and `modular_note.bin` are the kernel descriptor and the
code-object metadata note, with a 448-byte argument block, 144 KiB of local data share (LDS), 256 vector registers and 256 accumulator registers.
Comments with an original program counter identify instructions in the AITER code object.

- Each workgroup covers a 256 × 256 output tile in plain grid order. The kernel argument points to
  an array of 448-byte records, one per product segment. The workgroup advances through those records
  on the device.
- The main loop stages three 48 KiB operand tiles in LDS and supplies the constant E8M0 scale 130 to
  the MXFP6 instructions. The AITER packet layout and matrix multiply schedule apply to each segment.
- The epilogue checks each FP32 accumulator for an integer value of magnitude below 2²⁴. It applies
  the weight modulo the current modulus with integer multiply-high and holds partial residues in
  registers. The last segment of a modulus writes the tile's uint16 residues. An invalid accumulator
  sets the shared error flag.

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
