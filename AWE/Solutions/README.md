# Solutions

An AWE solution is an encoding for one pair of input widths, w_A and w_B bits. It computes the
exact integer matrix product in a given number of FP4 GEMMs. An encoding has three parts. The
limb weights of each operand. The planes, which are linear combinations of limbs. The
reconstruction coefficients, which combine the FP4 GEMM results into the exact integer product.
Encodings come in two families. Direct: the sum of the products is the exact integer. Modular:
the products are exact modulo m, for residue systems such as Ozaki scheme II.

The catalog is the source of truth: kernels take their weights, coefficients and block scales
from constants generated here and never hand-write them.

## Files

| File | Content |
|---|---|
| `contracts.jsonl` | One line per solution (`schema` = `awe-solutions-1`, unique `id`): limb weights of A and B, planes (products), reconstruction into the exact result, covered input range, safe dot-product length for an FP32 accumulator, verification state. 94 lines for INT4 to INT8 widths: 60 direct encodings, 33 from the width search, 1 five-product encoding. Planes are called faces in the library API. |
| `../CodeGen/gen_constants.py` | `python3 gen_constants.py contracts.jsonl <solution id> > constants.h` writes the C constants for one solution. Standard library only. The tables inside `Platforms/NVIDIA/CUDA/GEMM` are a snapshot of this catalog. |

## Storage Set

Every solution stores limbs in E2M1 with an E4M3 block scale. The limb value set is
{0, ±1, ±2, ±3, ±4, ±6, ±8, ±12}. A limb value v is stored as v/2, and the block scale
restores the integer weight.

## Verification States

| State | Meaning |
|---|---|
| CPU exact | The integer identity holds over the whole covered domain (exhaustive check on the host). |
| GPU bit-identical | A GPU run reproduced the host reference bit for bit on the stated shape. |
