#pragma once

#define CRT2_N1 8
__host__ __device__ constexpr int crt2_idx(int i) {
    return i == 0    ? 9
           : i == 1  ? 10
           : i == 2  ? 11
           : i == 3  ? 12
           : i == 4  ? 13
           : i == 5  ? 14
           : i == 6  ? 15
           : i == 7  ? 16
           : i == 8  ? 0
           : i == 9  ? 1
           : i == 10 ? 2
           : i == 11 ? 3
           : i == 12 ? 4
           : i == 13 ? 5
           : i == 14 ? 6
           : i == 15 ? 7
           : i == 16 ? 8
                     : 0;
}
