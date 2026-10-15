#pragma once

#define CRT2_N1 9
__host__ __device__ constexpr int crt2_idx(int i) {
    return i == 0    ? 1
           : i == 1  ? 2
           : i == 2  ? 3
           : i == 3  ? 4
           : i == 4  ? 5
           : i == 5  ? 8
           : i == 6  ? 9
           : i == 7  ? 11
           : i == 8  ? 15
           : i == 9  ? 0
           : i == 10 ? 6
           : i == 11 ? 7
           : i == 12 ? 10
           : i == 13 ? 12
           : i == 14 ? 13
           : i == 15 ? 14
           : i == 16 ? 16
           : i == 17 ? 17
                     : 0;
}
