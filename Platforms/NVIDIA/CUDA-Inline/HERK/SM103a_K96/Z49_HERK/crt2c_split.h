#pragma once

#define CRT2C_N1 9
__host__ __device__ constexpr int crt2c_idx(int i) {
    return i == 0    ? 0
           : i == 1  ? 5
           : i == 2  ? 7
           : i == 3  ? 10
           : i == 4  ? 12
           : i == 5  ? 13
           : i == 6  ? 15
           : i == 7  ? 17
           : i == 8  ? 18
           : i == 9  ? 1
           : i == 10 ? 2
           : i == 11 ? 3
           : i == 12 ? 4
           : i == 13 ? 6
           : i == 14 ? 8
           : i == 15 ? 9
           : i == 16 ? 11
           : i == 17 ? 14
           : i == 18 ? 16
                     : 0;
}
