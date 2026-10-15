#pragma once

#define CRT2C_N1 10
__host__ __device__ constexpr int crt2c_idx(int i) {
    return i == 0    ? 0
           : i == 1  ? 1
           : i == 2  ? 2
           : i == 3  ? 5
           : i == 4  ? 7
           : i == 5  ? 8
           : i == 6  ? 10
           : i == 7  ? 14
           : i == 8  ? 17
           : i == 9  ? 18
           : i == 10 ? 3
           : i == 11 ? 4
           : i == 12 ? 6
           : i == 13 ? 9
           : i == 14 ? 11
           : i == 15 ? 12
           : i == 16 ? 13
           : i == 17 ? 15
           : i == 18 ? 16
                     : 0;
}
