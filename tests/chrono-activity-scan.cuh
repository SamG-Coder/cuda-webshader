// Copyright (c) 2014 projectchrono.org. All rights reserved.
// Original Chrono declarations at a92c6f72f422fbcafe0b37125d4070cb6a3b5803.
// BSD license reproduced in THIRD_PARTY_NOTICES.md.
struct ActivityScanOp {
    __host__ __device__ int operator()(const int& a, const int& b) const {
        // Treat -1 the same as 0 (only add positive values)
        int b_value = (b <= 0) ? 0 : b;
        return a + b_value;
    }
};
