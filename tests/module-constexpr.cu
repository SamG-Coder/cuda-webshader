// MIT compiler prerequisite fixture for Chrono's integration helpers.
static constexpr double third = 1.0 / 3.0;
constexpr double exact = (16777216.0 + 1.0) - 16777216.0;
constexpr float rounded = (16777216.0f + 1.0f) - 16777216.0f;
constexpr int quotient = 7 / 3;
constexpr unsigned wrapped = 4294967295u + 1u;
constexpr double twice = third * 2.0;
__device__ void setFlag(volatile bool* flag, bool value) { *flag = value; }
__device__ float constantValue() { return float(twice); }
__global__ void constants(float* output, volatile bool* flags) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 65) return;
    output[i * 8] = float(third);
    output[i * 8 + 1] = float(exact);
    output[i * 8 + 2] = rounded;
    output[i * 8 + 3] = float(quotient);
    output[i * 8 + 4] = float(wrapped);
    output[i * 8 + 5] = constantValue();
    output[i * 8 + 6] = float(third * float(i));
    { float third = 7.f; output[i * 8 + 7] = third; }
    setFlag(flags + i, (i % 3) == 0);
}
