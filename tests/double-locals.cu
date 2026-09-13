// MIT: integration prerequisites, including scalar references into GPU records.
struct Pair { float x; float y; };
__device__ void updateScalar(const float& input, float& output) {
    output += 1.f;
    output += input;
}
__global__ void doubleLocals(const float* input, float* output, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    double ad = (double)input[i];
    double bd = ad * 2.0;
    double cd = -3.0;
    double disc = fmax(bd * bd - 4.0 * ad * cd, 0.0);
    double root = sqrt(disc);
    double r1 = (-bd + root) * (0.5 / ad);
    double r2 = (-bd - root) * (0.5 / ad);
    double chosen = fmax(r1, r2);
    chosen += 0.25;
    output[i*3] = (float)chosen;
    output[i*3+1] = (float)fmin(root, bd);
    output[i*3+2] = (float)sqrt(ad);
}
__global__ void recordRefs(Pair* pairs, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    updateScalar(pairs[i].x, pairs[i].y);
    updateScalar(pairs[i].x, pairs[i].x);
}
