// MIT capture adapter. Link with the unchanged NVIDIA sample main and kernels.
// Replace only its output writer: the upstream writer opens binary floats in
// text mode on Windows. Keep every frame for subsequent batch verification.
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" void dumpBGR(float* source, int pitch, int width, int height,
                        int batch, char* folder, char* tag) {
    std::vector<float> frame(size_t(width) * height * 3);
    char path[256];
    snprintf(path, sizeof(path), ".local/nv12-%s-native.bin", folder);
    FILE* file = fopen(path, "wb");
    if (!file) exit(1);
    for (int i = 0; i < batch; ++i) {
        checkCudaErrors(cudaMemcpy2D(frame.data(), width * sizeof(float),
            source + size_t(i) * pitch * height * 3, pitch * sizeof(float),
            width * sizeof(float), height * 3, cudaMemcpyDeviceToHost));
        if (fwrite(frame.data(), sizeof(float), frame.size(), file) != frame.size()) exit(1);
    }
    if (fclose(file)) exit(1);
    printf("Captured %s: %d frames, %dx%d, planar float BGR\n", tag, batch, width, height);
}

extern "C" void dumpYUV(unsigned char*, int, char*, char*) {
    // Neither of the original sample's default workflows calls this writer.
    fprintf(stderr, "Unexpected YUV output request\n");
    exit(1);
}
