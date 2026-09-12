// MIT probe: the Ray and matrix shapes used by NVIDIA volumeRender.
struct Ray { float3 o; float3 d; };
typedef struct { float4 m[3]; } float3x4;
__constant__ float3x4 camera;
__device__ Ray shifted(Ray ray) { ray.o = ray.o + ray.d; return ray; }
__global__ void structProbe(const float4* input, float4* output, unsigned int row) {
    Ray ray;
    ray.o = make_float3(input[0]);
    ray.d = make_float3(input[1]);
    Ray result = shifted(ray);
    float3x4 matrix = camera;
    matrix.m[0].x = 99.0f;
    output[0] = make_float4(result.o, 1.0f);
    output[1] = make_float4(ray.o, 1.0f);
    output[2] = matrix.m[row];
    output[3] = camera.m[0];
}
