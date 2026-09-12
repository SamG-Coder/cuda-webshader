// MIT project probe for NVIDIA helper_math operations used by volumeRender.
__global__ void rayMath(const float4* input, float4* output) {
    float3 direction = normalize(make_float3(input[0]));
    float4 homogeneous = make_float4(direction, 1.0f);
    float3 restored = make_float3(homogeneous);
    float3 low = fminf(restored, make_float3(0.5f));
    float3 high = fmaxf(restored, make_float3(0.5f));
    restored += make_float3(1.0f, 2.0f, 3.0f);
    restored *= 2.0f;
    restored -= 1.0f;
    restored /= 2.0f;
    output[0] = homogeneous;
    output[1] = make_float4(low, dot(direction, direction));
    output[2] = make_float4(high, dot(make_float4(1.0f), homogeneous));
    output[3] = make_float4(restored, 1.0f);
}
