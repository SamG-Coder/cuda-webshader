__device__ float softened_inverse_radius(float radiusSquared) {
    return rsqrtf(radiusSquared+0.35f);
}
// One lane owns one float4 record: no atomics, CPU readback or inter-lane synchronization.
__global__ void particles(float4* position, float4* velocity,
                         unsigned int n, float dt, float time, float attraction) {
    unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i<n) {
        float4 p = position[i];
        float4 v = velocity[i];
        float radiusSquared = p.x*p.x+p.z*p.z;
        float inv = softened_inverse_radius(radiusSquared);
        float radial = attraction*(1.4f-0.11f*sqrtf(radiusSquared+0.001f));
        float ax = (-p.z*0.52f-p.x*radial)*inv;
        float az = ( p.x*0.52f-p.z*radial)*inv;
        float ay = sinf(time*0.37f+p.w*6.283185f)*0.18f-p.y*0.22f;
        float drag = fmaxf(0.0f,1.0f-dt*0.12f);
        v.x = fmaf(ax,dt,v.x)*drag;
        v.y = fmaf(ay,dt,v.y)*drag;
        v.z = fmaf(az,dt,v.z)*drag;
        p.x = fmaf(v.x,dt,p.x);
        p.y = fmaf(v.y,dt,p.y);
        p.z = fmaf(v.z,dt,p.z);
        position[i] = p;
        velocity[i] = v;
    }
}
