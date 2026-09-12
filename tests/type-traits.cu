// Original feature regression fixture, MIT. Uses NVIDIA's separate vector traits.
template<class T>
__device__ typename vec4<T>::Type advancePosition(typename vec4<T>::Type p, typename vec3<T>::Type v, T dt) {
 p.x += v.x*dt;p.y += v.y*dt;p.z += v.z*dt;return p;
}
template<class T>
__global__ void traitPositions(typename vec4<T>::Type* positions, unsigned int n, T dt) {
 unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<n){typename vec3<T>::Type v=make_float3(0.5f,-0.25f,1.0f);typename vec4<T>::Type p=positions[i];positions[i]=advancePosition<T>(p,v,dt);}
}
