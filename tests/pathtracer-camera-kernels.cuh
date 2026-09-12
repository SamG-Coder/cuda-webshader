// MIT comparison harness; camera and RNG calls are the original APIs.
__global__ void check_camera(unsigned int *bits,float *out){
 unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<512){
 unsigned int seed=i==0u?0u:(i==1u?4294967295u:1984u+i);
 curandState state;curand_init(seed,0,0,&state);
 for(int j=0;j<8;j++){bits[i*9+j]=curand(&state);out[i*23+j]=curand_uniform(&state);}
 camera cam(vec3(13,2,3),vec3(0,0,0),vec3(0,1,0),20.0f+float(i%7u),1.5f,float(i%3u)*0.1f,10.0f);
 ray r=cam.get_ray(float(i%31u)/30.0f,float(i%17u)/16.0f,&state);
 vec3 o=r.origin(),d=r.direction();
 out[i*23+8]=o.x();out[i*23+9]=o.y();out[i*23+10]=o.z();
 out[i*23+11]=d.x();out[i*23+12]=d.y();out[i*23+13]=d.z();
 out[i*23+14]=cam.u.x();out[i*23+15]=cam.u.y();out[i*23+16]=cam.u.z();
 out[i*23+17]=cam.v.x();out[i*23+18]=cam.v.y();out[i*23+19]=cam.v.z();
 out[i*23+20]=cam.w.x();out[i*23+21]=cam.w.y();out[i*23+22]=cam.w.z();
 bits[i*9+8]=curand(&state);
 }
}
