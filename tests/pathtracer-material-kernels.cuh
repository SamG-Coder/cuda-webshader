// MIT native/WebGPU harness for original material implementations.
__global__ void check_material(float*out,unsigned int*bits){
 unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<512){
 curandState state;curand_init(1984u+i,0,0,&state);
 material *m=NULL;
 if(i%3u==0u)m=new lambertian(vec3(0.2f,0.4f,0.8f));
 else if(i%3u==1u)m=new metal(vec3(0.8f,0.6f,0.2f),float(i%5u)*0.3f);
 else m=new dielectric(1.5f);
 hit_record rec;rec.p=vec3(float(i)*0.01f,0.0f,0.0f);rec.normal=vec3(0.0f,1.0f,0.0f);rec.t=1.0f;rec.mat_ptr=m;
 ray incoming(vec3(0.0f,1.0f,0.0f),unit_vector(vec3(float(i%13u)*0.1f,i%2u==0u?-1.0f:1.0f,0.2f)));
 vec3 attenuation;ray scattered;
 bool ok=m->scatter(incoming,rec,attenuation,scattered,&state);
 vec3 origin=scattered.origin(),direction=scattered.direction();
 out[i*10]=float(ok);out[i*10+1]=attenuation.x();out[i*10+2]=attenuation.y();out[i*10+3]=attenuation.z();
 out[i*10+4]=origin.x();out[i*10+5]=origin.y();out[i*10+6]=origin.z();
 out[i*10+7]=direction.x();out[i*10+8]=direction.y();out[i*10+9]=direction.z();
 bits[i]=curand(&state);delete m;
 }
}
