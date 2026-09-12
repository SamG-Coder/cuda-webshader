// MIT harness kernels using the original sphere/ray implementations.
__global__ void create_objects(hitable** world,float offset){
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
 world[i]=new sphere(vec3(float(i)*0.25f+offset,0.0f,-1.0f),0.5f,NULL);
}
__global__ void trace_objects(hitable** world,float*out,float offset){
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
 ray r(vec3(float(i)*0.25f+offset,0.0f,0.0f),vec3(0.0f,0.0f,-1.0f));
 hit_record rec;rec.t=-1.0f;rec.p=vec3(0.0f,0.0f,0.0f);rec.normal=vec3(0.0f,0.0f,0.0f);rec.mat_ptr=NULL;
 bool hit=world[i]->hit(r,0.0f,100.0f,rec);
 out[i*5]=float(hit);out[i*5+1]=rec.t;out[i*5+2]=rec.p.x();out[i*5+3]=rec.normal.z();out[i*5+4]=float(rec.mat_ptr==NULL);
}
__global__ void free_objects(hitable** world){unsigned i=blockIdx.x*blockDim.x+threadIdx.x;delete world[i];world[i]=NULL;}
