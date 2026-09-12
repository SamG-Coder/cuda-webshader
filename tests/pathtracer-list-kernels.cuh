// MIT verification harness; original class implementations supplied separately.
__global__ void create_list(hitable **objects,hitable **world) {
 objects[0]=new sphere(vec3(0.0f,0.0f,-4.0f),0.5f,NULL);
 objects[1]=new sphere(vec3(0.0f,0.0f,-2.0f),0.5f,NULL);
 world[0]=new hitable_list(objects,2);
}
__global__ void update_list(hitable **objects) {
 delete objects[1];
 objects[1]=new sphere(vec3(0.0f,0.0f,-6.0f),0.5f,NULL);
}
__global__ void trace_list(hitable **world,float *out) {
 unsigned int i=threadIdx.x;
 if(i<2) {
 ray r(vec3(0.0f,0.0f,0.0f),vec3(float(i)*10.0f,0.0f,-1.0f));
 hit_record rec;rec.t=-1.0f;
 bool hit=world[0]->hit(r,0.0f,100.0f,rec);
 out[i*2]=float(hit);out[i*2+1]=rec.t;
 }
}
__global__ void free_list(hitable **objects,hitable **world) {
 delete objects[0];delete objects[1];delete world[0];
 objects[0]=NULL;objects[1]=NULL;world[0]=NULL;
}
