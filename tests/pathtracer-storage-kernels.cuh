__global__ void rand_init(curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, rand_state);
    }
}

__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    // Original: Each thread gets same seed, a different sequence number, no offset
    // curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
    // BUGFIX, see Issue#2: Each thread gets different seed, same sequence for
    // performance improvement of about 2x!
    curand_init(1984+pixel_index, 0, 0, &rand_state[pixel_index]);
}


// MIT verification kernels for buffer persistence, wide seeds and cleanup.
__global__ void seed_wide(curandState*rand_state){
 unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<512){size_t seed=i%2u?((size_t)(-int(i))):((size_t)i)*((size_t)4294967295u)+((size_t)i);curand_init(seed,0,0,&rand_state[i]);}
}
__global__ void draw_states(curandState*rand_state,vec3*fb,unsigned int*bits){
 unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<512){curandState state=rand_state[i];float x=curand_uniform(&state),y=curand_uniform(&state),z=curand_uniform(&state);fb[i]=vec3(x,y,z);bits[i]=curand(&state);rand_state[i]=state;}
}
__global__ void create_storage_scene(hitable**world){
 unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<512)world[i]=new sphere(vec3(0.0f,0.0f,-1.0f),0.5f,new lambertian(vec3(0.2f,0.4f,0.8f)));
}
__global__ void free_storage_scene(hitable**world){
 unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<512){delete ((sphere*)world[i])->mat_ptr;delete world[i];world[i]=NULL;}
}
