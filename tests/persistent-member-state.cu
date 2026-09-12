// MIT regression: mutations through member/helper references survive submissions.
class Counter {
public:
 int value;
 __device__ Counter(int v) : value(v) {}
 __device__ int advance(int amount) { value += amount; return value; }
};

__global__ void create(Counter** objects) {
 int i=0; objects[i++]=new Counter(i+10); objects[i++]=new Counter(i+20);
}
__global__ void update(Counter** objects, int* out) {
 out[0]=objects[0]->advance(3); out[1]=objects[1]->advance(7);
}
__global__ void cleanup(Counter** objects, int* out) {
 out[2]=objects[0]->value; out[3]=objects[1]->value;
 delete objects[0]; delete objects[1]; objects[0]=NULL; objects[1]=NULL;
}
