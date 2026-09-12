class Counter {
 int value;
 __device__ int twice() const {return value*2;}
public:
 __device__ Counter(int seed):value(seed){}
 __device__ void add(int delta){value+=delta;}
 __device__ int read() const {return twice();}
 __device__ int compare(const Counter& other) const {return value-other.value;}
};
__global__ void private_class_values(int* out){unsigned i=threadIdx.x+blockIdx.x*blockDim.x;Counter a((int)i);Counter b(2);a.add(3);out[i]=a.read()+a.compare(b);}
