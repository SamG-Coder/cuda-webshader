// MIT fixture: two implementations distinguish runtime virtual dispatch.
class DispatchBase {public:__device__ virtual float value(float x) const=0;};
class DispatchScale:public DispatchBase {public:float factor;__device__ DispatchScale(float f):factor(f){}__device__ float value(float x) const{return factor*x;}};
class DispatchOffset:public DispatchBase {public:float offset;__device__ DispatchOffset(float f):offset(f){}__device__ float value(float x) const{return offset+x;}};
