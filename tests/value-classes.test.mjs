import test from 'node:test';
import assert from 'node:assert/strict';
import {compile} from '../src/compiler/compiler.js';
import {executeCPU} from '../src/compiler/cpu-oracle.js';
const source=`class vec3 {
public:
 __host__ __device__ vec3() {}
 __host__ __device__ vec3(float e0,float e1,float e2){e[0]=e0;e[1]=e1;e[2]=e2;}
 __host__ __device__ inline float x() const{return e[0];}
 __host__ __device__ inline float y() const{return e[1];}
 __host__ __device__ inline float z() const{return e[2];}
 __host__ __device__ inline float squared_length() const{return e[0]*e[0]+e[1]*e[1]+e[2]*e[2];}
 float e[3];
};`;
test('Value-class constructors and const methods preserve array fields and copies',()=>{
 const c=compile(source+'__global__ void k(float*out){vec3 a(2.0f,3.0f,4.0f);vec3 b=a;a.e[0]=9.0f;vec3 c=vec3(1.0f,2.0f,3.0f);out[0]=b.x();out[1]=b.y();out[2]=b.z();out[3]=b.squared_length();out[4]=c.squared_length();}',{workgroupSize:[1,1,1]}),out=new Float32Array(5);
 executeCPU(c,{out},{},[1]);assert.deepEqual([...out],[2,3,4,29,14]);
});
test('Value classes reject unsupported access, mutable methods and const mutation',()=>{
 for(const body of ['class V {float x;};','class V {public: float x;__device__ void change(){x=1.0f;}};','class V {public: float x;__device__ float get() const{x=2.0f;return x;}};'])assert.throws(()=>compile(body+'__global__ void k(float*out){V v;out[0]=v.get();}'));
});
test('Resolved value-class method recursion is rejected',()=>{
 assert.throws(()=>compile('class V{public:float e;__device__ V(float x){e=x;}__device__ float get() const{V other(1.0f);return other.get();}};__global__ void k(float*out){V v(1.0f);out[0]=v.get();}'),/Recursive/);
});
