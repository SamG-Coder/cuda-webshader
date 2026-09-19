// The same fixtures are executed by Node's AST oracle and by WebGPU. No image
// comparisons or approximate tolerances: every output word must match baseline.
const wrap = (helpers, body) => `${helpers}\n__global__ void k(int* output,int input){int i=(int)threadIdx.x;${body}}`;
const sink = `struct Sink{int mode;int count;};
__device__ Sink newSink(int mode){Sink s;s.mode=mode;s.count=0;return s;}
__device__ int costly(int x){for(int n=0;n<7;n++)x=x*3+1;return x;}
__device__ Sink emit(Sink s,int x){if(s.mode==1)s.count+=x;else if(s.mode==2)s.count-=x;else s.count+=costly(x);return s;}`;
export const optimizerCases = [
 {name:'bounds-mode', source:wrap(sink,'Sink s=newSink(1);s=emit(s,i+input);output[i]=s.count;'),prunes:true},
 {name:'trace-mode', source:wrap(sink,'Sink s=newSink(0);s=emit(s,i+input);output[i]=s.count;'),prunes:true},
 {name:'mixed-record-modes', source:wrap(sink,'Sink a=newSink(1),b=newSink(0);a=emit(a,i);b=emit(b,input);output[i]=a.count+b.count;')},
 {name:'runtime-mode', source:wrap(sink,'Sink s=newSink(input);s=emit(s,i);output[i]=s.count;')},
 {name:'record-field-mutation', source:wrap(sink,'Sink s=newSink(1);s.mode=i%3;s=emit(s,input);output[i]=s.count;')},
 {name:'conditional-initialization',source:wrap(sink,'Sink s;if(i%2==0)s.mode=1;else s.mode=0;s.count=0;s=emit(s,input);output[i]=s.count;')},
 {name:'constructor-copy',source:wrap(sink,'Sink a=newSink(1);Sink b=a;for(int j=0;j<4;j++)b=emit(b,input+j);output[i]=b.count;'),prunes:true},
 {name:'argument-chain',source:wrap(`__device__ int leaf(int mode,int x){if(mode==3)return x+5;return x-5;} __device__ int middle(int m,int x){return leaf(m,x);}`,'output[i]=middle(3,input+i);'),prunes:true},
 {name:'mixed-scalar-callers',source:wrap(`__device__ int leaf(int mode,int x){if(mode==3)return x+5;return x-5;}`,'output[i]=leaf(3,input)+leaf(i,input);')},
 {name:'parameter-mutation',source:wrap(`__device__ int leaf(int mode,int x){mode+=x;if(mode==3)return 77;return -9;}`,'output[i]=leaf(3,input+i);')},
 {name:'local-shadow',source:wrap(`__device__ int leaf(int mode,int x){{int mode=x;if(mode==3)return 77;}if(mode==1)return -9;return 2;}`,'output[i]=leaf(1,input+i);'),prunes:true},
 {name:'const-local',source:wrap(`__device__ int leaf(int mode,int x){const int selected=mode+2;if(selected==3)return x+1;return x-1;}`,'output[i]=leaf(1,input+i);'),prunes:true},
 {name:'conditional-expression',source:wrap(`__device__ int leaf(int mode,int x){return mode==1?x+1:x-1;}`,'output[i]=leaf(1,input+i);'),prunes:true},
 {name:'default-argument',source:wrap(`__device__ int leaf(int x,int mode=1){if(mode==1)return x+1;return x-1;}`,'output[i]=leaf(input+i);'),prunes:true},
 {name:'short-circuit-effects',source:wrap('', 'int a=0;if(false && ++a>0)a=99;if(true || ++a>0)a+=3;output[i]=a;'),prunes:true},
 {name:'live-effects',source:wrap('', 'int a=input;if(++a>0)a+=i;output[i]=a;')},
 {name:'unsigned-wrap',source:wrap(`__device__ int leaf(unsigned int m,int x){if(m+1u==0u)return x+1;return x-1;}`,'output[i]=leaf(4294967295u,input+i);'),prunes:true},
 {name:'float-not-folded',source:wrap(`__device__ int leaf(float m,int x){if(m+0.1f==1.1f)return x+1;return x-1;}`,'output[i]=leaf(1.0f,input+i);')},
 {name:'negative-signed-mixed',source:wrap(`__device__ int leaf(int m,int x){if(m<1u)return x+1;return x-1;}`,'output[i]=leaf(-1,input+i);'),prunes:true},
 {name:'switch-local-scope',source:wrap(`__device__ int leaf(int mode,int x){switch(x%2){case 0:{int mode=x;return mode;}default:break;}if(mode==1)return x+1;return x-1;}`,'output[i]=leaf(1,input+i);'),prunes:true},
 {name:'reference-fallback',source:wrap(`__device__ void change(int& m){m=3;} __device__ int leaf(int mode,int x){change(mode);if(mode==3)return x+1;return x-1;}`,'output[i]=leaf(1,input+i);')},
];
