// Independent CPU references used by correctness tests, never by sandbox previews.
export const scalarCases=[{vectorN:1,elementN:1,blocks:1,threads:32},{vectorN:7,elementN:33,blocks:3,threads:128},{vectorN:17,elementN:1537,blocks:4,threads:128},{vectorN:5,elementN:4096,blocks:2,threads:256}];
export function scalarFixture({vectorN=17,elementN=1537,blocks=4,guards=16}={}){
 const count=vectorN*elementN,d_A=Float32Array.from({length:count+guards},(_,i)=>(i%23-11)/8),d_B=Float32Array.from({length:count+guards},(_,i)=>(i%17-8)/8),d_C=new Float32Array(vectorN+guards).fill(-12345),expected=d_C.slice();
 for(let v=0;v<vectorN;v++){let sum=0;for(let i=0;i<elementN;i++)sum+=d_A[v*elementN+i]*d_B[v*elementN+i];expected[v]=sum;}
 return {buffers:{d_C,d_A,d_B},scalars:{vectorN,elementN},out:'d_C',expected,groups:[blocks,1,1]};
}
export const mul24Inputs=[0,1,-1,8388607,8388608,16777215,16777216,2147483647,-2147483648,305419896,-305419896];
export const mul24Source='__global__ void multiply24(int* signedOut, unsigned int* unsignedOut, const int* a, const int* b, unsigned int n) {unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){signedOut[i]=__mul24(a[i],b[i]);unsignedOut[i]=__umul24((unsigned int)a[i],(unsigned int)b[i]);}}';
export function mul24Expected(a,b,signed){const extend=x=>{const low=BigInt(x>>>0)&0xffffffn;return signed&&low>=0x800000n?low-0x1000000n:low;};const product=extend(a)*extend(b);return Number(signed?BigInt.asIntN(32,product):BigInt.asUintN(32,product));}
