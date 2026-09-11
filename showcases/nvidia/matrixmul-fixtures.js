export function matrixFixture(tile=16,M=32,N=48,K=64,guards=0){
 if([M,N,K].some(n=>!Number.isInteger(n)||n<tile||n%tile))throw Error('NVIDIA matrixMul requires positive dimensions divisible by the tile.');
 const A=Float32Array.from({length:M*K},(_,i)=>(i%17-8)/16),B=Float32Array.from({length:K*N},(_,i)=>(i%13-6)/8),C=new Float32Array(M*N+guards).fill(-12345),expected=C.slice();
 for(let y=0;y<M;y++)for(let x=0;x<N;x++){let sum=0;for(let k=0;k<K;k++)sum+=A[y*K+k]*B[k*N+x];expected[y*N+x]=sum;}
 return {buffers:{A,B,C},scalars:{wA:K,wB:N},out:'C',expected,groups:[N/tile,M/tile,1]};
}
