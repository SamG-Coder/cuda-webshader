// Independent pairwise reference; used only by correctness harnesses.
export function bitonicFixture(arrayLength=256,batches=2,size=256,stride=128,dir=1,threads=128){
 const n=arrayLength*batches,special=[0,4294967295,2147483648,2147483647,7,7,1,0],d_SrcKey=Uint32Array.from({length:n+16},(_,i)=>i<8?special[i]:(Math.imul(i,1664525)+1013904223)>>>0),d_SrcVal=Uint32Array.from({length:n+16},(_,i)=>(2147483648+i)>>>0),d_DstKey=new Uint32Array(n+16).fill(0xdeadbeef),d_DstVal=d_DstKey.slice(),keys=d_DstKey.slice(),vals=d_DstVal.slice();
 for(let i=0;i<n;i++)if(!(i&stride)){const j=i^stride,ascending=!!dir!==!!((i%arrayLength)&size),swap=ascending?d_SrcKey[i]>d_SrcKey[j]:d_SrcKey[i]<=d_SrcKey[j];keys[i]=d_SrcKey[swap?j:i];keys[j]=d_SrcKey[swap?i:j];vals[i]=d_SrcVal[swap?j:i];vals[j]=d_SrcVal[swap?i:j];}
 return {buffers:{d_DstKey,d_DstVal,d_SrcKey,d_SrcVal},scalars:{arrayLength,size,stride,dir},groups:[n/2/threads,1,1],out:'d_DstKey',expected:keys,expectedOutputs:{d_DstKey:keys,d_DstVal:vals},absoluteTolerance:0};
}
