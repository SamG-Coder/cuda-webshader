export function scanUpdateFixture(blocks=3,threads=256,guards=0){
 const count=blocks*threads*4,d_Data=Uint32Array.from({length:count+guards},(_,i)=>i<count?[0,1,16777217,2147483647,2147483648,4294967294,4294967295][i%7]:0xdeadbeef),d_Buffer=Uint32Array.from({length:blocks},(_,i)=>[1,2147483648,4294967295,17][i%4]),expected=d_Data.slice();
 for(let i=0;i<count;i++)expected[i]=(d_Data[i]+d_Buffer[Math.floor(i/(threads*4))])>>>0;
 return {buffers:{d_Data,d_Buffer},scalars:{},groups:[blocks,1,1],out:'d_Data',expected};
}
