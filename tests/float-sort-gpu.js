export async function checkFloatSortPairs(runtime){
 const comparisons=[];
 const check=async(bits,values,expected,label)=>{
  const count=bits.length,ka=new Uint32Array(count+4).fill(0xabcdef01),va=ka.slice();ka.set(bits);va.set(values);
  const keys=runtime.createBuffer(ka),payloads=runtime.createBuffer(va);
  try{
   const before=runtime.stats.readbackBytes;await runtime.sortPairs(keys,payloads,{count,keyType:'f32'});if(runtime.stats.readbackBytes!==before)throw Error('Float sorting read back data');
   const k=await runtime.read(keys,Uint32Array),v=await runtime.read(payloads,Uint32Array);
   for(let i=0;i<count;i++)if(k[i]!==expected[i].bits||v[i]!==expected[i].value)throw Error('Float sort mismatch '+label+' at '+i);
   for(let i=count;i<ka.length;i++)if(k[i]!==0xabcdef01||v[i]!==0xabcdef01)throw Error('Float sort guard overwritten');
   comparisons.push({label,count,exactKeyBits:true,exactValues:true,guardsIntact:true,intermediateReadbackBytes:0});
  }finally{runtime.destroyBuffer(keys);runtime.destroyBuffer(payloads);}
 };
 const edges=[0,0x80000000,0x7f800000,0xff800000,1,0x80000001,0x007fffff,0x807fffff,0x00800000,0x80800000,0x7f7fffff,0xff7fffff,0x7fc00001,0xffc00002,0x7f800001,0xffffffff,0x3f800000,0xbf800000];
 for(const count of [1,2,127,128,129,257,16384]){
  const bits=Uint32Array.from({length:count},(_,i)=>edges[(i*7)%edges.length]),floats=new Float32Array(bits.buffer),values=Uint32Array.from({length:count},(_,i)=>(0xf0000000+i)>>>0);
  const expected=Array.from({length:count},(_,i)=>({bits:bits[i],value:values[i],key:floats[i],i})).sort((a,b)=>{
   if(Number.isNaN(a.key))return Number.isNaN(b.key)?a.i-b.i:1;if(Number.isNaN(b.key))return -1;return (a.key<b.key?-1:a.key>b.key?1:0)||a.i-b.i;
  });
  await check(bits,values,expected,'IEEE edge cases '+count);
 }
 for(const frame of [1,8,32,64]){
  const bits=new Uint32Array(await(await fetch('/reports/smoke-integration-'+frame+'-depth.bin')).arrayBuffer()),native=new Uint32Array(await(await fetch('/reports/smoke-sort-'+frame+'-native.bin')).arrayBuffer()),count=bits.length;
  if(native.length!==count*2)throw Error('Invalid native depth sort capture');
  await check(bits,Uint32Array.from({length:count},(_,i)=>i),Array.from({length:count},(_,i)=>({bits:native[i],value:native[count+i]})),'native smoke step '+frame);
 }
 return {comparisons,softwareAdapterRequested:false};
}
