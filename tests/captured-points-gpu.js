export async function checkCapturedPoints(runtime){
 const source=await(await fetch('/tests/captured-points.cu')).text(),options={objectHeap:'persistent',valueBuffers:['pts']},kernels={};
 for(const entry of ['bind_points','update_points','read_points'])kernels[entry]=await runtime.kernel(source,{...options,entry,workgroupSize:[entry==='bind_points'?1:32]});
 const x0=runtime.createBuffer(Float32Array.from({length:34},(_,i)=>i)),y0=runtime.createBuffer(Float32Array.from({length:34},(_,i)=>100+i)),x1=runtime.createBuffer(Float32Array.from({length:32},(_,i)=>200+i)),y1=runtime.createBuffer(Float32Array.from({length:32},(_,i)=>300+i)),pts=runtime.createBuffer(16),out=runtime.createBuffer(256),arena=runtime.createObjectArena(),imports={x0,y0,x1,y1};
 try{
  runtime.batch().dispatch(kernels.bind_points.bind({pts,...imports},{},{objectArena:arena}),[1]).submit();
  for(const selector of [1,0,1]){runtime.batch().dispatch(kernels.update_points.bind({pts},{selector},{objectArena:arena}),[1]).submit();await runtime.idle();}
  for(const selector of [0,1]){runtime.batch().dispatch(kernels.read_points.bind({pts,out},{selector},{objectArena:arena}),[1]).submit();const values=await runtime.read(out);for(let i=0;i<32;i++){const x=selector?202+i:3+i,y=selector?296+i:100+i;if(values[i*2]!==x||values[i*2+1]!==y)throw Error('Captured Points result mismatch');}}
  const prefix=await runtime.read(x0,Float32Array,8);if(prefix[0]!==0||prefix[1]!==1)throw Error('Captured offset damaged prefix');
  const other=runtime.createObjectArena();let rejected=false;try{kernels.read_points.bind({pts,out,...imports},{selector:0},{objectArena:other});}catch(e){rejected=/another arena/.test(e.message);}finally{other.dispose();}if(!rejected)throw Error('Cross-arena captured pointers were accepted');
  const guardSource='class View {float* p;public:__device__ View(float* a):p(a){}__device__ float get(unsigned i)const{return p[i];}__device__ void put(int i,float v){p[i]=v;}};__global__ void bounds(float* input,float2* output){View v(input+2);v.put(-1,7.f);v.put(999,8.f);output[0]=make_float2(v.get(4294967295u),v.get(0));}';
  const guard=await runtime.kernel(guardSource,{entry:'bounds',objectHeap:'persistent',workgroupSize:[1]}),guardArena=runtime.createObjectArena(),input=runtime.createBuffer(new Float32Array([1,2,3,4])),output=runtime.createBuffer(8);
  try{runtime.batch().dispatch(guard.bind({input,output},{},{objectArena:guardArena}),[1]).submit();const values=await runtime.read(output),storage=await runtime.read(input);if(values[0]!==0||values[1]!==3||storage.join(',')!=='1,7,3,4')throw Error('Captured reference bounds or unsigned index wrapped');}finally{guardArena.dispose();runtime.destroyBuffer(input);runtime.destroyBuffer(output);}
  return {boundsVerified:true,points:64,updates:3,originalPointsMethods:true,nonzeroOffsets:true,separateSubmissions:true,crossArenaRejected:true};
 }finally{arena.dispose();for(const b of [x0,y0,x1,y1,pts,out])runtime.destroyBuffer(b);}
}
