export async function checkPathtracerValues(runtime){
 const source=await(await fetch('/tests/pathtracer-value-class.cuh')).text()+`
 __global__ void values(float*out){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  vec3 a(float(i)*0.25f,float(i%7)-3.0f,float(i%11)*0.5f),b=a;
  a.e[0]=-99.0f;
  out[i*7]=b.x();out[i*7+1]=b.y();out[i*7+2]=b.z();
  out[i*7+3]=b.r();out[i*7+4]=b.g();out[i*7+5]=b.b();out[i*7+6]=b.squared_length();
 }`;
 const kernel=await runtime.kernel(source,{entry:'values',workgroupSize:[64,1,1]}),out=runtime.createBuffer(new Float32Array(512*7));
 try{runtime.batch().dispatch(kernel.bind({out},{}),[8,1,1]).submit();const actual=await runtime.read(out),expected=new Float32Array(await(await fetch('/reports/pathtracer-value-native.bin')).arrayBuffer());if(actual.length!==expected.length||actual.some((v,i)=>v!==expected[i]))throw Error('Value-class native mismatch');return {cases:512,values:actual.length,nativeExact:true,copyIsolation:true,fullPathtracerSupported:false};}finally{runtime.destroyBuffer(out);}
}
