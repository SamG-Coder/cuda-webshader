export async function checkPathtracerValues(runtime){
 const source=await(await fetch('/tests/pathtracer-value-class.cuh')).text()+`
 __global__ void values(float*out){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  vec3 a(float(i)*0.25f,float(i%7)-3.0f,float(i%11)*0.5f),b=a;
  a.e[0]=-99.0f;
  out[i*14]=b.x();out[i*14+1]=b.y();out[i*14+2]=b.z();
  out[i*14+3]=b.r();out[i*14+4]=b.g();out[i*14+5]=b.b();out[i*14+6]=b.squared_length();
  vec3 positive=+b,negative=-b;out[i*14+7]=positive.x();out[i*14+8]=negative.x();out[i*14+9]=negative.y();out[i*14+10]=negative.z();out[i*14+11]=b[i%3];vec3 assigned;assigned=b;b.e[1]=-88.0f;out[i*14+12]=assigned.y();out[i*14+13]=b.y();
 }`;
 const kernel=await runtime.kernel(source,{entry:'values',workgroupSize:[64,1,1]}),out=runtime.createBuffer(new Float32Array(512*14));
 try{runtime.batch().dispatch(kernel.bind({out},{}),[8,1,1]).submit();const actual=await runtime.read(out),expected=new Float32Array(await(await fetch('/reports/pathtracer-value-native.bin')).arrayBuffer());if(actual.length!==expected.length||actual.some((v,i)=>v!==expected[i]))throw Error('Value-class native mismatch '+JSON.stringify({length:actual.length,expectedLength:expected.length,first:Array.from(actual).map((v,i)=>({i,actual:v,expected:expected[i]})).filter(x=>x.actual!==x.expected).slice(0,8)}));return {cases:512,values:actual.length,nativeExact:true,copyIsolation:true,fullPathtracerSupported:false};}finally{runtime.destroyBuffer(out);}
}
