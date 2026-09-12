export async function checkPathtracerValues(runtime){
 const source=await(await fetch('/tests/pathtracer-value-class.cuh')).text()+`
 __global__ void values(float*out){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  vec3 a(float(i)*0.25f,float(i%7)-3.0f,float(i%11)*0.5f),b=a;
  a.e[0]=-99.0f;
  out[i*38]=b.x();out[i*38+1]=b.y();out[i*38+2]=b.z();
  out[i*38+3]=b.r();out[i*38+4]=b.g();out[i*38+5]=b.b();out[i*38+6]=b.squared_length();
  vec3 positive=+b,negative=-b;out[i*38+7]=positive.x();out[i*38+8]=negative.x();out[i*38+9]=negative.y();out[i*38+10]=negative.z();out[i*38+11]=b[i%3];vec3 assigned;assigned=b;b.e[1]=-88.0f;out[i*38+12]=assigned.y();out[i*38+13]=b.y();assigned[i%3]=float(i)*0.5f;assigned[(i+1)%3]+=2.0f;out[i*38+14]=assigned.x();out[i*38+15]=assigned.y();out[i*38+16]=assigned.z();vec3 rhs(2.0f,4.0f,8.0f);vec3 step(8.0f,16.0f,32.0f);step+=rhs;out[i*38+17]=step.x();out[i*38+18]=step.y();out[i*38+19]=step.z();step-=rhs;out[i*38+20]=step.x();out[i*38+21]=step.y();out[i*38+22]=step.z();step*=rhs;out[i*38+23]=step.x();out[i*38+24]=step.y();out[i*38+25]=step.z();step/=rhs;out[i*38+26]=step.x();out[i*38+27]=step.y();out[i*38+28]=step.z();step*=2.0f;out[i*38+29]=step.x();out[i*38+30]=step.y();out[i*38+31]=step.z();step/=2.0f;out[i*38+32]=step.x();out[i*38+33]=step.y();out[i*38+34]=step.z();vec3 unit(3.0f,0.0f,4.0f);unit.make_unit_vector();out[i*38+35]=unit.x();out[i*38+36]=unit.y();out[i*38+37]=unit.z();
 }`;
 const kernel=await runtime.kernel(source,{entry:'values',workgroupSize:[64,1,1]}),out=runtime.createBuffer(new Float32Array(512*38));
 try{runtime.batch().dispatch(kernel.bind({out},{}),[8,1,1]).submit();const actual=await runtime.read(out),expected=new Float32Array(await(await fetch('/reports/pathtracer-value-native.bin')).arrayBuffer());if(actual.length!==expected.length||actual.some((v,i)=>v!==expected[i]))throw Error('Value-class native mismatch '+JSON.stringify({length:actual.length,expectedLength:expected.length,first:Array.from(actual).map((v,i)=>({i,actual:v,expected:expected[i]})).filter(x=>x.actual!==x.expected).slice(0,8)}));return {cases:512,values:actual.length,nativeExact:true,copyIsolation:true,fullPathtracerSupported:false};}finally{runtime.destroyBuffer(out);}
}
