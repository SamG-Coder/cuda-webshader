export async function checkPathtracerCamera(runtime){
 const source=(await Promise.all(['value-class','camera-class','camera-kernels'].map(async n=>await(await fetch('/tests/pathtracer-'+n+'.cuh')).text()))).join('\n');
 const kernel=await runtime.kernel(source,{entry:'check_camera',workgroupSize:[64,1,1],libraries:['curand-xorwow']});
 const bits=runtime.createBuffer(512*9*4),out=runtime.createBuffer(512*23*4);
 try{
  runtime.batch().dispatch(kernel.bind({bits,out}),[8,1,1]).submit();
  const actualBits=await runtime.read(bits,Uint32Array),actual=await runtime.read(out),expectedBits=new Uint32Array(await(await fetch('/reports/pathtracer-camera-native-bits.bin')).arrayBuffer()),expected=new Float32Array(await(await fetch('/reports/pathtracer-camera-native.bin')).arrayBuffer());
  for(let i=0;i<actualBits.length;i++)if(actualBits[i]!==expectedBits[i])throw Error('RNG state mismatch '+i+': '+actualBits[i]+' vs '+expectedBits[i]);
  let maxError=0;for(let i=0;i<actual.length;i++){const error=Math.abs(actual[i]-expected[i]);maxError=Math.max(maxError,error);if(!Number.isFinite(actual[i])||error>(i%23<8?0:0.00002))throw Error('Camera mismatch '+i+': '+actual[i]+' vs '+expected[i]);}
  return {cameras:512,randomIntegers:actualBits.length,randomIntegersExact:true,uniformFloatsExact:true,floatValues:actual.length,maxAbsoluteError:maxError,fullPathtracerSupported:false};
 }finally{runtime.destroyBuffer(bits);runtime.destroyBuffer(out);}
}
