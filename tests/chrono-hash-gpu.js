export async function checkChronoHash(runtime){
 const load=name=>fetch(new URL(name,import.meta.url));
 const source=await(await load('chrono-hash.cu')).text(),params=await(await load('../reports/chrono-params.json')).json(),points=new Float32Array(await(await load('../reports/chrono-hash-input.bin')).arrayBuffer()),native=new Uint32Array(await(await load('../reports/chrono-hash-native.bin')).arrayBuffer()),n=points.length/3;
 if(n!==16739||native.length!==n*4*9)throw Error('Incomplete Chrono native hash reference');
 const kernel=await runtime.kernel(source,{entry:'hashProbe',workgroupSize:[128]}),buffers={points:runtime.createBuffer(points),hashes:runtime.createBuffer(n*4),bins:runtime.createBuffer(n*12)};let compared=0;
 if(kernel.artifact.metadata.bindings.find(b=>b.name==='points').stride!==12)throw Error('Chrono Real3 layout changed');
 for(const scalar of kernel.artifact.metadata.scalars)if(scalar.origin==='constant'&&!Object.hasOwn(params,scalar.name))throw Error('Missing captured parameter '+scalar.name);
 try{for(let mode=-1;mode<8;mode++){
  const values={...params,n};if(mode>=0)for(const [i,axis]of [...'xyz'].entries())values['constant.paramsD.'+axis+'_periodic']=!!(mode&(1<<i));
  runtime.batch().dispatch(kernel.bind(buffers,values),[Math.ceil(n/128)]).submit();
  let offset=(mode+1)*n*4;
  for(const key of ['hashes','bins']){const actual=await runtime.read(buffers[key],Uint32Array);for(let i=0;i<actual.length;i++){if(actual[i]!==native[offset+i])throw Error(`Chrono ${key}, mode ${mode}, index ${i}: ${actual[i]} != ${native[offset+i]}`);compared++;}offset+=actual.length;}
 }}finally{for(const b of Object.values(buffers))runtime.destroyBuffer(b);}
 const precision=await runtime.kernel(source,{entry:'doubleFieldProbe',workgroupSize:[1]}),out=runtime.createBuffer(4),bits=new DataView(new ArrayBuffer(8));bits.setFloat64(0,1+2**-40,true);
 try{runtime.batch().dispatch(precision.bind({output:out},{...params,'constant.paramsD.pressure_height.lo':bits.getUint32(0,true),'constant.paramsD.pressure_height.hi':bits.getUint32(4,true)}),[1]).submit();if((await runtime.read(out))[0]!==2**-40)throw Error('Double constant field was demoted to float');}finally{runtime.destroyBuffer(out);}
 return {nativeParticles:16731,edgePoints:8,boundaryConfigurations:9,compared,nativeExact:true,fullParameterRecord:true,real3Stride:12,doubleFieldPreserved:true};
}
