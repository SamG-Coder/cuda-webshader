export async function checkChronoEos(runtime){
 const load=p=>fetch(new URL(p,import.meta.url)),source=await(await load('chrono-integration.cu')).text()+'\n'+await(await load('chrono-eos-probe.cuh')).text(),input=new Float32Array((await(await load('../reports/chrono-rk2-half-native.bin')).arrayBuffer()).slice(30327*28,30327*44)),native=new Float32Array(await(await load('../reports/chrono-eos-native.bin')).arrayBuffer()),params=await(await load('../reports/chrono-params.json')).json(),n=input.length/4;
 if(n!==30327||native.length!==input.length)throw Error('Incomplete equation-of-state native reference');
 const kernel=await runtime.kernel(source,{entry:'eosProbe',defines:{__CUDA_ARCH__:1},workgroupSize:[128]}),a=runtime.createBuffer(input),b=runtime.createBuffer(input.byteLength),info=kernel.artifact.metadata.diagnostics,d=runtime.createBuffer((2+info.capacity*info.strideWords)*4);
 try{runtime.batch().dispatch(kernel.bind({input:a,output:b,[info.buffer]:d},{...params,n}),[Math.ceil(n/128)]).submit();const actual=await runtime.read(b,Float32Array),sections=['ratio','pow','pressure','B'].map(name=>({name,unequal:0,max:0,examples:[]}));let nativeOracleMismatch=0,gpuOracleMismatch=0;
 for(let i=0;i<actual.length;i++){const s=sections[i%4],error=Math.abs(actual[i]-native[i]);if(actual[i]!==native[i]){s.unequal++;s.max=Math.max(s.max,error);if(s.examples.length<5)s.examples.push({i,actual:actual[i],native:native[i],error,inputDensity:input[i-i%4],nativeQ:native[i-i%4]});}}
 for(let i=0;i<n;i++){const q=native[i*4],oracle=Math.fround(Math.pow(q,7));if(native[i*4+1]!==oracle)nativeOracleMismatch++;if(actual[i*4+1]!==oracle)gpuOracleMismatch++;}
 if(sections.some(s=>s.unequal)||nativeOracleMismatch||gpuOracleMismatch)throw Error('Original Eos/native mismatch');return {n,sections,nativeOracleMismatch,gpuOracleMismatch};
 }finally{runtime.destroyBuffer(a);runtime.destroyBuffer(b);runtime.destroyBuffer(d);}
}
