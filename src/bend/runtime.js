export const runtimeErrors = {
  0:'OK', 1:'Arena exhausted', 2:'Step limit exceeded', 3:'Invalid runtime value',
  4:'Unreachable match reached', 5:'Nat exceeds the 32-bit runtime limit', 6:'Result type mismatch',
};

export function packBendInputs(program,rows) {
  if(!Array.isArray(rows)||rows.length<1||rows.length>1024) throw Error('Supply between 1 and 1,024 input rows.');
  const words=new Uint32Array(Math.max(1,rows.length*program.parameters.length));
  const floats=new Float32Array(words.buffer);
  rows.forEach((row,r)=>{
    if(!Array.isArray(row)||row.length!==program.parameters.length) throw Error(`Row ${r+1} needs ${program.parameters.length} arguments.`);
    row.forEach((value,c)=>{
      const p=program.parameters[c],i=r*row.length+c;
      if(typeof value!=='number'||!Number.isFinite(value)) throw Error(`Row ${r+1}, ${p.name}: expected a finite number.`);
      if(p.type==='F32') {
        if(!Number.isFinite(Math.fround(value))) throw Error(`${p.name} exceeds F32 range.`);
        floats[i]=value;
      } else {
        if(!Number.isInteger(value)||value<0||value>0xffffffff) throw Error(`${p.name} must be an unsigned 32-bit integer.`);
        words[i]=value;
      }
    });
  });
  return words;
}

export async function runBend(runtime,compiled,rows,{arenaWords=65536,maxSteps=100000}={}) {
  const inputs=packBendInputs(compiled.program,rows), count=rows.length;
  if(!Number.isInteger(arenaWords)||arenaWords<64||arenaWords>1048576||arenaWords%4) throw Error('Arena words must be a multiple of four between 64 and 1,048,576.');
  if(!Number.isInteger(maxSteps)||maxSteps<1||maxSteps>1000000) throw Error('Step limit must be between 1 and 1,000,000.');
  const bytes=count*arenaWords*4;
  if(bytes>Math.min(128*1024*1024,runtime.device.limits.maxStorageBufferBindingSize)) throw Error('The requested arenas exceed the 128 MiB/device buffer limit.');
  const kernel=await runtime.kernel(compiled.artifact), buffers={}, owned=[];
  const allocate=(name,value)=>{const b=runtime.createBuffer(value,{label:'Bend '+name});owned.push(b);buffers[name]=b;};
  try {
    allocate('inputs',inputs);allocate('outputs',count*4);allocate('status',count*4);
    allocate('usage',count*8);allocate('heap',bytes);
    const start=performance.now();
    runtime.batch().dispatch(kernel.bind(buffers,{count,arena_words:arenaWords,max_steps:maxSteps}),[Math.ceil(count/64),1,1]).submit();
    const [output,status,usage]=await Promise.all([
      runtime.read(buffers.outputs,compiled.program.resultType==='F32'?Float32Array:Uint32Array),
      runtime.read(buffers.status,Uint32Array),runtime.read(buffers.usage,Uint32Array),
    ]);
    return {output:[...output],status:[...status],usage:[...usage],elapsedMs:performance.now()-start,
      timing:'dispatch plus readback; excludes compilation',arenaWords,maxSteps,
      execution:'WebGPU; one invocation per input row; internal fork/join is sequential'};
  } finally {
    await runtime.idle();
    for(const b of owned) runtime.destroyBuffer(b);
  }
}
