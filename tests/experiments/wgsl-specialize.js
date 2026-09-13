import {trimWgsl} from './wgsl-trim.js';
// Test-only explicit specialization. Values are supplied by the caller, never
// chosen by application/entry name. The binding guard is part of the contract.
export function specializeIntegerUniforms(input,values){
 let source=input.wgsl;const fixed=[],helpers=[];
 if(/\bcw_trim_fixed_\d+\b/.test(source))throw Error('Already specialized');
 for(const scalar of input.metadata.scalars){
  if(scalar.origin!=='constant'||!['i32','u32'].includes(scalar.type)||!Object.hasOwn(values,scalar.name))continue;
  const value=Number(values[scalar.name]);
  if(!Number.isInteger(value)||value<(scalar.type==='u32'?0:-2147483648)||value>(scalar.type==='u32'?0xffffffff:2147483647)||(scalar.sourceType==='bool'&&value!==0&&value!==1))throw Error('Invalid integer specialization');
  const field=scalar.field??('p_'+scalar.name),name='cw_trim_fixed_'+fixed.length,escaped=field.replace(/[.*+?^${}()|[\]\\]/g,'\\$&');
  const pattern=new RegExp('\\bcw_params\\.'+escaped+'\\b','g');if(!pattern.test(source))continue;pattern.lastIndex=0;
  source=source.replace(pattern,name+'()');helpers.push(`fn ${name}()->${scalar.type}{return ${scalar.type==='u32'?value+'u':'bitcast<i32>('+(value>>>0)+'u)'};}`);fixed.push({name:scalar.name,value,defaultValue:scalar.defaultValue});
 }
 return {artifact:trimWgsl({...input,wgsl:source+'\n'+helpers.join('\n')}).artifact,fixed};
}
export function guardSpecializedKernel(kernel,fixed){
 const check=values=>{for(const scalar of fixed){const value=Object.hasOwn(values,scalar.name)?values[scalar.name]:scalar.defaultValue;if(Number(value)!==scalar.value)throw Error('Recompile after changing specialized scalar '+scalar.name);}};
 const bind=kernel.bind.bind(kernel);
 kernel.bind=(buffers,values={},options={})=>{
  check(values);if(fixed.some(s=>Object.hasOwn(options.scalarBuffers??{},s.name)))throw Error('Specialized scalars cannot use GPU counters');
  const invocation=bind(buffers,values,options),set=invocation.setScalars.bind(invocation);
  invocation.setScalars=update=>{check({...invocation.values,...update});return set(update);};return invocation;
 };return kernel;
}
