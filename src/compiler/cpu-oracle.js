/**
 * Deterministic typed-AST interpreter for frontend/reference testing ONLY.
 * This is not a WebGPU fallback and does not validate emitted WGSL or measure GPU performance.
 * Lanes run as cooperative generators, yielding at each __syncthreads site.
 */
import {isArray, vectorLength, vectorElement, walk} from './compiler.js';
const f = Math.fround;
function convert(value,type){
  if(type==='cw_size64')return BigInt.asUintN(64,BigInt(value));
  if(type==='cw_extent')return structuredClone(value);
  if(typeof type==='string'&&type.startsWith('cw_struct_'))return structuredClone(value);
  if(type==='cw_short')return (Number(value)<<16)>>16;
  if(type==='cw_ushort')return Number(value)&65535;
  if(type==='cw_uchar')return Number(value)&255;
  if(type==='cw_uchar4')return Number(value)>>>0;
  if(type==='cw_f64')return Number(value);if(type==='f32')return f(Number(value));if(type==='u32')return typeof value==='bigint'?Number(BigInt.asUintN(32,value)):Number(value)>>>0;if(type==='i32')return typeof value==='bigint'?Number(BigInt.asIntN(32,value)):Number(value)|0;if(type==='bool')return !!value;
  const size=vectorLength(type);if(size){if(!Array.isArray(value)||value.length!==size)throw new Error('Invalid vector value.');return value.map(v=>convert(v,vectorElement(type)));}return value;
}
function zero(type,structs=[]){const spec=structs.find(s=>s.type===type);if(spec)return Object.fromEntries(spec.fields.map(f=>[f.name,zero(f.resolvedType,structs)]));if(isArray(type))return Array.from({length:type.length},()=>zero(type.element,structs));const n=vectorLength(type);return n?Array(n).fill(0):type==='bool'?false:0;}
class BufferView {
  constructor(data,type,offset=0){this.data=data;this.type=type;this.offset=offset;this.records=Array.isArray(data);this.width=this.records?1:vectorLength(type)||1;this.length=data.length/this.width;if(!Number.isInteger(this.length))throw new Error('Buffer record count is not integral.');}
  check(i){if(!Number.isInteger(i)||i+this.offset<0||i+this.offset>=this.length)throw new RangeError(`CPU oracle detected out-of-bounds access at ${i}, offset ${this.offset}, length ${this.length}.`);}
  get(i){this.check(i);i+=this.offset;return this.records?structuredClone(this.data[i]):this.width===1?this.data[i]:Array.from(this.data.subarray(i*this.width,(i+1)*this.width));}
  set(i,v){this.check(i);i+=this.offset;if(this.width===1)this.data[i]=convert(v,this.type);else this.data.set(convert(v,this.type),i*this.width);}
}
function binary(op,a,b,type){
  if(vectorLength(type))return Array.from({length:vectorLength(type)},(_,i)=>binary(op,Array.isArray(a)?a[i]:a,Array.isArray(b)?b[i]:b,vectorElement(type)));
  if(op==='&&')return !!a&&!!b;if(op==='||')return !!a||!!b;
  a=convert(a,type);b=convert(b,type);
  if(type==='cw_size64'&&op==='*')return BigInt.asUintN(64,a*b);
  switch(op){
    case '+':return convert(a+b,type);case '-':return convert(a-b,type);case '*':return convert(['f32','cw_f64'].includes(type)?a*b:Math.imul(a,b),type);
    case '/':if(!b&&!['f32','cw_f64'].includes(type))throw new Error('Integer division by zero.');return convert(['f32','cw_f64'].includes(type)?a/b:Math.trunc(a/b),type);
    case '%':if(!b)throw new Error('Integer remainder by zero.');return convert(a%b,type);
    case '<':return a<b;case '>':return a>b;case '<=':return a<=b;case '>=':return a>=b;case '==':return a===b;case '!=':return a!==b;
    case '&':return convert(a&b,type);case '|':return convert(a|b,type);case '^':return convert(a^b,type);
    case '<<':return convert(a<<b,type);case '>>':return type==='u32'?a>>>b:a>>b;
    default:throw new Error(`Unimplemented binary operator ${op}`);
  }
}
class Context {
  constructor(artifact,env,ids,budget){this.artifact=artifact;this.env=env;this.ids=ids;this.budget=budget;this.steps=0;}
  tick(){if(++this.steps>this.budget)throw new Error('CPU oracle instruction budget exceeded; possible nonterminating kernel.');}
  *ref(n){
    if(n.kind==='id'){const cell=this.env.get(n.symbol);if(!cell)throw new Error(`Uninitialized symbol ${n.name}`);return {get:()=>cell.value,set:v=>{cell.value=convert(v,n.type);}};}
    if(n.kind==='index'){
      const base=yield* this.eval(n.base),i=yield* this.eval(n.index);
      if(base instanceof BufferView)return {get:()=>base.get(i),set:v=>base.set(i,v)};
      if(!Number.isInteger(i)||i<0||i>=base.length)throw new RangeError(`Shared/local index ${i} is out of bounds (${base.length}).`);
      return {get:()=>base[i],set:v=>{base[i]=convert(v,n.type);}};
    }
    if(n.kind==='member'){
      const reference=yield* this.ref(n.base),structure=typeof n.base.type==='string'&&n.base.type.startsWith('cw_struct_'),i=structure?n.member:'xyzw'.indexOf(n.member);
      if(n.base.type==='cw_uchar4'){const shift=i*8;return {get:()=>(reference.get()>>>shift)&255,set:v=>reference.set(((reference.get()&~(255<<shift))|((Number(v)&255)<<shift))>>>0)};}
      if(structure)return {get:()=>reference.get()[i],set:v=>{const copy=structuredClone(reference.get());copy[i]=convert(v,n.type);reference.set(copy);}};
      return {get:()=>reference.get()[i],set:v=>{const copy=[...reference.get()];copy[i]=convert(v,n.type);reference.set(copy);}};
    }
    throw new Error(`Expression ${n.kind} is not an lvalue.`);
  }
  *eval(n){
    if(n.kind==='sequence'){let result;for(const expression of n.expressions)result=yield* this.eval(expression);return result;}
    this.tick();
    switch(n.kind){
      case 'initializer':{const values=[];for(const item of n.items)values.push(yield* this.eval(item));while(values.length<Number(n.type[3]))values.push(0);return values;}
      case 'byte-index':{const offset=yield* this.eval(n.offset);if(typeof offset==='bigint'){const index=offset/BigInt(n.resolvedStride);return Number(index>2147483647n?2147483647n:index);}return Math.trunc(offset/n.resolvedStride);}
      case 'sizeof':return BigInt(n.numericValue);
      case 'literal':return convert(n.numericValue,n.type);
      case 'id':if(n.name==='true')return true;if(n.name==='false')return false;return this.env.get(n.symbol)?.value;
      case 'index':if(n.pointerArrayElement){const slots=yield* this.eval(n.base),index=yield* this.eval(n.index);if(!Number.isInteger(index)||index<0||index>=slots.length)throw Error('Pointer slot out of bounds.');return new BufferView(this.env.get(n.pointerBaseSymbol).value,n.pointerBaseSymbol.type.element,slots[index]);}return (yield* this.ref(n)).get();
      case 'member':{
        if(n.base.type==='cw_extent')return (yield* this.eval(n.base))[n.member];
        if(n.base.kind==='id'&&this.ids[n.base.name])return this.ids[n.base.name]['xyz'.indexOf(n.member)];
        const base=yield* this.eval(n.base);if(n.base.type==='cw_uchar4')return(base>>>('xyzw'.indexOf(n.member)*8))&255;return base[typeof n.base.type==='string'&&n.base.type.startsWith('cw_struct_')?n.member:'xyzw'.indexOf(n.member)];
      }
      case 'ptx-sad4':{const values=[];for(const a of n.args)values.push((yield* this.eval(a))>>>0);let sum=values[2];for(let shift=0;shift<32;shift+=8)sum+=Math.abs(((values[0]>>>shift)&255)-((values[1]>>>shift)&255));return sum>>>0;}
      case 'cast':if(n.byteScale!==undefined)return convert((yield* this.eval(n.byteScaleValue))*n.byteScale,n.target);return convert(yield* this.eval(n.value),n.target);
      case 'unary':{
        if(['++','--'].includes(n.op)){const r=yield* this.ref(n.value),old=r.get(),value=convert(old+(n.op==='++'?1:-1),n.type);r.set(value);return n.prefix?value:old;}
        if(n.op==='&')return yield* this.ref(n.value);
        const value=yield* this.eval(n.value);return n.op==='!'?!value:convert(n.op==='-'?-value:n.op==='~'?~value:value,n.type);
      }
      case 'binary':{
        const a=yield* this.eval(n.left);if(n.op==='&&'&&!a)return false;if(n.op==='||'&&a)return true;
        const b=yield* this.eval(n.right);return binary(n.op,a,b,n.operandType||n.type);
      }
      case 'conditional':return convert(yield* this.eval((yield* this.eval(n.condition))?n.yes:n.no),n.type);
      case 'assign':{
        if(n.pointerArrayAssignment){const target=yield* this.ref(n.left),offset=yield* this.eval(n.right.value.index);target.set(convert(offset,'i32'));return offset;}
        if(n.pointerRebind){let base=n.right.pointerArrayElement?yield* this.eval(n.right):this.env.get(n.right.pointerBaseSymbol)?.value;const offset=n.right.pointerOffset?yield* this.eval(n.right.pointerOffset):0;if(Array.isArray(base))base=new BufferView(base,n.right.pointerBaseSymbol.type.element);if(!(base instanceof BufferView))throw Error('Expected pointer allocation.');const result=new BufferView(base.data,base.type,convert(base.offset+convert(offset,'i32'),'i32'));this.env.get(n.left.symbol).value=result;return result;}
        if(n.pointerShift){const cell=this.env.get(n.left.symbol),base=cell.value,delta=yield* this.eval(n.right);if(!(base instanceof BufferView))throw Error('Expected a storage pointer.');const offset=binary(n.op==='+='?'+':'-',base.offset,convert(delta,'i32'),'i32');cell.value=new BufferView(base.data,base.type,offset);return cell.value;}
        let target,value;if(n.packedAtomicAssignment){value=yield* this.eval(n.right);target=yield* this.ref(n.left);}else{target=yield* this.ref(n.left);value=yield* this.eval(n.right);}
        target.set(n.op==='='?value:binary(n.op.slice(0,-1),target.get(),value,n.operandType||n.type));return target.get();
      }
      case 'call':return yield* this.call(n);
      default:throw new Error(`CPU oracle: unsupported expression ${n.kind}.`);
    }
  }
  *call(n){
    const name=n.callName;
    if(name==='__syncthreads'){yield n.token.offset;return;}
    const args=[];for(const [i,a] of n.args.entries()){
      if(n.localPointerArgs?.[i])args.push(yield* this.ref(a.value));
      else if(n.pointerArgs?.[i]&&a.pointerArrayElement)args.push(yield* this.eval(a));
      else if(n.pointerArgs?.[i]){let base=this.env.get(a.pointerBaseSymbol)?.value;const offset=a.pointerOffset?yield* this.eval(a.pointerOffset):0;if(Array.isArray(base))base=new BufferView(base,a.pointerBaseSymbol.type.element);if(!(base instanceof BufferView)||!Number.isInteger(offset) )throw new RangeError('CPU helper pointer needs an integer offset.');args.push(new BufferView(base.data,base.type,convert(base.offset+convert(offset,'i32'),'i32')));}
      else if(n.constRefTemporaries?.[i]){let value=yield* this.eval(a);args.push({get:()=>value,set:v=>{value=v;}});}
      else args.push(n.groupArgs?.[i]?null:yield* (n.referenceArgs?.[i]?this.ref(a):this.eval(a)));
    }
    if(['sincosf','__sincosf'].includes(name)){const phase=f(args[0]),s=f(Math.sin(phase)),c=f(Math.cos(phase));args[1].set(s);args[2].set(c);return;}
    if(name==='__ffs'){const x=args[0]|0;return x===0?0:32-Math.clz32(x&-x);}
    if(name==='__mul24')return Math.imul((args[0]<<8)>>8,(args[1]<<8)>>8);
    if(name==='__umul24')return Math.imul(args[0]&0xffffff,args[1]&0xffffff)>>>0;
    if(name==='atomicCAS'){const old=args[0].get();if(old===convert(args[1],n.type))args[0].set(convert(args[2],n.type));return old;}
    if(['atomicAdd','atomicMin','atomicMax','atomicExch'].includes(name)){
      const old=args[0].get(),value=name==='atomicAdd'?old+args[1]:name==='atomicMin'?Math.min(old,args[1]):name==='atomicMax'?Math.max(old,args[1]):args[1];args[0].set(value);return old;
    }
    if(name==='make_uchar4')return args.reduce((packed,v,i)=>packed|((Number(v)&255)<<(i*8)),0)>>>0;
    if(['float','int','uint','bool','uchar'].includes(name))return convert(args[0],n.type);
    if(name==='make_float3'&&Array.isArray(args[0]))return args[0].slice(0,3);
    if(name==='make_float4'&&Array.isArray(args[0]))return [...args[0],args[1]];
    if(name==='length')return f(Math.sqrt(args[0].reduce((sum,a)=>f(sum+f(a*a)),0)));
    if(name==='dot'||name==='normalize'){const sum=args[0].reduce((sum,a,i)=>f(sum+f(a*(name==='dot'?args[1][i]:a))),0);return name==='dot'?sum:args[0].map(a=>f(a/Math.sqrt(sum)));}
    if(['fminf','fmaxf'].includes(name)&&Array.isArray(args[0]))return args[0].map((a,i)=>f(name==='fminf'?Math.min(a,args[1][i]):Math.max(a,args[1][i])));
    if(/^make_(float|uint|int)[234]$/.test(name))return (args.length===1?Array(vectorLength(n.type)).fill(args[0]):args).map(v=>convert(v,vectorElement(n.type)));
    if(name==='__fdividef')return f(args[0]/args[1]);
    if(name==='__saturatef')return f(Number.isNaN(args[0])?0:Math.max(0,Math.min(1,args[0])));
    if(name==='sqrt')return f(Math.sqrt(args[0]));
    if(name==='rintf'){const x=args[0],lo=Math.floor(x),fraction=x-lo,result=fraction>.5||fraction===.5&&lo%2!==0?lo+1:lo;return result===0&&x<0?-0:result;}
    if(name==='roundf'){const x=args[0],whole=Math.trunc(x);return f(Math.abs(x-whole)>=.5?whole+(x>=0?1:-1):whole);}
    if(name==='abs')return Math.abs(args[0])|0;
    const unary={sinf:Math.sin,cosf:Math.cos,tanf:Math.tan,sqrtf:Math.sqrt,rsqrtf:x=>1/Math.sqrt(x),exp:Math.exp,expf:Math.exp,__expf:Math.exp,exp2f:x=>2**x,logf:Math.log,__logf:Math.log,log2f:Math.log2,fabs:Math.abs,fabsf:Math.abs,floorf:Math.floor,ceilf:Math.ceil,truncf:Math.trunc};
    if(unary[name])return f(unary[name](args[0]));
    if(['fminf','fmaxf','min','max','powf','atan2f','fmaf'].includes(name)){
      const value=name==='fmaf'?args[0]*args[1]+args[2]:['fminf','min'].includes(name)?Math.min(...args):['fmaxf','max'].includes(name)?Math.max(...args):name==='powf'?Math.pow(...args):Math.atan2(...args);
      return convert(value,n.type);
    }
    const helper=this.artifact.ast.functions.find(x=>x.name===name);
    if(!helper)throw new Error(`No CPU implementation of ${name}.`);
    const env=new Map([...this.env].filter(([symbol])=>['constant-global','shared'].includes(symbol.kind)));helper.params.forEach((p,i)=>env.set(p.symbol,p.reference?{get value(){return args[i].get();},set value(v){args[i].set(v);}}:{value:p.pointer?args[i]:convert(args[i],p.type)}));
    const child=new Context(this.artifact,env,this.ids,this.budget-this.steps),result=yield* child.statement(helper.body);this.steps+=child.steps;
    return convert(result?.value,helper.result);
  }
  *statement(n){
    this.tick();
    switch(n.kind){
      case 'block':for(const s of n.body){const signal=yield* this.statement(s);if(signal)return signal;}return;
      case 'decl':if(n.aliasBase){let base=this.env.get(n.aliasBase).value;if(Array.isArray(base))base=new BufferView(base,n.aliasBase.type.element);const offset=convert(base.offset+convert(n.aliasOffset?yield* this.eval(n.aliasOffset):0,'i32'),'i32');if(!Number.isInteger(offset))throw new RangeError('CPU alias needs an integer offset.');this.env.set(n.symbol,{value:new BufferView(base.data,base.type,offset)});}else if(!n.shared)this.env.set(n.symbol,{value:n.init?convert(yield* this.eval(n.init),n.resolvedType):zero(n.resolvedType,this.artifact.ast.structs)});return;
      case 'decls':for(const d of n.declarations)yield* this.statement(d);return;
      case 'expr':yield* this.eval(n.value);return;
      case 'if':if(yield* this.eval(n.condition))return yield* this.statement(n.yes);else if(n.no)return yield* this.statement(n.no);return;
      case 'do':{do{const signal=yield* this.statement(n.body);if(signal?.control==='return')return signal;if(signal?.control==='break')break;this.tick();}while(yield* this.eval(n.condition));return;}
      case 'for':case 'while':{
        if(n.init){if(['decl','decls'].includes(n.init.kind))yield* this.statement(n.init);else yield* this.eval(n.init);}
        while(!n.condition||(yield* this.eval(n.condition))){const signal=yield* this.statement(n.body);if(signal?.control==='return')return signal;if(signal?.control==='break')break;if(n.step)yield* this.eval(n.step);this.tick();}return;
      }
      case 'return':return {control:'return',value:n.value?yield* this.eval(n.value):undefined};
      case 'break':case 'continue':return {control:n.kind};
      case 'empty':return;
      case 'thread-block':return;
      default:throw new Error(`CPU oracle: unsupported statement ${n.kind}.`);
    }
  }
}
export function executeCPU(artifact,buffers,scalars,workgroups,{instructionBudget=1_000_000}={}){
  for(const c of artifact.metadata.scalarConstraints||[]){const v=scalars[c.name];if(!Number.isInteger(v)||v<c.minimum||v%c.multipleOf!==0)throw new RangeError(`${c.name} must be a nonnegative multiple of ${c.multipleOf} for full-workgroup execution.`);}
  buffers={...buffers};for(const [alias,target]of Object.entries(artifact.metadata.bufferAliases||{})){if(Object.hasOwn(buffers,alias)&&buffers[alias]!==buffers[target])throw Error('Declared CPU buffer alias must use the canonical resource.');buffers[alias]=buffers[target];}
  if(!artifact.ast||!artifact.kernel)throw new Error('The CPU oracle needs an in-memory compiled AST, not a serialized WGSL artifact.');
  const grid=Array.isArray(workgroups)?[...workgroups]:[workgroups];while(grid.length<3)grid.push(1);
  if(grid.some(x=>!Number.isInteger(x)||x<0)||grid.length!==3)throw new RangeError('Invalid workgroup shape.');
  for(const p of artifact.metadata.scalars.filter(p=>p.origin==='constant'&&['cw_short','cw_ushort'].includes(p.sourceType))){const value=scalars[p.name]??p.defaultValue;if(!Number.isInteger(value)||value<(p.sourceType==='cw_short'?-32768:0)||value>(p.sourceType==='cw_short'?32767:65535))throw Error('Short scalar out of range.');}
  scalars={...Object.fromEntries(artifact.metadata.scalars.filter(p=>p.defaultValue!==undefined).map(p=>[p.name,p.defaultValue])),...scalars};
  const block=artifact.metadata.workgroupSize,baseEnv=new Map();
  for(const global of artifact.ast.constantGlobals)if(global.symbol){if(global.symbol.aggregate){const build=node=>{if(node.kind==='struct')return Object.fromEntries(node.fields.map(([name,value])=>[name,build(value)]));if(node.kind==='array')return node.items.map(build);const value=scalars[node.name]??0;if(!Number.isFinite(value))throw Error('Invalid constant struct component '+node.name);return convert(value,node.type);};baseEnv.set(global.symbol,{value:build(global.symbol.aggregate)});continue;}if(global.symbol.elements){const values=global.symbol.elements.map(name=>{const meta=artifact.metadata.scalars.find(s=>s.name===name),value=Object.hasOwn(scalars,name)?scalars[name]:meta.defaultValue;if(!Number.isFinite(value))throw new Error('Invalid constant array value '+name);return convert(value,global.type);});baseEnv.set(global.symbol,{value:values});continue;}const meta=artifact.metadata.scalars.find(s=>s.name===global.symbol.name),value=Object.hasOwn(scalars,meta.name)?scalars[meta.name]:meta.defaultValue;if(!Number.isFinite(value))throw new Error('Invalid constant global '+meta.name);baseEnv.set(global.symbol,{value:convert(value,global.type)});}
  for(const p of artifact.kernel.params){
    if(p.symbol.components){const element=vectorElement(p.type),value=p.symbol.components.map(name=>{const v=scalars[name];if(!Number.isFinite(v)||element==='f32'&&!Number.isFinite(Math.fround(v))||element==='u32'&&(!Number.isInteger(v)||v<0||v>0xffffffff)||element==='i32'&&(!Number.isInteger(v)||v<-2147483648||v>2147483647))throw Error('Invalid vector component '+name);return convert(v,element);});baseEnv.set(p.symbol,{value});continue;}
    if(p.type==='cw_size64'){const v=scalars[p.name];if(!Number.isInteger(v)||v<0||v>0xffffffff)throw Error('size_t launch value must fit u32.');baseEnv.set(p.symbol,{value:BigInt(v)});continue;}
    if(p.type==='cw_extent'){const value={};for(const field of ['width','height','depth']){const v=scalars[p.name+'.'+field];if(!Number.isInteger(v)||v<0||v>0xffffffff)throw Error('Invalid cudaExtent component '+field);value[field]=BigInt(v);}baseEnv.set(p.symbol,{value});continue;}
    if(p.pointer){if(p.type==='cw_uchar'&&!(buffers[p.name] instanceof Uint8Array))throw Error('Byte CPU buffers require Uint8Array.');if(!ArrayBuffer.isView(buffers[p.name]))throw new Error(`Missing CPU buffer ${p.name}.`);baseEnv.set(p.symbol,{value:new BufferView(buffers[p.name],p.type)});}
    else{if(['cw_short','cw_ushort'].includes(p.type)&&(!Number.isInteger(scalars[p.name])||scalars[p.name]<(p.type==='cw_short'?-32768:0)||scalars[p.name]>(p.type==='cw_short'?32767:65535)))throw Error('Short launch value out of range.');if(p.type==='bool'&&![true,false,0,1].includes(scalars[p.name]))throw new Error(`Invalid Boolean scalar ${p.name}.`);if(p.type==='cw_uchar4'&&(!Number.isInteger(scalars[p.name])||scalars[p.name]<0||scalars[p.name]>0xffffffff))throw new Error(`Invalid packed scalar ${p.name}.`);if(!Number.isFinite(scalars[p.name])&&!(p.type==='bool'&&typeof scalars[p.name]==='boolean'))throw new Error(`Missing/invalid scalar ${p.name}.`);baseEnv.set(p.symbol,{value:convert(scalars[p.name],p.type)});}
  }
  const shared=(artifact.ast.sharedGlobals||[]).filter(n=>n.symbol);for(const fn of artifact.ast.functions)walk(fn.body,n=>{if(n.kind==='decl'&&n.shared&&n.symbol&&!shared.some(d=>d.symbol===n.symbol))shared.push(n);});
  let groups=0,barriers=0;
  for(let z=0;z<grid[2];z++)for(let y=0;y<grid[1];y++)for(let x=0;x<grid[0];x++){
    const groupEnv=new Map(baseEnv);shared.forEach(n=>groupEnv.set(n.symbol,{value:zero(n.resolvedType)}));
    const lanes=[];
    for(let tz=0;tz<block[2];tz++)for(let ty=0;ty<block[1];ty++)for(let tx=0;tx<block[0];tx++){
      const laneEnv=new Map(groupEnv);for(const p of artifact.kernel.params)laneEnv.set(p.symbol,{...groupEnv.get(p.symbol)});
      const context=new Context(artifact,laneEnv,{threadIdx:[tx,ty,tz],blockIdx:[x,y,z],blockDim:block,gridDim:grid},instructionBudget);lanes.push(context.statement(artifact.kernel.body));
    }
    while(true){
      const states=lanes.map(lane=>lane.next());if(states.every(s=>s.done))break;
      if(states.some(s=>s.done)||states.some(s=>s.value!==states[0].value))throw new Error('Nonuniform __syncthreads: lanes exited or reached different barrier sites.');
      barriers++;
    }
    groups++;
  }
  return {mode:'CPU typed-AST oracle; NOT WGSL/GPU execution',groups,barriers};
}
