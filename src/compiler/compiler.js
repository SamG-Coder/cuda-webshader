import {FLOAT64_WGSL} from './float64.js';
import {inferTextureTypes,textureShape} from './texture-types.js';
import {integerExpression} from './integer-expression.js';
import {parse, CompileError,builtinType} from './parser.js';
import {uniformBlockGuards} from './full-workgroups.js';
export {CompileError, parse};
export const COMPILER_VERSION = '0.1.0';
export const isArray = t => !!t && typeof t === 'object' && t.kind === 'array';
export const arrayOf = (element, length = null) => ({kind: 'array', element, length});
export const typeName = t => isArray(t) ? `array<${typeName(t.element)}${t.length == null ? '' : `, ${t.length}`}>` : t;
export const vectorLength = t => typeof t === 'string' && /^vec[234]</.test(t) ? Number(t[3]) : 0;
export const vectorElement = t => vectorLength(t)?t.slice(5,-1):t;
export const typeStride = t => isArray(t) ? typeStride(t.element) * t.length : vectorLength(t) === 3 ? 16 : (vectorLength(t) || 1) * 4;
const numeric = t => ['f32', 'i32', 'u32','cw_uchar','cw_short','cw_ushort'].includes(t);
const uniformScalar=t=>['cw_short','cw_ushort'].includes(t)?{type:t==='cw_short'?'i32':'u32',sourceType:t}:{type:t};
const narrow = t => ['cw_uchar','cw_short','cw_ushort'].includes(t);
const indent = lines => lines.map(l => `  ${l}`);
const rootName = n => n?.kind === 'id' ? n.name : ['index', 'member'].includes(n?.kind) ? rootName(n.base) : null;
// Separate a named pointer root from left-associated element offsets.
const pointerParts=n=>{
  if(n?.kind==='id')return {base:n,offset:null};
  if(n?.kind==='unary'&&n.op==='&'&&n.value.kind==='index'&&n.value.base.kind==='id')return {base:n.value.base,offset:n.value.index};
  if(n?.kind==='binary'&&['+','-'].includes(n.op)){const p=pointerParts(n.left);if(p)return {base:p.base,offset:p.offset?{...n,left:p.offset}:n.op==='+'?n.right:{kind:'unary',op:'-',value:n.right,token:n.token}};}
  return null;
};
const shiftedPointers=fn=>{const names=new Set();walk(fn.body,n=>{if(n.kind==='assign'&&['=','+=','-='].includes(n.op)&&n.left.kind==='id')names.add(n.left.name);});return names;};
export function walk(node, visit) {
  if (!node || typeof node !== 'object') return;
  if (node.kind) visit(node);
  for (const [key, val] of Object.entries(node)) {
    if (['token', 'symbol', 'type', 'source', 'resolved','pointerBaseSymbol'].includes(key)) continue;
    if (Array.isArray(val)) val.forEach(n => walk(n, visit));
    else if (val && typeof val === 'object') walk(val, visit);
  }
}
const cudaValueSize=type=>['cw_short','cw_ushort'].includes(type)?2:['cw_uchar','bool'].includes(type)?1:type==='cw_uchar4'?4:type==='cw_extent'?24:['f32','i32','u32'].includes(type)?4:vectorLength(type)?vectorLength(type)*4:null;
function constantValue(n) {
  if(n.kind==='sizeof'){const size=cudaValueSize(n.target);if(size===null)throw new CompileError('sizeof requires a supported built-in value type.',n.token);return size;}
  if (n.kind === 'literal') return Number(n.value.replace(/^0[xX]/.test(n.value) ? /[uU]$/ : /[fFuU]$/, ''));
  if (n.kind === 'unary' && ['+', '-'].includes(n.op)) return (n.op === '-' ? -1 : 1) * constantValue(n.value);
  if (n.kind === 'binary') {
    const a = constantValue(n.left), b = constantValue(n.right);
    const ops = {'+': () => a + b, '-': () => a - b, '*': () => a * b, '/': () => Math.trunc(a / b), '%': () => a % b, '<<': () => a << b, '>>': () => a >> b};
    if (ops[n.op]) return ops[n.op]();
  }
  throw new CompileError('Array dimensions must be positive compile-time integer expressions.', n.token);
}
function analyse(functions, params) {
  const atomic = new Set(),atomicBindings=new Set(), reads = new Set(), writes = new Set(), bufferNames = new Set(params.filter(p => p.pointer).map(p => p.name));
  let aliases=new Map();const resolve=name=>aliases.has(name)?aliases.get(name):name;
  const scan = (n, mode = 'read') => {
    if (!n || !n.kind) return;
    if(n.kind==='block'||n.kind==='for'){const saved=aliases;aliases=new Map(aliases);if(n.kind==='block')n.body.forEach(s=>scan(s));else{scan(n.init);scan(n.condition);scan(n.body);scan(n.step);}aliases=saved;return;}
    if(['if','while','do'].includes(n.kind)){scan(n.condition);const saved=aliases;aliases=new Map(saved);scan(n.kind==='if'?n.yes:n.body);aliases=new Map(saved);if(n.kind==='if')scan(n.no);aliases=saved;return;}
    if(n.kind==='decl'){scan(n.init);const base=pointerParts(n.init)?.base;aliases.set(n.name,n.pointer&&base?.kind==='id'?resolve(base.name):null);return;}
    if (n.kind === 'assign') { scan(n.left, n.op === '=' ? 'write' : 'both'); scan(n.right); return; }
    if (n.kind === 'unary' && ['++', '--'].includes(n.op)) { scan(n.value, 'both'); return; }
    if (n.kind === 'call' && n.callee.kind === 'id' && ['atomicAdd', 'atomicMin', 'atomicMax', 'atomicExch','atomicCAS'].includes(n.callee.name)) {
      const target = n.args[0];
      if (target?.kind === 'unary' && target.op === '&') { const name = rootName(target.value); if (name){atomic.add(resolve(name)??name);if(bufferNames.has(resolve(name)))atomicBindings.add(resolve(name));} scan(target.value, 'both'); }
      n.args.slice(1).forEach(a => scan(a)); return;
    }
    if (n.kind === 'index') {
      const name = resolve(rootName(n)); if (bufferNames.has(name)) { if (mode !== 'write') reads.add(name); if (mode !== 'read') writes.add(name); }
      scan(n.index); if (n.base.kind === 'index') scan(n.base, mode); return;
    }
    if (n.kind === 'member') {const name=resolve(rootName(n.base));if(mode!=='read'&&n.base.kind==='index'&&params.some(p=>p.pointer&&p.name===name&&p.type==='cw_uchar4')){atomic.add(name);atomicBindings.add(name);reads.add(name);}scan(n.base, mode); return; }
    for (const [key, val] of Object.entries(n)) {
      if (['token', 'source'].includes(key)) continue;
      if (Array.isArray(val)) val.forEach(x => scan(x)); else if (val?.kind) scan(val);
    }
  };
  functions.forEach(f => {aliases=new Map(f.params.filter(p=>!p.pointer).map(p=>[p.name,null]));scan(f.body);});
  return {atomic,atomicBindings, reads, writes, storageBarrier: [...writes].some(x => reads.has(x))};
}
class Emitter {
  constructor(ast, kernel, options, templates,bufferUsage) {
    this.ast = ast; this.kernel = kernel; this.options = options; this.scopes = [new Map()]; this.temp = 0; this.loopDepth = 0; this.integerIntrinsics=new Set();
    this.structs=new Map((ast.structs||[]).map(s=>[s.type,s]));for(const s of this.structs.values())for(const field of s.fields){let type=field.type;for(const dim of [...field.dimensions].reverse()){const length=constantValue(dim);if(!Number.isSafeInteger(length)||length<1||length>256)this.fail('Struct field array dimensions must be 1..256.',field);type=arrayOf(type,length);}field.resolvedType=type;}
    inferTextureTypes(ast.functions,kernel,walk,(message,node)=>this.fail(message,node));
    this.overloads=new Map();for(const f of ast.functions)if(f.overloadName){const list=this.overloads.get(f.overloadName)||[];list.push(f);this.overloads.set(f.overloadName,list);}
    this.functions = new Map(); this.shared = [];this.pointerConstraints=[]; this.templates=templates;this.helperCalls=new Map();this.globalSymbols=new Map();this.constantScalars=[];
    for (const f of ast.functions) {
      if (this.functions.has(f.name)) this.fail(`Duplicate function '${f.name}'.`, f);
      this.functions.set(f.name, f);
    }
    this.helpers = ast.functions.filter(f => f.qualifier === '__device__'&&!f.params.some(p=>p.pointer));
    this.pointerHelpers=new Map();this.referenceHelpers=new Map();this.bufferSymbols=new Map();
    this.usage = analyse([kernel], kernel.params);this.usage.atomic=this.usage.atomicBindings;
    for(const kind of ['reads','writes','atomic'])for(const name of bufferUsage?.[kind]||[])this.usage[kind].add(name);
    for(const p of kernel.params)if(p.pointer&&p.type==='cw_uchar'&&this.usage.writes.has(p.name))this.usage.atomic.add(p.name);
    this.usage.storageBarrier=[...this.usage.writes].some(n=>this.usage.reads.has(n));
    this.initialBufferUsage=Object.fromEntries(['reads','writes','atomic'].map(k=>[k,new Set(this.usage[k])]));
    this.workgroupSize = [...(options.workgroupSize || [128, 1, 1])];
    while (this.workgroupSize.length < 3) this.workgroupSize.push(1);
    if (this.workgroupSize.length !== 3 || this.workgroupSize.some(v => !Number.isSafeInteger(v) || v < 1) || this.workgroupSize.reduce((a, b) => a * b, 1) > 1024) this.fail('Workgroup dimensions must be positive integers with at most 1024 total invocations.', kernel);
    if(kernel.launchThreads&&this.workgroupSize.reduce((a,b)=>a*b,1)>kernel.launchThreads)this.fail(`Launch exceeds __launch_bounds__(${kernel.launchThreads}).`,kernel);
    this.currentFunction = kernel;
    this.dynamicSharedBytes=options.sharedMemoryBytes??0;this.dynamicSharedUsed=false;
    if(!Number.isSafeInteger(this.dynamicSharedBytes)||this.dynamicSharedBytes<0||this.dynamicSharedBytes>65536)this.fail('sharedMemoryBytes must be an integer in [0,65536].',kernel);
  }
  fail(message, n) { throw new CompileError(message, n?.token, this.ast.source); }
  lookup(name, n) {
    for (let i = this.scopes.length - 1; i >= 0; --i) { const s = this.scopes[i].get(name); if (s) return s; }
    const global=this.ast.constantGlobals.find(g=>g.name===name);
    if(global){
      if(this.structs.has(global.type)||vectorLength(global.type)){
        if(!this.globalSymbols.has(name)){if(global.init||this.structs.has(global.type)&&global.dimensions.length)this.fail('Constant aggregates require zero initialization; struct arrays are unsupported.',global);let aggregateType=global.type;if(global.dimensions.length){const length=constantValue(global.dimensions[0]);if(!Number.isInteger(length)||length<1||length>256)this.fail('Constant vector arrays require 1..256 elements.',global);aggregateType=arrayOf(global.type,length);}const leaves=[];
          const build=(type,path)=>{if(this.structs.has(type)){const fields=this.structs.get(type).fields.map(f=>[f.name,build(f.resolvedType,path+'.'+f.name)]);return {code:`${type}(${fields.map(([,v])=>v.code).join(', ')})`,shape:{kind:'struct',fields:fields.map(([name,v])=>[name,v.shape])}};}if(isArray(type)||vectorLength(type)){const count=isArray(type)?type.length:vectorLength(type),element=isArray(type)?type.element:vectorElement(type),items=Array.from({length:count},(_,i)=>build(element,path+(isArray(type)?'['+i+']':'.'+'xyzw'[i])));return {code:`${typeName(type)}(${items.map(v=>v.code).join(', ')})`,shape:{kind:'array',items:items.map(v=>v.shape)}};}if(!numeric(type)||leaves.length>=(vectorLength(global.type)?1024:256))this.fail('Constant aggregates exceed the supported component limit.',global);const field='cw_struct_constant_'+this.ast.constantGlobals.indexOf(global)+'_'+leaves.length;leaves.push({name:path,...uniformScalar(type),origin:'constant',field,defaultValue:0});return {code:'cw_params.'+field,shape:{kind:'scalar',name:path,type}};};
          const value=build(aggregateType,'constant.'+name),symbol={name:'constant.'+name,type:aggregateType,code:value.code,constant:true,atomic:false,kind:'constant-global',aggregate:value.shape};this.constantScalars.push(...leaves);this.globalSymbols.set(name,symbol);global.symbol=symbol;
        }return this.globalSymbols.get(name);
      }
      if(!numeric(global.type))this.fail('Referenced constant globals require float, int or unsigned int scalar values.',n);
      if(!this.globalSymbols.has(name)){
        if(global.dimensions?.length){
          const length=constantValue(global.dimensions[0]);if(!Number.isInteger(length)||length<1||length>256)this.fail('Constant arrays require 1..256 scalar elements.',global);
          if(global.init&&(global.init.kind!=='initializer'||global.init.items.length>length))this.fail('Constant array initializers require a scalar list no longer than the array.',global);
          const fields=[],values=[];
          for(let i=0;i<length;i++){
            const init=global.init?.items[i];let value=0;
            if(init){const literal=init.kind==='unary'&&['+','-'].includes(init.op)?init.value:init;if(literal.kind!=='literal')this.fail('Constant array initializers must be numeric literals with an optional sign.',global);value=constantValue(init);}
            if(['cw_short','cw_ushort'].includes(global.type)&&(!Number.isInteger(value)||value<(global.type==='cw_short'?-32768:0)||value>(global.type==='cw_short'?32767:65535)))this.fail('Short constant initializer is outside its 16-bit range.',global);
            if(!Number.isFinite(value)||(global.type==='f32'&&!Number.isFinite(Math.fround(value)))||(global.type==='u32'&&(!Number.isInteger(value)||value<0||value>4294967295))||(global.type==='i32'&&(!Number.isInteger(value)||value<-2147483648||value>2147483647)))this.fail('Constant array initializer is outside its scalar range.',global);
            const field=`cw_array_${this.ast.constantGlobals.indexOf(global)}_${i}`,scalarName=`constant.${name}[${i}]`;
            fields.push('cw_params.'+field);values.push(scalarName);this.constantScalars.push({name:scalarName,...uniformScalar(global.type),origin:'constant',field,defaultValue:global.type==='f32'?Math.fround(value):value});
          }
          const symbol={name:'constant.'+name,type:arrayOf(global.type,length),code:`array<${global.type}, ${length}>(${fields.join(', ')})`,constant:true,atomic:false,kind:'constant-global',elements:values};this.globalSymbols.set(name,symbol);global.symbol=symbol;return symbol;
        }
        let value=0;
        if(global.init){const literal=global.init.kind==='unary'&&['+','-'].includes(global.init.op)?global.init.value:global.init;if(literal.kind!=='literal')this.fail('Constant global initializers must be numeric literals with an optional sign.',global);value=constantValue(global.init);}
        if(['cw_short','cw_ushort'].includes(global.type)&&(!Number.isInteger(value)||value<(global.type==='cw_short'?-32768:0)||value>(global.type==='cw_short'?32767:65535)))this.fail('Short constant initializer is outside its 16-bit range.',global);
            if(!Number.isFinite(value)||(global.type==='f32'&&!Number.isFinite(Math.fround(value)))||(global.type==='u32'&&(!Number.isInteger(value)||value<0||value>4294967295))||(global.type==='i32'&&(!Number.isInteger(value)||value<-2147483648||value>2147483647)))this.fail('Constant global initializer is outside its supported scalar range.',global);
        const scalarName='constant.'+name,symbol={name:scalarName,type:global.type,code:'cw_params.c_'+name,constant:true,atomic:false,kind:'constant-global'};
        this.globalSymbols.set(name,symbol);global.symbol=symbol;
        this.constantScalars.push({name:scalarName,...uniformScalar(global.type),origin:'constant',field:'c_'+name,defaultValue:global.type==='f32'?Math.fround(value):value});
      }
      return this.globalSymbols.get(name);
    }
    this.fail(`Unknown identifier '${name}'.`, n);
  }
  add(name, symbol, n, global = false) { const scope = global ? this.scopes[0] : this.scopes.at(-1); if (scope.has(name)) this.fail(`Duplicate identifier '${name}'.`, n); scope.set(name, symbol); return symbol; }
  common(a, b, n) {
    if (typeName(a) === typeName(b) && !isArray(a) && a !== 'void') return a;
    if((a==='cw_f64'||b==='cw_f64')&&[a,b].every(t=>t==='cw_f64'||numeric(t)||t==='bool'))return 'cw_f64';
    // C++ promotes a bool to int before the usual scalar arithmetic conversions.
    if(a==='bool'&&numeric(b))a='i32';
    if(b==='bool'&&numeric(a))b='i32';
    if (numeric(a) && numeric(b)) return a === 'f32' || b === 'f32' ? 'f32' : a === 'u32' || b === 'u32' ? 'u32' : 'i32';
    this.fail(`Incompatible operand types: ${typeName(a)} and ${typeName(b)}. Use explicit scalar/vector components.`, n);
  }
  convert(code, from, to, n) {
    if(to==='bool'&&isArray(from)&&from.length===null&&n){return 'true';} // All runtime storage bindings are required and non-null.
    if (typeName(from) === typeName(to) && !isArray(to)) return code;
    if(to==='cw_f64'&&(numeric(from)||from==='bool')){this.float64Used=true;return from==='bool'?`cw_d_from_u32(select(0u,1u,${code}))`:from==='f32'?`cw_d_from_f32(${code})`:['i32','cw_short'].includes(from)?`cw_d_from_i32(i32(${code}))`:`cw_d_from_u32(u32(${code}))`;}
    if(from==='cw_f64'&&to==='f32'){this.float64Used=true;return `cw_d_to_f32(${code})`;}
    if(from==='cw_f64'&&to==='bool'){this.float64Used=true;return `!cw_d_zero(${code})`;}
    if(from==='cw_size64'&&to==='f32'){this.float64Used=true;return `cw_d_u64_to_f32(${code})`;}
    if(to==='cw_short'||to==='cw_ushort'){if(from==='bool')return to==='cw_short'?`select(0i,1i,${code})`:`select(0u,1u,${code})`;if(numeric(from)){const word=`i32(${code})`;return to==='cw_short'?`(bitcast<i32>((bitcast<u32>(${word}) & 65535u) << 16u) >> 16u)`:`(u32(${word}) & 65535u)`;}}
    if(from==='cw_short'||from==='cw_ushort'){if(to==='bool')return `(${code} != ${from==='cw_short'?'0i':'0u'})`;if(numeric(to)&&to!=='cw_uchar')return `${to}(${code})`;}
    if(to==='cw_uchar'){if(from==='bool')return `select(0u,1u,${code})`;if(numeric(from))return `(${from==='f32'?`u32(i32(${code}))`:`u32(${code})`} & 255u)`;}
    if(from==='cw_uchar'){if(to==='bool')return `(${code} != 0u)`;if(numeric(to))return `${to}(${code})`;}
    if(from==='cw_size64'&&['i32','u32'].includes(to))return `${to}((${code}).x)`;
    if (numeric(from) && numeric(to)) return `${to}(${code})`;
    if (to === 'bool' && numeric(from)) return `(${code} != ${from === 'f32' ? '0.0f' : from === 'u32' ? '0u' : '0i'})`;
    if (from === 'bool' && numeric(to)) return `select(${to}(0), ${to}(1), ${code})`;
    this.fail(`Cannot convert ${typeName(from)} to ${typeName(to)}.`, n);
  }
  result(n, type, code, pre = [], extra = {}) { n.type = type; return {type, code, pre, ...extra}; }
  expr(n, raw = false, captureIndex = false) {
    if (!n) this.fail('Missing expression.', this.kernel);
    switch (n.kind) {
      case 'initializer': {
        const width=vectorLength(n.target),element=vectorElement(n.target);
        if(!width||n.items.length>width)this.fail('Vector initializers require at most one scalar per component.',n);
        const values=n.items.map(item=>this.expr(item));
        if(values.some((v,i)=>{if(v.type===element)return false;if(element==='f32'&&['i32','u32'].includes(v.type)){try{const value=constantValue(n.items[i]);return !Number.isInteger(value)||Math.fround(value)!==value;}catch{}}return true;}))this.fail('Vector initializer components must match the element type; use explicit casts for conversions.',n);
        const codes=values.map(v=>this.convert(v.code,v.type,element,n));while(codes.length<width)codes.push(`${element}(0)`);
        return this.result(n,n.target,`${n.target}(${codes.join(', ')})`,values.flatMap(v=>v.pre));
      }
      case 'sizeof': {const size=cudaValueSize(n.target);if(size===null)this.fail('sizeof requires a supported built-in value type.',n);this.extentUsed=true;n.numericValue=size;return this.result(n,'cw_size64',`vec2<u32>(${size}u, 0u)`);}
      case 'literal': {
        const isHex = /^0[xX]/.test(n.value), isFloat = !isHex && /[fF]$/.test(n.value), isUnsigned = /[uU]$/.test(n.value);
        if (!isFloat && /[.eE]/.test(n.value) && !/^0[xX]/.test(n.value)) {const value=Number(n.value);if(!Number.isFinite(value))this.fail('Invalid double literal.',n);const bits=new DataView(new ArrayBuffer(8));bits.setFloat64(0,value,true);this.float64Used=true;n.numericValue=value;return this.result(n,'cw_f64',`vec2<u32>(${bits.getUint32(0,true)}u, ${bits.getUint32(4,true)}u)`);}
        let value = Number(n.value.replace(isHex ? /[uU]$/ : /[fFuU]$/, ''));
        if (!Number.isFinite(value)) this.fail('Invalid numeric literal.', n);
        let type = isFloat ? 'f32' : isUnsigned || (/^0[xX]/.test(n.value) && value > 2147483647) ? 'u32' : 'i32';
        if (!isFloat && (value < (type === 'u32' ? 0 : -2147483648) || value > (type === 'u32' ? 4294967295 : 2147483647))) this.fail('Integer literal is outside the supported 32-bit range.', n);
        if (isFloat && !Number.isFinite(Math.fround(value))) this.fail('Floating literal overflows f32.', n);
        n.numericValue = value;
        if(type==='i32'&&value===-2147483648)return this.result(n,type,'i32(2147483648u)');
        return this.result(n, type, `${isFloat && Number.isInteger(value) ? value + '.0' : value}${type === 'f32' ? 'f' : type === 'u32' ? 'u' : 'i'}`);
      }
      case 'id': {
        if (['true', 'false'].includes(n.name)) return this.result(n, 'bool', n.name);
        const s = this.lookup(n.name, n); n.symbol = s;
        if(s.kind==='thread-block')this.fail('A thread_block handle can only be used for block synchronization.',n);
        const atomic=s.atomic&&!isArray(s.type);return this.result(n, s.type, atomic&&!raw?`atomicLoad(&${s.code})`:s.code, [], {rootSymbol: s, atomicRoot: s.atomic,atomic});
      }
      case 'index': {
        const base = this.expr(n.base, true), index = this.expr(n.index);
        if(base.rootSymbol?.kind==='pointer-array'){
          const slots=base.rootSymbol,root=slots.pointerRoot;if(!root)this.fail('Pointer array must be assigned a shared-array address before use.',n);
          if(!['i32','u32'].includes(index.type))this.fail('Pointer array indices must be 32-bit integers.',n);
          const temp='cw_shared_pointer_'+this.temp++;n.pointerArrayElement=true;n.pointerBaseSymbol=root;
          const pointerCode=temp,symbol={...root,kind:'buffer-alias',constant:slots.constant||root.constant,sharedPointer:root.code,offsetCode:pointerCode};
          return this.result(n,arrayOf(slots.elementType),root.code,[...base.pre,...index.pre,`let ${temp}: i32 = ${slots.code}[${index.code}];`],{rootSymbol:symbol,pointerCode});
        }
        if(n.dereference&&!['buffer','buffer-alias'].includes(base.rootSymbol?.kind))this.fail('Dereference requires a storage-buffer pointer.',n);
        if (!isArray(base.type) || !['i32', 'u32'].includes(index.type)) this.fail('Indexing requires an array and a 32-bit integer index.', n);
        const offset=base.rootSymbol?.offsetCode;let indexCode=offset?`(${offset} + ${this.convert(index.code,index.type,'i32',n)})`:index.code;
        const capturePre=[];if(captureIndex&&['local','shared'].includes(base.rootSymbol?.kind)){const temp='cw_argument_index_'+this.temp++;capturePre.push(`let ${temp} = ${indexCode};`);indexCode=temp;}
        let code = `${base.code}[${indexCode}]`;const type = base.type.element;
        if(type==='cw_uchar'&&base.rootSymbol?.rootBufferName){const temp='cw_byte_index_'+this.temp++,word=`${base.code}[${temp} >> 2u]`,shift=`((${temp} & 3u) * 8u)`,read=base.atomicRoot?`atomicLoad(&${word})`:word;return this.result(n,type,`((${read} >> ${shift}) & 255u)`,[...base.pre,...index.pre,`let ${temp} = u32(${indexCode});`],{rootSymbol:base.rootSymbol,...(raw&&base.atomicRoot?{packedBase:word,packedShiftCode:shift,packedAtomic:true}:{})});}
        const atomic = base.atomicRoot && !isArray(type);
        const addressPre=[];if(atomic&&raw&&type==='cw_uchar4'){const temp='cw_pixel_index_'+this.temp++;addressPre.push(`let ${temp} = ${indexCode};`);code=`${base.code}[${temp}]`;}
        return this.result(n, type, atomic && !raw ? `atomicLoad(&${code})` : code, [...base.pre, ...index.pre,...capturePre,...addressPre], {rootSymbol: base.rootSymbol, atomicRoot: base.atomicRoot, atomic});
      }
      case 'member': {
        if (n.base.kind === 'id' && ['threadIdx', 'blockIdx', 'blockDim', 'gridDim'].includes(n.base.name)) {
          if (!['x', 'y', 'z'].includes(n.member)) this.fail('CUDA dimensions have x, y and z components only.', n);
          const code = {threadIdx: 'cw_thread', blockIdx: 'cw_block', blockDim: 'cw_block_size', gridDim: 'cw_grid'}[n.base.name];
          return this.result(n, 'u32', `${code}.${n.member}`);
        }
        const base = this.expr(n.base, raw), size = vectorLength(base.type);
        if(base.type==='cw_extent'){if(!['width','height','depth'].includes(n.member))this.fail('cudaExtent has width, height and depth fields.',n);return this.result(n,'cw_size64',`${base.code}.${n.member}`,base.pre,{rootSymbol:base.rootSymbol});}
        if(base.type==='cw_uchar4'){if(n.member.length!==1||!'xyzw'.includes(n.member))this.fail('uchar4 has x, y, z and w byte components.',n);const shift='xyzw'.indexOf(n.member)*8,read=base.atomic&&raw?`atomicLoad(&${base.code})`:base.code;return this.result(n,'cw_uchar',`((${read} >> ${shift}u) & 255u)`,base.pre,{rootSymbol:base.rootSymbol,packedBase:base.code,packedShift:shift,packedAtomic:!!base.atomic});}
        if(this.structs.has(base.type)){const field=this.structs.get(base.type).fields.find(f=>f.name===n.member);if(!field)this.fail('Unknown struct field '+n.member,n);return this.result(n,field.resolvedType,`${base.code}.cw_field_${n.member}`,base.pre,{rootSymbol:base.rootSymbol});}
        if (!size || n.member.length !== 1 || 'xyzw'.indexOf(n.member) < 0 || 'xyzw'.indexOf(n.member) >= size) this.fail('Only valid single vector components (.x/.y/.z/.w) are supported.', n);
        return this.result(n, vectorElement(base.type), `${base.code}.${n.member}`, base.pre, {rootSymbol: base.rootSymbol});
      }
      case 'cast': {
        if(n.target==='cw_uchar'&&n.value.kind==='binary'&&n.value.op==='*'){
          const doubleInteger=a=>a.kind==='literal'&&/[.eE]/.test(a.value)&&!/^0[xX]/.test(a.value)&&!/[fFuU]$/.test(a.value)&&Number.isInteger(Number(a.value))&&Number(a.value)>=1&&Number(a.value)<=65535;
          const literal=doubleInteger(n.value.right)?n.value.right:doubleInteger(n.value.left)?n.value.left:null;
          if(literal){const operand=literal===n.value.right?n.value.left:n.value.right,value=this.expr(operand);if(value.type!=='f32')this.fail('Exact byte scaling requires a float32 operand.',n);this.exactByteScaleUsed=true;n.byteScale=Number(literal.value);n.byteScaleValue=operand;return this.result(n,'cw_uchar',`cw_exact_byte_scale(${value.code}, ${n.byteScale}u)`,value.pre);}
        }
        const literal=n.value.kind==='unary'&&['+','-'].includes(n.value.op)?n.value.value:n.value;if(n.target==='f32'&&literal.kind==='literal'&&/[.eE]/.test(literal.value)&&!/^0[xX]/.test(literal.value)&&!/[fFuU]$/.test(literal.value)){const rounded=Math.fround(Number(literal.value));if(!Number.isFinite(rounded))this.fail('Explicit float literal conversion overflows f32.',literal);literal.value=String(rounded)+'f';}const value = this.expr(n.value); return this.result(n, n.target, this.convert(value.code, value.type, n.target, n), value.pre); }
      case 'unary': {
        if (['++', '--'].includes(n.op)){if(n.value.kind!=='id')this.fail('Expression increments require named local scalars or references.',n);const value=this.expr(n.value,true);if(!['local','reference'].includes(value.rootSymbol?.kind))this.fail('Expression increments require named local scalars or references.',n);const update=this.effect(n),tmp='cw_update_'+this.temp++,snapshot=`let ${tmp}: ${value.type} = ${value.code};`;return this.result(n,value.type,tmp,n.prefix?[...update,snapshot]:[snapshot,...update]);}
        if (n.op === '&' || n.op === '*') this.fail('Pointers are supported only as kernel buffer parameters and &buffer[index] atomic targets.', n);
        let value = this.expr(n.value);if(narrow(value.type))value={...value,type:'i32',code:`i32(${value.code})`};
        if (n.op === '!') return this.result(n, 'bool', `(!${this.convert(value.code, value.type, 'bool', n)})`, value.pre);
        if(value.type==='cw_f64'&&['+','-'].includes(n.op))return this.result(n,'cw_f64',n.op==='+'?value.code:`cw_d_neg(${value.code})`,value.pre);
        if (!numeric(value.type) || (n.op === '~' && value.type === 'f32')) this.fail('Invalid unary operator/type.', n);
        if (n.op === '-' && value.type === 'u32') return this.result(n, 'u32', `(0u - ${value.code})`, value.pre);
        return this.result(n, value.type, n.op === '+' ? value.code : `(${n.op}${value.code})`, value.pre);
      }
      case 'binary': {
        let a = this.expr(n.left), b = this.expr(n.right);if(narrow(a.type))a={...a,type:'i32',code:`i32(${a.code})`};if(narrow(b.type))b={...b,type:'i32',code:`i32(${b.code})`};
        if (['&&', '||'].includes(n.op)) {
          const ac = this.convert(a.code, a.type, 'bool', n), bc = this.convert(b.code, b.type, 'bool', n);
          if (!b.pre.length) return this.result(n, 'bool', `(${ac} ${n.op} ${bc})`, [...a.pre]);
          const tmp = `cw_tmp_${this.temp++}`;
          const pre = [...a.pre, `var ${tmp}: bool = ${ac};`, `if (${n.op === '&&' ? tmp : `!${tmp}`}) {`, ...indent([...b.pre, `${tmp} = ${bc};`]), '}'];
          return this.result(n, 'bool', tmp, pre);
        }
        if(a.type==='cw_size64'||b.type==='cw_size64'){
          if(!['==','!=','<','>','<=','>=','*','+','-'].includes(n.op)||![a.type,b.type].every(t=>['cw_size64','i32','u32','bool'].includes(t)))this.fail('Size values support integer comparisons and multiplication only.',n);
          const left='cw_size_left_'+this.temp++,right='cw_size_right_'+this.temp++,promote=(v,code)=>v.type==='cw_size64'?code:v.type==='bool'?`vec2<u32>(select(0u, 1u, ${code}), 0u)`:v.type==='i32'?`vec2<u32>(u32(${code}), select(0u, 4294967295u, ${code} < 0i))`:`vec2<u32>(u32(${code}), 0u)`;
          const pre=[...a.pre,`let ${left} = ${a.code};`,...b.pre,`let ${right} = ${b.code};`],ac=promote(a,left),bc=promote(b,right);this.extentUsed=true;n.operandType='cw_size64';
          if(['+','-'].includes(n.op)){this.float64Used=true;return this.result(n,'cw_size64',`${n.op==='+'?'cw_d_uadd':'cw_d_usub'}(${ac}, ${bc})`,pre);}
          if(n.op==='*'){this.sizeMultiplyUsed=true;return this.result(n,'cw_size64',`cw_size_multiply(${ac}, ${bc})`,pre);}
          const code=n.op==='=='?`all(${ac} == ${bc})`:n.op==='!='?`any(${ac} != ${bc})`:n.op==='<'?`cw_size_less(${ac}, ${bc})`:n.op==='>'?`cw_size_less(${bc}, ${ac})`:n.op==='<='?`!cw_size_less(${bc}, ${ac})`:`!cw_size_less(${ac}, ${bc})`;
          return this.result(n,'bool',code,pre);
        }
        if(a.type==='cw_uchar4'||b.type==='cw_uchar4')this.fail('uchar4 arithmetic requires explicit byte components.',n);
        if(vectorLength(a.type)||vectorLength(b.type)){const type=vectorLength(a.type)?a.type:b.type;const element=vectorElement(type),ops=element==='f32'?['+','-','*','/']:['+','-','*'];const compatible=v=>v.type===type||v.type===element||element==='f32'&&numeric(v.type);if(!ops.includes(n.op)||!compatible(a)||!compatible(b))this.fail('Vector arithmetic requires matching vector/scalar element types; integer vectors support +, -, * only.',n);const code=v=>v.type===type?v.code:`${type}(${this.convert(v.code,v.type,element,n)})`;n.operandType=type;return this.result(n,type,`(${code(a)} ${n.op} ${code(b)})`,[...a.pre,...b.pre]);}
        let common = ['<<', '>>'].includes(n.op) ? a.type : this.common(a.type, b.type, n);
        if (vectorLength(common)) this.fail('CUDA vector arithmetic requires explicit components; operator overloads are outside this subset.', n);
        if (['&', '|', '^', '<<', '>>', '%'].includes(n.op) && !['i32', 'u32'].includes(common)) this.fail('Bitwise, shift and remainder operators require integers.', n);
        if (common === 'bool' && !['==', '!='].includes(n.op)) this.fail('Boolean values support only logical/equality operators.', n);
        const ac = this.convert(a.code, a.type, common, n), bc = this.convert(b.code, b.type, ['<<', '>>'].includes(n.op) ? 'u32' : common, n);
        const out = ['==', '!=', '<', '>', '<=', '>='].includes(n.op) ? 'bool' : common;
        n.operandType = common;
        if(common==='cw_f64'){
          this.float64Used=true;const functions={'+':'cw_d_add','-':'cw_d_sub','*':'cw_d_mul','/':'cw_d_div','==':'cw_d_eq','<':'cw_d_lt'};
          const code=functions[n.op]?`${functions[n.op]}(${ac}, ${bc})`:n.op==='!='?`!cw_d_eq(${ac}, ${bc})`:n.op==='>'?`cw_d_lt(${bc}, ${ac})`:n.op==='<='?`cw_d_le(${ac}, ${bc})`:`cw_d_le(${bc}, ${ac})`;
          return this.result(n,out,code,[...a.pre,...b.pre]);
        }
        if(n.op==='/'&&common==='f32'){this.compensatedDivisionUsed=true;return this.result(n,out,`cw_divide_f32(${ac}, ${bc})`,[...a.pre,...b.pre]);}
        return this.result(n, out, `(${ac} ${n.op} ${bc})`, [...a.pre, ...b.pre]);
      }
      case 'conditional': {
        const cond = this.expr(n.condition), yes = this.expr(n.yes), no = this.expr(n.no), type = this.common(yes.type, no.type, n);
        const tmp = `cw_tmp_${this.temp++}`;
        // select() is eager. A branch is mandatory to preserve guarded loads, divisions and atomic side effects.
        const pre = [...cond.pre, `var ${tmp}: ${typeName(type)};`, `if (${this.convert(cond.code, cond.type, 'bool', n)}) {`, ...indent([...yes.pre, `${tmp} = ${this.convert(yes.code, yes.type, type, n)};`]), '} else {', ...indent([...no.pre, `${tmp} = ${this.convert(no.code, no.type, type, n)};`]), '}'];
        return this.result(n, type, tmp, pre);
      }
      case 'call': return this.call(n);
      case 'assign': this.fail('Assignments with a used return value are unsupported. Put assignments in their own statements.', n); break;
      default: this.fail(`Unsupported expression '${n.kind}'.`, n);
    }
  }
  surfaceGridCoordinate(node,axis,scale,seen=new Set()){
    // Restrict trap-mode stores to coordinates whose full range the runtime can
    // validate before submission. Never silently turn a CUDA trap into a drop.
    const literal=(n,value)=>n?.kind==='sizeof'?cudaValueSize(n.target)===value:n?.kind==='literal'&&Number(n.value.replace(/[uUlL]+$/,''))===value;
    if(scale===1&&node?.kind==='binary'&&node.op==='*'&&(literal(node.left,1)||literal(node.right,1)))return this.surfaceGridCoordinate(literal(node.left,1)?node.right:node.left,axis,1,seen);
    if(scale!==1)return node?.kind==='binary'&&node.op==='*'&&((literal(node.right,scale)&&this.surfaceGridCoordinate(node.left,axis,1,seen))||(literal(node.left,scale)&&this.surfaceGridCoordinate(node.right,axis,1,seen)));
    if(node?.kind==='id'){
      if(seen.has(node.name))return false;const declarations=[];let changed=false;
      walk(this.kernel.body,n=>{if(n.kind==='decl'&&n.name===node.name)declarations.push(n);if(n.kind==='call'&&!['surf1Dwrite','surf2Dwrite','surf3Dwrite','surf2DLayeredwrite'].includes(n.callee?.name)&&!(/^(?:make_(?:float|int|uint)[234]|float|int|uint|unsigned)$/.test(n.callee?.name||'')&&!this.ast.functions.some(f=>f.name===n.callee.name)))for(const arg of n.args)walk(arg,a=>{if(a.kind==='id'&&a.name===node.name)changed=true;});if((n.kind==='assign'&&n.left?.kind==='id'&&n.left.name===node.name)||(n.kind==='unary'&&['++','--'].includes(n.op)&&n.value?.name===node.name))changed=true;});
      if(changed||declarations.length!==1)return false;return this.surfaceGridCoordinate(declarations[0].init,axis,1,new Set([...seen,node.name]));
    }
    const member=(n,name)=>n?.kind==='member'&&n.base?.kind==='id'&&n.base.name===name&&n.member===axis;
    const product=n=>n?.kind==='binary'&&n.op==='*'&&((member(n.left,'blockIdx')&&member(n.right,'blockDim'))||(member(n.right,'blockIdx')&&member(n.left,'blockDim')));
    return node?.kind==='binary'&&node.op==='+'&&((product(node.left)&&member(node.right,'threadIdx'))||(product(node.right)&&member(node.left,'threadIdx')));
  }
  argument(n){
    if(n.kind==='unary'&&n.op==='&'&&n.value.kind==='id'){const value=this.expr(n.value),symbol=value.rootSymbol;if(!symbol||!['local','reference'].includes(symbol.kind)||symbol.constant||!numeric(value.type))this.fail('Local pointer arguments require a mutable named numeric scalar.',n);return this.result(n,arrayOf(value.type),value.code,value.pre,{rootSymbol:symbol,localPointer:true,pointerCode:symbol.kind==='reference'?symbol.pointerCode:'&'+value.code});}
    const address=n.kind==='unary'&&n.op==='&'&&n.value.kind==='index'?n.value:null;
    const parts=pointerParts(n),base=parts?.base;
    if(base?.kind==='id'&&!['true','false'].includes(base.name)){
      const symbol=this.lookup(base.name,base);
      if(symbol.kind==='thread-block'&&n.kind==='id'){n.symbol=symbol;return {type:'thread-block',code:'',pre:[],rootSymbol:symbol};}
      if(symbol.kind==='shared'&&isArray(symbol.type)&&!isArray(symbol.type.element)){
        if(symbol.atomic)this.fail('Shared helper pointers do not support atomic arrays.',n);
        const node=parts.offset,offset=node?this.expr(node):{type:'i32',code:'0i',pre:[]};
        if(!['i32','u32'].includes(offset.type))this.fail('Shared helper offsets must be 32-bit integers.',n);
        n.pointerBaseSymbol=symbol;n.pointerOffset=node;
        return this.result(n,symbol.type,symbol.code,offset.pre,{rootSymbol:{...symbol,sharedPointer:symbol.code},pointerCode:this.convert(offset.code,offset.type,'i32',n)});
      }
      if(['buffer','buffer-alias'].includes(symbol.kind)){
        const node=parts.offset,offset=node?this.expr(node):{type:'i32',code:'0i',pre:[]};
        if(!['i32','u32'].includes(offset.type))this.fail('Helper buffer offsets must be 32-bit integers.',n);
        n.pointerBaseSymbol=symbol;n.pointerOffset=node;
        const pointerCode=symbol.offsetCode?`(${symbol.offsetCode} + ${this.convert(offset.code,offset.type,'i32',n)})`:this.convert(offset.code,offset.type,'i32',n);
        return this.result(n,symbol.type,symbol.code,offset.pre,{rootSymbol:symbol,pointerCode});
      }
    }
    return this.expr(n,false,n.kind==='index');
  }
  bindPointerHelper(helper,args,n){
    const uses=analyse([helper],helper.params),roots=[];
    for(const [i,p]of helper.params.entries())if(p.pointer){
      const a=args[i],symbol=a?.rootSymbol,root=symbol?.rootBufferName;
      if(a?.localPointer){if(a.type.element!==p.type)this.fail('Local pointer type must exactly match the helper parameter.',n.args[i]);roots.push([i,'@local',false]);continue;}
      if(symbol?.sharedPointer){if(!isArray(a.type)||a.type.element!==p.type)this.fail('Shared pointer type must exactly match the helper parameter.',n.args[i]);if(symbol.constant&&!p.constant)this.fail('Cannot discard const through a shared helper pointer.',n.args[i]);roots.push([i,'@shared:'+symbol.sharedPointer,!!symbol.constant]);continue;}
      if(!root||!isArray(a.type)||a.type.element!==p.type)this.fail('Helper pointers require a same-type storage buffer or buffer offset.',n.args[i]);
      if(symbol.constant&&!p.constant)this.fail('Cannot discard const through a helper pointer argument.',n.args[i]);
      roots.push([i,root,!!symbol.constant]);
      if(uses.reads.has(p.name))this.usage.reads.add(root);
      if(uses.writes.has(p.name))this.usage.writes.add(root);
      if(uses.atomicBindings.has(p.name)||p.type==='cw_uchar'&&uses.writes.has(p.name)){this.usage.atomic.add(root);this.bufferSymbols.get(root).atomic=true;}
    }
    this.usage.storageBarrier=[...this.usage.writes].some(x=>this.usage.reads.has(x));
    const key=JSON.stringify([helper.name,roots]);
    if(!this.pointerHelpers.has(key)){
      if(this.pointerHelpers.size>=128)this.fail('At most 128 helper buffer specializations are supported.',n);
      const clone=structuredClone(helper);let name='cw_buffer_helper_'+this.pointerHelpers.size;while(this.functions.has(name))name+='_';
      clone.name=name;clone.pointerOrigin=helper.name;
      const localNames=new Set(roots.filter(([,root])=>root==='@local').map(([i])=>clone.params[i].name));
      const inspect=(node,parent)=>{if(!node||typeof node!=='object')return;if(node.kind==='decl'&&localNames.has(node.name))this.fail('Shadowed local pointer parameters are unsupported.',node);if(node.kind==='id'&&localNames.has(node.name)&&!(parent?.kind==='index'&&parent.base===node&&parent.index.kind==='literal'&&Number(parent.index.value.replace(/[uU]$/,''))===0))this.fail('Local pointers support only dereference or index zero; arithmetic and escapes are unsupported.',node);for(const [key,value]of Object.entries(node))if(!['token','type'].includes(key)){if(Array.isArray(value))value.forEach(v=>inspect(v,node));else if(value&&typeof value==='object')inspect(value,node);}};inspect(clone.body,null);
      walk(clone.body,node=>{if(node.kind==='index'&&node.base.kind==='id'&&localNames.has(node.base.name)){const name=node.base.name;delete node.base;delete node.index;delete node.dereference;node.kind='id';node.name=name;}});
      for(const [i,root,constant]of roots){if(root==='@local'){clone.params[i].pointer=false;clone.params[i].reference=true;clone.params[i].localPointer=true;}else{if(root.startsWith('@shared:'))clone.params[i].boundShared=root.slice(8);else clone.params[i].boundBuffer=root;clone.params[i].boundConstant=constant;}}
      this.pointerHelpers.set(key,clone);this.functions.set(name,clone);this.helpers.push(clone);this.ast.functions.push(clone);
    }
    return this.pointerHelpers.get(key);
  }
  bindReferenceHelper(helper,args,n){
    const spaces=helper.params.map((p,i)=>p.reference?(args[i].rootSymbol?.kind==='shared'||args[i].rootSymbol?.referenceSpace==='workgroup'?'workgroup':'function'):null);
    if(spaces.every((space,i)=>space===null||space===(helper.params[i].referenceSpace||'function')))return helper;
    const origin=helper.referenceOrigin||helper.name,key=JSON.stringify([origin,spaces]);
    if(!this.referenceHelpers.has(key)){
      if(this.referenceHelpers.size>=128)this.fail('At most 128 helper reference specializations are supported.',n);
      const base=this.functions.get(origin)||helper;walk(base.body,node=>{if(node.kind==='decl'&&node.shared)this.fail('Reference address-space specialization of helpers with shared declarations is unsupported.',node);});
      const clone=structuredClone(base);let name='cw_reference_helper_'+this.referenceHelpers.size;while(this.functions.has(name))name+='_';clone.name=name;clone.referenceOrigin=origin;
      spaces.forEach((space,i)=>{if(space)clone.params[i].referenceSpace=space;});this.referenceHelpers.set(key,clone);this.functions.set(name,clone);this.helpers.push(clone);this.ast.functions.push(clone);
    }
    return this.referenceHelpers.get(key);
  }
  call(n) {
    const groupSync=n.callee.kind==='id'&&n.callee.name==='cooperative_groups::sync';
    const memberSync=n.callee.kind==='member'&&n.callee.member==='sync';
    if(groupSync||memberSync){
      const group=groupSync?n.args[0]:n.callee.base;
      if(n.args.length!==(groupSync?1:0)||group?.kind!=='id'||this.lookup(group.name,group).kind!=='thread-block')this.fail('Block sync requires a local thread_block handle from this_thread_block().',n);
      n.callName='__syncthreads';return this.result(n,'void','workgroupBarrier()');
    }
    if (n.callee.kind !== 'id') this.fail('Only named functions are supported.', n);
    const name = n.callee.name; n.callName = name;
    if(name==='tex1D'){if(n.callee.templateArgument!=='float4'||n.args.length!==2||n.args[0].kind!=='id')this.fail('tex1D supports a bound float4 texture and one float coordinate.',n);const texture=this.lookup(n.args[0].name,n.args[0]),coordinate=this.expr(n.args[1]);if(texture.kind!=='texture'||texture.dimension!=='2d'||!numeric(coordinate.type))this.fail('tex1D requires a matching kernel texture parameter and float coordinate.',n);return this.result(n,'vec4<f32>',`textureSampleLevel(${texture.code}, ${texture.sampler}, vec2<f32>(${this.convert(coordinate.code,coordinate.type,'f32',n)}, 0.5f), 0.0f)`,coordinate.pre);}
    if(name==='tex2DLayered'){
      if(n.callee.templateArgument!=='float4'||n.args.length!==4||n.args[0]?.kind!=='id')this.fail('tex2DLayered requires a float4 texture, two coordinates and an integer layer.',n);
      const texture=this.lookup(n.args[0].name,n.args[0]),x=this.expr(n.args[1]),y=this.expr(n.args[2]),layer=this.expr(n.args[3]);
      if(texture.kind!=='texture'||texture.dimension!=='2d-array'||![x.type,y.type].every(numeric)||!['i32','u32'].includes(layer.type))this.fail('Layered sampling requires matching float4 layers and scalar coordinates.',n);
      return this.result(n,'vec4<f32>',`textureSampleLevel(${texture.code}, ${texture.sampler}, vec2<f32>(${this.convert(x.code,x.type,'f32',n)}, ${this.convert(y.code,y.type,'f32',n)}), i32(${layer.code}), 0.0f)`,[...x.pre,...y.pre,...layer.pre]);
    }
    if(name==='surf2DLayeredwrite'){
      if(![5,6].includes(n.args.length)||n.args[1]?.kind!=='id'||n.args.length===6&&(n.args[5]?.kind!=='id'||n.args[5].name!=='cudaBoundaryModeTrap'))this.fail('Layered stores require float4, global XY byte coordinates and trap mode.',n);
      const surface=this.lookup(n.args[1].name,n.args[1]),value=this.expr(n.args[0]),x=this.expr(n.args[2]),y=this.expr(n.args[3]),layer=this.expr(n.args[4]);
      if(surface.kind!=='surface'||surface.dimension!=='2d-array'||value.type!=='vec4<f32>'||![x.type,y.type].every(t=>['i32','u32','cw_size64'].includes(t))||!['i32','u32'].includes(layer.type)||!this.surfaceGridCoordinate(n.args[2],'x',16)||!this.surfaceGridCoordinate(n.args[3],'y',1))this.fail('Layered stores require float4 and checked global XY byte coordinates.',n);
      const selector=n.args[4].kind==='id'&&layer.rootSymbol?.kind==='uniform'?{scalar:layer.rootSymbol.name}:n.args[4].kind==='literal'?{value:n.args[4].numericValue}:null;
      if(!selector||surface.metadata.layer&&JSON.stringify(surface.metadata.layer)!==JSON.stringify(selector))this.fail('Layered stores require one uniform or literal layer per surface.',n);
      surface.metadata.layer=selector;surface.metadata.format='rgba32float';surface.metadata.coordinates='global-xy-layer';
      return this.result(n,'void',`textureStore(${surface.code}, vec2<i32>(${this.convert(x.code,x.type,'i32',n)} / 16i, ${this.convert(y.code,y.type,'i32',n)}), i32(${layer.code}), ${value.code})`,[...value.pre,...x.pre,...y.pre,...layer.pre]);
    }
    if(name==='surf1Dwrite'){

      if(![3,4].includes(n.args.length)||n.args[1]?.kind!=='id'||n.args.length===4&&(n.args[3]?.kind!=='id'||n.args[3].name!=='cudaBoundaryModeTrap'))this.fail('surf1Dwrite requires a float4 value, checked global X byte offset and default or explicit trap mode.',n);
      const surface=this.lookup(n.args[1].name,n.args[1]),value=this.expr(n.args[0]),x=this.expr(n.args[2]);
      if(surface.kind!=='surface'||surface.dimension!=='2d'||value.type!=='vec4<f32>'||!['i32','u32','cw_size64'].includes(x.type)||!this.surfaceGridCoordinate(n.args[2],'x',16))this.fail('surf1Dwrite requires float4 and globalX * sizeof(float4); other offsets cannot be checked before dispatch.',n);
      if(surface.storeFormat&&surface.storeFormat!=='rgba32float')this.fail('A surface cannot mix storage formats.',n);
      surface.storeFormat='rgba32float';surface.metadata.format='rgba32float';surface.metadata.coordinates='global-x';
      return this.result(n,'void',`textureStore(${surface.code}, vec2<i32>(${this.convert(x.code,x.type,'i32',n)} / 16i, 0i), ${value.code})`,[...value.pre,...x.pre]);
    }
    if(name==='surf3Dwrite'){

      if(![5,6].includes(n.args.length)||n.args[1].kind!=='id'||n.args.length===6&&(n.args[5].kind!=='id'||n.args[5].name!=='cudaBoundaryModeTrap'))this.fail('surf3Dwrite supports float or byte surfaces with checked global XYZ coordinates and default or explicit cudaBoundaryModeTrap.',n);
      const surface=this.lookup(n.args[1].name,n.args[1]),value=this.expr(n.args[0]),coords=n.args.slice(2,5).map(a=>this.expr(a)),bytes=value.type==='cw_uchar'?1:4;
      if(surface.kind!=='surface'||surface.dimension!=='3d'||!['f32','cw_uchar'].includes(value.type)||coords.some(c=>!['i32','u32','cw_size64'].includes(c.type))||!['x','y','z'].every((axis,i)=>this.surfaceGridCoordinate(n.args[i+2],axis,i===0?bytes:1)))this.fail('Surface writes require float or byte values and checked global XYZ byte offsets; other coordinates cannot be bounds-checked before dispatch.',n);
      const format=bytes===1?'rgba8unorm':'r32float';if(surface.storeFormat&&surface.storeFormat!==format)this.fail('A surface cannot mix float and byte writes.',n);surface.storeFormat=format;surface.metadata.format=format;if(bytes===1)surface.metadata.sourceElementType='cw_uchar';
      const code=coords.map(c=>this.convert(c.code,c.type,'i32',n)),colour=bytes===1?`vec4<f32>(f32(${value.code}) / 255.0f, 0.0f, 0.0f, 1.0f)`:`vec4<f32>(${value.code}, 0.0f, 0.0f, 0.0f)`;
      return this.result(n,'void',`textureStore(${surface.code}, vec3<i32>(${code[0]} / ${bytes}i, ${code[1]}, ${code[2]}), ${colour})`,[...value.pre,...coords.flatMap(c=>c.pre)]);
    }
    if(name==='surf2Dwrite'){
      if(n.args.length!==5||n.args[1].kind!=='id'||n.args[4].kind!=='id'||n.args[4].name!=='cudaBoundaryModeTrap')this.fail('surf2Dwrite supports float surfaces with checked global XY coordinates and cudaBoundaryModeTrap.',n);
      const surface=this.lookup(n.args[1].name,n.args[1]),value=this.expr(n.args[0]),x=this.expr(n.args[2]),y=this.expr(n.args[3]);
      if(surface.kind!=='surface'||surface.dimension!=='2d'||value.type!=='f32'||!['i32','u32'].includes(x.type)||!['i32','u32'].includes(y.type)||!this.surfaceGridCoordinate(n.args[2],'x',4)||!this.surfaceGridCoordinate(n.args[3],'y',1))this.fail('Surface writes require float values, byte offset globalX * 4 and row globalY; other coordinates cannot be bounds-checked before dispatch.',n);
      return this.result(n,'void',`textureStore(${surface.code}, vec2<i32>(i32(${x.code}) / 4i, i32(${y.code})), vec4<f32>(${value.code}, 0.0f, 0.0f, 0.0f))`,[...value.pre,...x.pre,...y.pre]);
    }
    if(name==='tex2D'&&['uchar','unsigned char'].includes(n.callee.templateArgument)) {
      if(n.args.length!==3||n.args[0].kind!=='id')this.fail('Byte tex2D requires a bound texture and two coordinates.',n);
      const texture=this.lookup(n.args[0].name,n.args[0]),coords=n.args.slice(1).map(a=>this.expr(a));
      if(texture.kind!=='texture'||texture.dimension!=='2d'||texture.format!=='r8uint'||coords.some(c=>!numeric(c.type)))this.fail('Byte tex2D requires a matching unsigned-byte texture and numeric coordinates.',n);
      this.byte2DSamplingUsed=true;
      return this.result(n,'cw_uchar',`cw_sample_byte2d(${texture.code}, vec2<f32>(${coords.map(c=>this.convert(c.code,c.type,'f32',n)).join(', ')}))`,coords.flatMap(c=>c.pre));
    }
    if(name==='tex2D'){const type=n.callee.templateArgument,vector=['float2','float4'].includes(type),format=type==='float2'?'rg32float':type==='float4'?'rgba32float':'r32float';if(!['float','float2','float4'].includes(type)||n.args.length!==3||n.args[0].kind!=='id')this.fail('tex2D supports a bound float, float2 or float4 texture and two numeric coordinates.',n);const texture=this.lookup(n.args[0].name,n.args[0]),coords=n.args.slice(1).map(a=>this.expr(a));if(texture.kind!=='texture'||texture.dimension!=='2d'||texture.format!==format||coords.some(c=>!['f32','i32','u32','cw_uchar'].includes(c.type)))this.fail('tex2D requires a matching kernel texture parameter and numeric coordinates.',n);this[vector?'rgba2DSamplingUsed':'float2DSamplingUsed']=true;return this.result(n,type==='float2'?'vec2<f32>':type==='float4'?'vec4<f32>':'f32',`cw_sample_${vector?'rgba':'float'}2d(${texture.code}, ${texture.sampler}, vec2<f32>(${coords.map(c=>this.convert(c.code,c.type,'f32',n)).join(', ')}), ${texture.coordinateScale}, ${texture.pixelPoint})${type==='float2'?'.xy':''}`,coords.flatMap(c=>c.pre));}
    if(name==='tex1Dfetch'){
      const kind=n.callee.templateArgument;if(!['float','uint'].includes(kind)||n.args.length!==2||n.args[0].kind!=='id')this.fail('tex1Dfetch supports float normalized bytes or uint elements and one integer index.',n);
      const texture=this.lookup(n.args[0].name,n.args[0]),index=this.expr(n.args[1]);if(texture.kind!=='texture'||texture.linearFetch!==kind||!['i32','u32'].includes(index.type))this.fail('tex1Dfetch requires a matching linear texture and an integer index.',n);
      this.linearFetchUsed ||= new Set();this.linearFetchUsed.add(kind);return this.result(n,kind==='uint'?'u32':'f32',`cw_fetch_${kind}(${texture.code}, u32(${index.code}), ${texture.lengthCode})`,index.pre);
    }
    if(name==='tex3D'){if(!['float','float4'].includes(n.callee.templateArgument)||n.args.length!==4||n.args[0].kind!=='id')this.fail('tex3D supports a bound texture object and three float coordinates, returning float or float4.',n);const texture=this.lookup(n.args[0].name,n.args[0]);if(texture.kind!=='texture'||texture.dimension!=='3d')this.fail('tex3D requires a kernel texture parameter.',n);const coords=n.args.slice(1).map(a=>this.expr(a));if(coords.some(c=>c.type!=='f32'))this.fail('tex3D coordinates must be floats.',n);const vector=n.callee.templateArgument==='float4';if(vector)this.float4_3DSamplingUsed=true;else this.float3DSamplingUsed=true;return this.result(n,vector?'vec4<f32>':'f32',`${vector?'cw_sample_float4_3d':'cw_sample_float3d'}(${texture.code}, ${texture.sampler}, vec3<f32>(${coords.map(c=>c.code).join(', ')}), ${texture.coordinateScale}, ${texture.pixelPoint})`,coords.flatMap(c=>c.pre));}
    if (name === '__syncthreads') { if (n.args.length) this.fail('__syncthreads takes no arguments.', n); return this.result(n, 'void', 'workgroupBarrier()'); }
    if(name==='atomicCAS'){
      if(n.args.length!==3||n.args[0].kind!=='unary'||n.args[0].op!=='&'||!['index','id'].includes(n.args[0].value.kind))this.fail('atomicCAS requires &buffer[index] or &sharedScalar, compare and replacement.',n);
      const target=this.expr(n.args[0].value,true),compare=this.expr(n.args[1]),replacement=this.expr(n.args[2]);
      if(!target.atomic||!['i32','u32'].includes(target.type))this.fail('atomicCAS requires 32-bit integer atomic storage.',n);this.writable(target,n.args[0].value);
      const id=`cw_cas_${this.temp++}`,type=target.type;
      return this.result(n,type,id+'_old',[...target.pre,`let ${id}_ptr = &${target.code};`,...compare.pre,`let ${id}_compare = ${this.convert(compare.code,compare.type,type,n)};`,...replacement.pre,`let ${id}_value = ${this.convert(replacement.code,replacement.type,type,n)};`,`var ${id}_old: ${type};`,'loop {',`  let ${id}_result = atomicCompareExchangeWeak(${id}_ptr, ${id}_compare, ${id}_value);`,`  ${id}_old = ${id}_result.old_value;`,`  if (${id}_result.exchanged || ${id}_old != ${id}_compare) { break; }`,'}']);
    }
    const atomics = {atomicAdd: 'atomicAdd', atomicMin: 'atomicMin', atomicMax: 'atomicMax', atomicExch: 'atomicExchange'};
    if (atomics[name]) {
      if (n.args.length !== 2 || n.args[0].kind !== 'unary' || n.args[0].op !== '&' || !['index','id'].includes(n.args[0].value.kind)) this.fail(`${name} requires &buffer[index] or &sharedScalar and a scalar value.`, n);
      const target = this.expr(n.args[0].value, true), value = this.expr(n.args[1]);
      if (!target.atomic || !['u32', 'i32'].includes(target.type)) this.fail('Only integer buffer/shared-array atomics are supported; CUDA float atomicAdd is not silently emulated.', n);
      this.writable(target, n.args[0].value);
      return this.result(n, target.type, `${atomics[name]}(&${target.code}, ${this.convert(value.code, value.type, target.type, n)})`, [...target.pre, ...value.pre]);
    }
    const aliasType=n.aliasType;if(aliasType&&['f32','i32','u32','bool','cw_uchar'].includes(aliasType)){if(n.args.length!==1)this.fail('Scalar type aliases require one constructor argument.',n);const value=n.args[0];delete n.args;delete n.callee;Object.assign(n,{kind:'cast',target:aliasType,value});return this.expr(n);}
    const casts = {uchar:'cw_uchar',float: 'f32', int: 'i32', uint: 'u32', bool: 'bool'};
    if(name==='float'&&n.args.length===1){const value=this.expr({kind:'cast',target:'f32',value:n.args[0],token:n.token});return this.result(n,'f32',value.code,value.pre);}
    const args = n.args.map(a => this.argument(a)), pre = args.flatMap(a => a.pre);
    if(name==='__mul24'||name==='__umul24'){
      if(args.length!==2||args.some(a=>!['i32','u32'].includes(a.type)))this.fail(`${name} requires two 32-bit integer arguments.`,n);
      const signed=name==='__mul24',type=signed?'i32':'u32';
      this.integerIntrinsics.add(name);
      return this.result(n,type,`cw_${signed?'mul24':'umul24'}(${args.map(a=>this.convert(a.code,a.type,type,n)).join(', ')})`,pre);
    }
    if(name==='make_uchar4'){if(args.length!==4||args.some(a=>!numeric(a.type)&&a.type!=='bool'))this.fail('make_uchar4 requires four scalar components.',n);const components=args.map((a,i)=>{const value=a.type==='f32'?`u32(i32(${a.code}))`:this.convert(a.code,a.type,'u32',n);return `((${value} & 255u) << ${i*8}u)`;});return this.result(n,'cw_uchar4',`(${components.join(' | ')})`,pre);}
    if (casts[name]) { if (args.length !== 1) this.fail('Scalar casts require one argument.', n); return this.result(n, casts[name], this.convert(args[0].code, args[0].type, casts[name], n), pre); }
    if (/^make_(float|uint|int)[234]$/.test(name)) {
      if(name==='make_float3'&&args.length===1&&args[0].type==='vec4<f32>')return this.result(n,'vec3<f32>',`${args[0].code}.xyz`,pre);
      if(name==='make_float4'&&args.length===2&&args[0].type==='vec3<f32>'&&args[1].type==='f32')return this.result(n,'vec4<f32>',`vec4<f32>(${args[0].code}, ${args[1].code})`,pre);
      const count = Number(name.at(-1)); if (args.length !== count && args.length !== 1) this.fail(`${name} needs ${count} arguments.`, n);
      const element=name.startsWith('make_uint')?'u32':name.startsWith('make_int')?'i32':'f32';
      return this.result(n, `vec${count}<${element}>`, `vec${count}<${element}>(${args.map(a => this.convert(a.code, a.type, element, n)).join(', ')})`, pre);
    }
    if(name==='length'){if(args.length!==1||!vectorLength(args[0].type)||vectorElement(args[0].type)!=='f32')this.fail('length requires a float vector.',n);return this.result(n,'f32',`length(${args[0].code})`,pre);}
    if(name==='dot'||name==='normalize'){const type=args[0]?.type;if(args.length!==(name==='dot'?2:1)||!vectorLength(type)||vectorElement(type)!=='f32'||(name==='dot'&&args[1].type!==type))this.fail(name+' requires matching float vectors.',n);return this.result(n,name==='dot'?'f32':type,`${name}(${args.map(a=>a.code).join(', ')})`,pre);}
    if(['fminf','fmaxf'].includes(name)&&args.some(a=>vectorLength(a.type))){const type=args[0]?.type;if(args.length!==2||!vectorLength(type)||vectorElement(type)!=='f32'||args[1].type!==type)this.fail(name+' requires matching float vectors.',n);return this.result(n,type,`${name==='fminf'?'min':'max'}(${args.map(a=>a.code).join(', ')})`,pre);}
    if(name==='__fdividef'){if(args.length!==2)this.fail('__fdividef requires two arguments.',n);return this.result(n,'f32',`(${args.map(a=>this.convert(a.code,a.type,'f32',n)).join(' / ')})`,pre);}
    if(name==='exp'){if(args.length!==1||args[0].type!=='f32')this.fail('exp supports the float overload only.',n);return this.result(n,'f32',`exp(${args[0].code})`,pre);}
    if(name==='sqrt'){if(args.length!==1||args[0].type!=='f32')this.fail('sqrt supports the single float overload only; double/integer overloads are unavailable.',n);return this.result(n,'f32',`sqrt(${args[0].code})`,pre);}
    if(name==='roundf'){if(args.length!==1||args[0].type!=='f32')this.fail('roundf requires one float argument.',n);this.roundAwayUsed=true;return this.result(n,'f32',`cw_round_away(${args[0].code})`,pre);}
    if(name==='abs'){if(args.length!==1||!(args[0].type==='i32'||narrow(args[0].type)))this.fail('abs requires a signed integer or promoted narrow integer.',n);return this.result(n,'i32',`abs(${this.convert(args[0].code,args[0].type,'i32',n)})`,pre);}
    if(name==='__saturatef'){if(n.args.length!==1)this.fail('__saturatef requires one float argument.',n);const a=args[0];if(a.type!=='f32')this.fail('__saturatef requires a float argument.',n);return this.result(n,'f32',`clamp(${a.code}, 0.0f, 1.0f)`,a.pre);}
    if(name==='fabs'&&(args.length!==1||args[0].type!=='f32'))this.fail('fabs supports the CUDA float overload; double precision is unavailable.',n);
    const unary = {sinf: 'sin', cosf: 'cos', tanf: 'tan', sqrtf: 'sqrt', rsqrtf: 'inverseSqrt', expf: 'exp', __expf:'exp', exp2f: 'exp2', logf: 'log', __logf:'log', log2f: 'log2', fabs:'abs', fabsf: 'abs', floorf: 'floor', ceilf: 'ceil', truncf: 'trunc'};
    const binary = {fminf: 'min', fmaxf: 'max', powf: 'pow', atan2f: 'atan2'};
    if (unary[name] || binary[name] || name === 'fmaf') {
      const count = unary[name] ? 1 : binary[name] ? 2 : 3;
      if (args.length !== count) this.fail(`${name} requires ${count} arguments.`, n);
      const target = unary[name] || binary[name] || 'fma';
      return this.result(n, 'f32', `${target}(${args.map(a => this.convert(a.code, a.type, 'f32', n)).join(', ')})`, pre);
    }
    if (['min', 'max'].includes(name)) {
      if (args.length !== 2) this.fail(`${name} needs two arguments.`, n);
      const type = this.common(args[0].type==='cw_uchar'?'i32':args[0].type, args[1].type==='cw_uchar'?'i32':args[1].type, n);
      if (!numeric(type)) this.fail('min/max accept scalars.', n);
      return this.result(n, type, `${name}(${args.map(a => this.convert(a.code, a.type, type, n)).join(', ')})`, pre);
    }
    let helper = this.functions.get(name);
    if(this.overloads.has(name)){const matches=this.overloads.get(name).filter(f=>args.length<=f.params.length&&f.params.every((p,i)=>i>=args.length?p.defaultValue!==undefined:typeName(p.pointer?(isArray(args[i].type)?args[i].type.element:null):args[i].type)===typeName(p.type)));if(matches.length!==1)this.fail('Overload '+name+' requires one exact parameter-type match; implicit conversions and ambiguous calls are unsupported.',n);helper=matches[0];}
    if(!helper&&this.templates){
      helper=this.templates.deduce(name,args.map(a=>a.type),n.callee);
      if(helper)for(const fn of this.ast.functions)if(fn.qualifier==='__device__'&&!this.functions.has(fn.name)){this.functions.set(fn.name,fn);if(!fn.params.some(p=>p.pointer))this.helpers.push(fn);}
    }
    if (!helper || helper.qualifier !== '__device__') this.fail(`Unsupported function '${name}'. CUDA host APIs, warp intrinsics, dynamic launches and libraries are not available.`, n);
    if(args.length>helper.params.length||helper.params.slice(args.length).some(p=>p.defaultValue===undefined))this.fail(`Wrong number of arguments for '${name}'.`,n);
    for(const param of helper.params)if(param.defaultValue!==undefined&&!['f32','i32','u32','bool','cw_uchar'].includes(param.type))this.fail('Default arguments require scalar value parameters.',param);
    while(args.length<helper.params.length){const value=structuredClone(helper.params[args.length].defaultValue);n.args.push(value);const argument=this.argument(value);args.push(argument);pre.push(...argument.pre);}
    if(helper.params.some(p=>p.pointer))helper=this.bindPointerHelper(helper,args,n);
    if(helper.params.some(p=>p.reference))helper=this.bindReferenceHelper(helper,args,n);
    const caller=this.currentFunction.name,edges=this.helperCalls.get(caller)||new Set();edges.add(helper.name);this.helperCalls.set(caller,edges);
    const reaches=(from,target,seen=new Set())=>{if(from===target)return true;if(seen.has(from))return false;seen.add(from);return [...(this.helperCalls.get(from)||[])].some(next=>reaches(next,target,seen));};
    if(reaches(helper.name,caller))this.fail('Recursive helper calls are unsupported.',n);
    n.callee.name=helper.name;n.callName=helper.name;
    const references=new Set();n.constRefTemporaries=[];n.localPointerArgs=helper.params.map(p=>!!p.localPointer);n.referenceArgs=helper.params.map(p=>!!p.reference);n.groupArgs=helper.params.map(p=>p.type==='thread-block');n.pointerArgs=helper.params.map(p=>!!p.pointer);
    const codes=args.map((a,i)=>{const p=helper.params[i];if(p.type==='texture3d'){const shape=textureShape(p.textureSampling),symbol=a.rootSymbol;if(symbol?.kind!=='texture'||symbol.dimension!==shape.dimension||symbol.format!==shape.format)this.fail('Texture helper argument requires the same inferred sampling format.',n.args[i]);return symbol.code+', '+symbol.sampler+(symbol.linearFetch?', '+symbol.lengthCode:'')+(['tex2D','tex2Dfloat2','tex2Dfloat4','tex3D','tex3Dfloat4'].includes(p.textureSampling)?', '+symbol.coordinateScale+', '+symbol.pixelPoint:'');}if(p.localPointer){if(references.has(a.rootSymbol))this.fail('Aliased local pointer/reference arguments are unsupported.',n.args[i]);references.add(a.rootSymbol);return a.pointerCode;}if(p.pointer)return a.pointerCode;if(p.type==='thread-block'){if(a.type!=='thread-block'||a.rootSymbol?.kind!=='thread-block')this.fail('thread_block arguments require a block handle.',n.args[i]);return null;}if(!p.reference)return this.convert(a.code,a.type,p.type,n);
      const node=n.args[i],s=a.rootSymbol;
      if(s?.kind==='shared'&&s.atomic)this.fail('Atomic shared values cannot bind ordinary references.',node);
      if(p.constant){if(node.kind==='member'&&node.base.type==='cw_uchar4'&&p.type==='cw_uchar'){const temp='cw_const_ref_'+this.temp++;pre.push(`var ${temp}: cw_uchar = ${a.code};`);n.constRefTemporaries[i]=true;return '&'+temp;}if(s?.rootBufferName)this.fail('Const references to storage elements are unsupported; copy the value to a local first.',node);if(a.type!==p.type||!(numeric(p.type)||p.type==='cw_uchar4'||vectorLength(p.type)||this.structs.has(p.type)))this.fail('Const references require the exact scalar, vector or struct type.',node);if(s&&references.has(s))this.fail('Aliased reference arguments are unsupported.',node);if(s)references.add(s);if(s?.kind==='reference')return s.pointerCode;if(s?.kind==='shared'){if(!['id','index'].includes(node.kind))this.fail('Shared references require whole scalars/vectors or array elements.',node);return '&'+a.code;}if(s?.kind==='local'&&!s.constant&&['id','member'].includes(node.kind))return '&'+a.code;const temp='cw_const_ref_'+this.temp++;pre.push(`var ${temp}: ${p.type} = ${a.code};`);n.constRefTemporaries[i]=true;return '&'+temp;}
      if(!['id','index'].includes(node.kind)||!s||!['local','reference','shared'].includes(s.kind)||s.constant||isArray(a.type)||!(numeric(a.type)||vectorLength(a.type))||a.type!==p.type)this.fail('Reference arguments require a mutable scalar/vector or array element in local or shared memory, of the exact type.',node);
      if(references.has(s))this.fail('Aliased reference arguments are unsupported.',node);references.add(s);return s.kind==='reference'?s.pointerCode:`&${a.code}`;
    });
    return this.result(n, helper.result, `f_${helper.name}(${[...codes.filter(c=>c!==null),'cw_thread','cw_block','cw_grid'].join(', ')})`, pre);
  }
  writable(target, n) {
    const s = target.rootSymbol;
    if(target.packedBase&&!target.packedAtomic&&(n.base?.kind!=='id'||s?.kind!=='local'||s.type!=='cw_uchar4'))this.fail('Byte component writes require a named local uchar4; write complete uchar4 records to storage or shared memory.',n);
    if (!s || !['id', 'index', 'member'].includes(n.kind) || isArray(target.type)) this.fail('Assignment requires a scalar/vector variable or array element.', n);
    if (s.constant) this.fail(`Cannot write through const '${s.name}'.`, n);
    if (s.kind === 'uniform') this.fail('Scalar kernel parameters are read-only in this subset. Copy the parameter to a local variable first.', n);
  }
  effect(n) {
    if(n.kind==='sequence')return n.expressions.flatMap(e=>this.effect(e));
    if (n.kind === 'assign') {
      walk(n.left,node=>{if(node.kind==='unary'&&['++','--'].includes(node.op))this.fail('Increment/decrement inside assignment destinations are unsupported.',node);});
      if(n.op==='='&&n.right.kind==='assign'){let current=n;while(current.kind==='assign'){if(current.op!=='='||current.left.kind!=='id'||!['local','reference'].includes(this.lookup(current.left.name,current.left).kind))this.fail('Chained assignment requires named local variables or references and =.',current);current=current.right;}const inner=this.effect(n.right),target=this.expr(n.left,true);this.writable(target,n.left);const value=this.expr(n.right.left);n.type=target.type;return [...inner,`${target.code} = ${this.convert(value.code,value.type,target.type,n)};`];}
      if(n.left.kind==='id'&&n.op==='='){
        const symbol=this.lookup(n.left.name,n.left);
        if(['buffer','buffer-alias'].includes(symbol.kind)){
          const value=this.argument(n.right);if(!symbol.offsetCode||!value.pointerCode||value.rootSymbol?.code!==symbol.code||typeName(value.type)!==typeName(symbol.type))this.fail('Pointer reassignment must stay within the same typed allocation.',n);
          if(value.rootSymbol.constant&&!symbol.constant)this.fail('Cannot discard const through pointer reassignment.',n);
          n.left.symbol=symbol;n.pointerRebind=true;n.type=symbol.type;
          return [...value.pre,`${symbol.offsetCode} = ${value.pointerCode};`];
        }
      }
      if(n.left.kind==='id'&&['+=','-='].includes(n.op)){
        const symbol=this.lookup(n.left.name,n.left);
        if(['buffer','buffer-alias'].includes(symbol.kind)){
          const value=this.expr(n.right);if(!['i32','u32'].includes(value.type)||!symbol.offsetCode)this.fail('Pointer updates require a 32-bit integer offset.',n);
          n.left.symbol=symbol;n.pointerShift=true;n.type=symbol.type;
          return [...value.pre,`${symbol.offsetCode} ${n.op} ${this.convert(value.code,value.type,'i32',n)};`];
        }
      }
      if(n.left.kind==='index'&&n.left.base.kind==='id'&&this.lookup(n.left.base.name,n.left).kind==='pointer-array'){
        const slots=this.lookup(n.left.base.name,n.left),address=n.right.kind==='unary'&&n.right.op==='&'?n.right.value:null;
        if(n.op!=='='||address?.kind!=='index'||address.base.kind!=='id')this.fail('Pointer array assignments require a shared-array element address.',n);
        const root=this.lookup(address.base.name,address);if(root.kind!=='shared'||root.atomic||!isArray(root.type)||root.type.element!==slots.elementType)this.fail('Pointer arrays require a matching non-atomic shared array.',n);
        if(root.constant&&!slots.constant)this.fail('Cannot discard const in a pointer array.',n);
        if(slots.pointerRoot&&slots.pointerRoot!==root)this.fail('A pointer array must refer to one shared allocation.',n);
        slots.pointerRoot=root;const left=this.expr(n.left.base),index=this.expr(n.left.index),offset=this.expr(address.index);if(!['i32','u32'].includes(index.type)||!['i32','u32'].includes(offset.type))this.fail('Pointer array offsets must be 32-bit integers.',n);
        n.pointerArrayAssignment=true;n.left.type='i32';n.type='i32';
        return [...left.pre,...index.pre,...offset.pre,`${slots.code}[${index.code}] = ${this.convert(offset.code,offset.type,'i32',n)};`];
      }
      const target = this.expr(n.left, true); this.writable(target, n.left); let value = this.expr(n.right);let packedPre;if(target.packedAtomic){n.packedAtomicAssignment=true;const tmp='cw_byte_value_'+this.temp++;packedPre=[...value.pre,`let ${tmp}: ${typeName(value.type)} = ${value.code};`,...target.pre];value={...value,code:tmp,pre:[]};}
      if(n.op!=='='&&vectorLength(target.type)){const op=n.op.slice(0,-1);if(n.left.kind!=='id'||vectorElement(target.type)!=='f32'||!['+','-','*','/'].includes(op)||![target.type,'f32'].includes(value.type))this.fail('Vector compound assignments require a named float vector and matching vector or float scalar.',n);const rhs=value.type===target.type?value.code:`${target.type}(${value.code})`;n.operandType=target.type;n.type=target.type;return [...target.pre,...value.pre,`${target.code} = ${target.code} ${op} ${rhs};`];}
      if(target.type==='cw_uchar4'&&n.op!=='=')this.fail('uchar4 compound arithmetic requires explicit byte components.',n);
      let code = this.convert(value.code, value.type, target.type, n);
      if (n.op !== '=') {
        const op = n.op.slice(0, -1); if (target.atomic) this.fail('Use explicit atomicAdd/Min/Max/Exch rather than compound assignments to atomic arrays.', n);
        const type = ['<<', '>>'].includes(op) ? (narrow(target.type)?'i32':target.type) : this.common(narrow(target.type)?'i32':target.type, value.type==='cw_uchar'?'i32':value.type, n), rhsType = ['<<', '>>'].includes(op) ? 'u32' : type;
        if (['<<', '>>', '%', '&', '|', '^'].includes(op) && !['i32', 'u32'].includes(type)) this.fail('Integer operator requires integer operands.', n);
        if(type==='cw_f64'){const fn={'+':'cw_d_add','-':'cw_d_sub','*':'cw_d_mul','/':'cw_d_div'}[op];if(!fn)this.fail('Unsupported double compound operator.',n);this.float64Used=true;code=this.convert(`${fn}(${this.convert(target.code,target.type,type,n)}, ${this.convert(value.code,value.type,type,n)})`,type,target.type,n);}else if(op==='/'&&type==='f32'){this.compensatedDivisionUsed=true;code=this.convert(`cw_divide_f32(${this.convert(target.code,target.type,type,n)}, ${this.convert(value.code,value.type,rhsType,n)})`,type,target.type,n);}else code = this.convert(`(${this.convert(target.code, target.type, type, n)} ${op} ${this.convert(value.code, value.type, rhsType, n)})`, type, target.type, n);
        n.operandType = type;
      }
      n.type = target.type;
      if(target.packedAtomic){this.packedAtomicUsed=true;return [...packedPre,`cw_store_byte(&${target.packedBase}, ${target.packedShiftCode??target.packedShift+'u'}, u32(${code}));`];}
      if(target.packedBase)return [...target.pre,...value.pre,`${target.packedBase} = (${target.packedBase} & ${(~(255<<target.packedShift))>>>0}u) | ((u32(${code}) & 255u) << ${target.packedShift}u);`];
      return [...target.pre, ...value.pre, target.atomic ? `atomicStore(&${target.code}, ${code});` : `${target.code} = ${code};`];
    }
    if (n.kind === 'unary' && ['++', '--'].includes(n.op)) {
      const target = this.expr(n.value, true); this.writable(target, n.value); if (target.atomic || !numeric(target.type)) this.fail('Increment/decrement require a non-atomic scalar.', n);
      n.type = target.type;if(target.packedAtomic){this.packedAtomicUsed=true;return [...target.pre,`cw_store_byte(&${target.packedBase}, ${target.packedShiftCode??target.packedShift+'u'}, u32(i32(${target.code}) ${n.op==='++'?'+':'-'} 1i));`];}if(['cw_short','cw_ushort'].includes(target.type))return [...target.pre,`${target.code} = ${this.convert(`i32(${target.code}) ${n.op==='++'?'+':'-'} 1i`,'i32',target.type,n)};`];if(target.type==='cw_uchar'&&!target.packedBase)return [...target.pre,`${target.code} = (${target.code} ${n.op==='++'?'+':'-'} 1u) & 255u;`];if(target.packedBase)return [...target.pre,`${target.packedBase} = (${target.packedBase} & ${(~(255<<target.packedShift))>>>0}u) | ((u32(i32(${target.code}) ${n.op==='++'?'+':'-'} 1i) & 255u) << ${target.packedShift}u);`]; return [...target.pre, `${target.code} ${n.op === '++' ? '+=' : '-='} ${target.type}(1);`];
    }
    const value = this.expr(n);
    if (n.kind !== 'call') this.fail('Only assignments, increments and function calls may stand alone as statements.', n);
    if (n.callName === '__syncthreads') return [...value.pre, 'workgroupBarrier();', ...(this.usage.storageBarrier ? ['storageBarrier();'] : [])];
    return [...value.pre, value.type === 'void' ? `${value.code};` : `_ = ${value.code};`];
  }
  declare(n) {
    if(n.type==='cw_extent')this.fail('cudaExtent supports read-only by-value kernel parameters only.',n);
    if(['texture3d','surface2d'].includes(n.type))this.fail('Texture and surface handles require kernel or supported helper parameters, not local aliases.',n);
    if(n.init?.kind==='shared-conversion'){
      if(!n.pointer||n.reference||n.shared||n.external||n.dimensions.length||n.type!==n.init.target)this.fail('Shared conversion requires a matching local pointer declaration.',n);
      if(!n.init.conversions.includes(false)&&!n.constant)this.fail('Cannot discard const from shared wrapper conversion.',n);
      // A stateless conversion exposes the dispatch's dynamic shared allocation.
      n.pointer=false;n.shared=true;n.external=true;n.dimensions=[null];n.init=null;
      if(n.constant)this.fail('Const shared wrapper views are not yet supported.',n);
    }
    if(n.external&&(!n.shared||n.pointer||n.reference||n.constant||n.init||n.dimensions.length!==1||n.dimensions[0]!==null))this.fail('extern is supported only as extern __shared__ T name[].',n);
    if(n.reference)this.fail('References are supported only as helper parameters, not local declarations.',n);
    if(n.pointer&&n.dimensions.length){
      const length=n.dimensions.length===1?constantValue(n.dimensions[0]):null;
      if(n.shared||n.external||n.init||!Number.isInteger(length)||length<1||length>256||!(numeric(n.type)||vectorLength(n.type)))this.fail('Pointer arrays require 1..256 local slots and scalar/vector shared pointees.',n);
      const code='v_'+n.name,symbol={name:n.name,kind:'pointer-array',type:arrayOf(arrayOf(n.type),length),elementType:n.type,code,constant:n.constant};this.add(n.name,symbol,n);n.symbol=symbol;n.resolvedDimensions=[length];n.resolvedType=arrayOf('i32',length);
      return [`var ${code}: array<i32, ${length}>;`];
    }
    if(n.pointer){
      const parts=pointerParts(n.init),baseNode=parts?.base;let offsetNode=parts?.offset;
      if(n.shared||n.dimensions.length||baseNode?.kind!=='id')this.fail('Local pointers require a buffer alias with an optional integer offset.',n);
      const base=this.lookup(baseNode.name,baseNode);if(!['buffer','buffer-alias','shared'].includes(base.kind)||base.type.element!==n.type||(base.kind==='shared'&&base.atomic))this.fail('Local pointers can alias only same-type storage buffers or non-atomic shared arrays.',n);
      if(base.constant&&!n.constant)this.fail('Cannot discard const through a buffer alias.',n);
      if(n.byteOffsetCast&&n.type!=='cw_uchar'){
        const stride=cudaValueSize(n.type);if(!stride)this.fail('Byte-address casts need a known pointee alignment.',n);
        const multiple=node=>{if(node.kind==='literal')return Number(node.value.replace(/[uU]$/,''))%stride===0;if(node.kind==='binary'&&node.op==='*')return multiple(node.left)||multiple(node.right);return false;};
        if(!multiple(offsetNode)){
          const factor=offsetNode?.kind==='binary'&&offsetNode.op==='*'?[offsetNode.left,offsetNode.right].find(a=>a.kind==='id'&&this.lookup(a.name,a).kind==='uniform'&&['i32','u32','cw_short','cw_ushort'].includes(this.lookup(a.name,a).type)):null;
          if(!factor)this.fail('Byte-address casts to wider types need provable alignment or an integer launch pitch factor.',n);
          this.pointerConstraints.push({name:factor.name,minimum:0,multipleOf:stride});
        }
        offsetNode={kind:'binary',op:'/',left:offsetNode,right:{kind:'literal',value:String(stride),token:n.token},token:n.token};
      }
      const offset=offsetNode?this.expr(offsetNode):{type:'i32',code:'0i',pre:[]};if(!['i32','u32'].includes(offset.type))this.fail('Buffer alias offsets must be 32-bit integers.',n);
      const offsetCode=`cw_offset_${this.temp++}`,symbol={...base,...(base.kind==='shared'?{sharedPointer:base.code}:{}),name:n.name,constant:n.constant||base.constant,kind:'buffer-alias',offsetCode};this.add(n.name,symbol,n);n.symbol=symbol;n.aliasBase=base;n.aliasOffset=offsetNode;
      return [...offset.pre,`var ${offsetCode} = ${base.offsetCode?base.offsetCode+' + ':''}${this.convert(offset.code,offset.type,'i32',n)};`];
    }
    if(this.structs.has(n.type)&&(n.shared||n.dimensions.length))this.fail('Structs currently support local values only, not shared memory or arrays of structs.',n);
    if (n.type === 'void') this.fail('Variables cannot have void type.', n);
    let type = n.type;const sharedOwner=(this.currentFunction.pointerOrigin||this.currentFunction.name)+':'+n.token.offset+':'+n.name;
    const dims = n.dimensions.map(d => {if(d===null){if(!n.external||!n.shared)this.fail('Unsized arrays require extern __shared__.',n);if(this.dynamicSharedUsed&&this.dynamicSharedOwner!==sharedOwner)this.fail('Only one dynamic shared array is supported; CUDA declarations alias the same allocation.',n);const stride=n.type==='cw_uchar'?1:typeStride(n.type);if(n.type==='bool'||['cw_short','cw_ushort'].includes(n.type)||vectorLength(n.type)===3)this.fail('Dynamic shared arrays require 32-bit scalars or two/four-component vectors.',n);if(!this.dynamicSharedBytes||this.dynamicSharedBytes%stride)this.fail('Set sharedMemoryBytes to a positive multiple of the dynamic shared element size.',n);this.dynamicSharedUsed=true;this.dynamicSharedOwner=sharedOwner;return this.dynamicSharedBytes/stride;}const value = constantValue(d); if (!Number.isSafeInteger(value) || value < 1 || value > 65536) this.fail('Invalid fixed array dimension (1..65536).', n); return value; });
    for (let i = dims.length - 1; i >= 0; i--) type = arrayOf(type, dims[i]);
    if (n.shared && n.init) this.fail('__shared__ variables cannot have an initializer.', n);
    if (isArray(type) && n.init) this.fail('Array initializers are unsupported. Initialize elements explicitly.', n);
    const atomic = n.shared && analyse([this.currentFunction],[]).atomic.has(n.name);
    if (atomic && !['i32', 'u32'].includes(n.type)) this.fail('Shared atomics require int or unsigned int.', n);
    if(n.init?.kind==='initializer')n.init.target=type;
    const init = n.init ? this.expr(n.init) : null;
    if (n.constant && !init && !n.shared) this.fail('A const local variable needs an initializer.', n);
    const code = `${n.shared ? (this.currentFunction===this.kernel?'s':'s_'+(this.currentFunction.pointerOrigin||this.currentFunction.name)) : 'v'}_${n.name}`;
    const symbol = {name: n.name, type, code, constant: n.constant, atomic, kind: n.shared ? 'shared' : 'local',...(n.shared?{sharedOwner}:{})};
    const existing=n.shared&&this.shared.find(x=>x.code===code);
    if(existing){if(existing.sharedOwner!==sharedOwner||typeName(existing.type)!==typeName(type)||existing.atomic!==atomic)this.fail('Shared declaration conflicts across helper specializations.',n);this.add(n.name,existing,n);n.symbol=existing;n.resolvedDimensions=dims;n.resolvedType=type;return [];}
    this.add(n.name, symbol, n); n.symbol = symbol; n.resolvedDimensions = dims; n.resolvedType = type;
    if (n.shared) {
      if (this.shared.some(x => x.code === code)) this.fail('Shared array names must be unique.', n);
      this.shared.push(symbol); return [];
    }
    return [...(init?.pre || []), `${n.constant ? 'let' : 'var'} ${code}: ${typeName(type)}${init ? ` = ${this.convert(init.code, init.type, type, n)}` : ''};`];
  }
  body(n) {
    this.scopes.push(new Map());
    const lines = n.kind === 'block' ? n.body.flatMap(s => this.statement(s)) : this.statement(n);
    this.scopes.pop(); return lines;
  }
  statement(n) {
    switch (n.kind) {
      case 'empty': return [];
      case 'thread-block':
        n.symbol=this.add(n.name,{name:n.name,kind:'thread-block',constant:true},n);return [];
      case 'block': return ['{', ...indent(this.body(n)), '}'];
      case 'decl': return this.declare(n);
      case 'decls': return n.declarations.flatMap(d=>this.declare(d));
      case 'expr': return this.effect(n.value);
      case 'if': { const condition = this.expr(n.condition); return [...condition.pre, `if (${this.convert(condition.code, condition.type, 'bool', n)}) {`, ...indent(this.body(n.yes)), ...(n.no ? ['} else {', ...indent(this.body(n.no))] : []), '}']; }
      case 'do': {
        const condition=this.expr(n.condition);this.loopDepth++;const inner=this.body(n.body);this.loopDepth--;
        if(!condition.pre.length&&((n.condition.kind==='literal'&&constantValue(n.condition)!==0)||(n.condition.kind==='id'&&n.condition.name==='true')))return ['loop {',...indent(inner),'}'];
        return ['loop {',...indent(inner),'  continuing {',...indent(indent([...condition.pre,`break if !${this.convert(condition.code,condition.type,'bool',n)};`])),'  }','}'];
      }
      case 'for': case 'while': {
        this.scopes.push(new Map()); this.loopDepth++;
        const init = n.kind === 'for' && n.init ? (['decl','decls'].includes(n.init.kind) ? this.statement(n.init) : this.effect(n.init)) : [];
        const condition = n.condition ? this.expr(n.condition) : {code: 'true', type: 'bool', pre: []};
        const inner = this.body(n.body), step = n.kind === 'for' && n.step ? this.effect(n.step) : [];
        this.loopDepth--; this.scopes.pop();
        return ['{', ...indent(init), '  loop {', ...indent(indent([...condition.pre, `if (!${this.convert(condition.code, condition.type, 'bool', n)}) { break; }`, ...inner, ...(step.length ? ['continuing {', ...indent(step), '}'] : [])])), '  }', '}'];
      }
      case 'return': {
        if (n.value) { const value = this.expr(n.value); if (this.currentFunction.result === 'void') this.fail('Void functions cannot return a value.', n); return [...value.pre, `return ${this.convert(value.code, value.type, this.currentFunction.result, n)};`]; }
        if (this.currentFunction.result !== 'void') this.fail('Non-void functions must return a value.', n); return ['return;'];
      }
      case 'break': case 'continue': if (!this.loopDepth) this.fail(`${n.kind} is only valid in a loop.`, n); return [`${n.kind};`];
      default: this.fail(`Unsupported statement ${n.kind}.`, n);
    }
  }
  checkRecursion() {
    const edges = new Map(this.ast.functions.map(f => { const e = []; walk(f.body, n => { if (n.kind === 'call' && n.callee.kind === 'id' && this.functions.has(n.callee.name)) e.push(n.callee.name); }); return [f.name, e]; }));
    const visiting = new Set(), done = new Set();
    const dfs = name => { if (visiting.has(name)) this.fail(`Recursive function call involving '${name}' is unsupported.`, this.functions.get(name)); if (done.has(name)) return; visiting.add(name); edges.get(name).forEach(dfs); visiting.delete(name); done.add(name); };
    for (const name of edges.keys()) dfs(name);
  }
  emit() {
    this.checkRecursion();
    if (this.kernel.result !== 'void') this.fail('__global__ kernels must return void.', this.kernel);
    const textures=[],surfaces=[],bufferCount=this.kernel.params.filter(p=>p.pointer).length;
    const bindings = [], scalars = [], header = [`// CUDA WebShader ${COMPILER_VERSION}. Generated from kernel ${this.kernel.name}.`];
    for(const s of this.structs.values())header.push(`struct ${s.type} {`,...s.fields.map(f=>`  cw_field_${f.name}: ${typeName(f.resolvedType)},`),'}');
    const sharedAtomicType = t => isArray(t) ? `array<${sharedAtomicType(t.element)}, ${t.length}>` : `atomic<${t}>`;
    for (const p of this.kernel.params) {
      if (p.shared || p.reference || p.external || p.type === 'void') this.fail('Invalid kernel parameter type.', p);
      if(p.type==='cw_extent'){
        if(p.pointer)this.fail('cudaExtent supports read-only by-value kernel parameters only.',p);
        this.extentUsed=true;const fields=['width','height','depth'];for(const field of fields)scalars.push({name:p.name+'.'+field,type:'u32',sourceType:'cudaExtent-component',field:'cw_extent_'+p.name+'_'+field,offset:scalars.length*4});
        const symbol={name:p.name,type:p.type,code:`cw_extent(${fields.map(field=>`vec2<u32>(cw_params.cw_extent_${p.name}_${field}, 0u)`).join(', ')})`,constant:true,atomic:false,kind:'uniform'};
        this.add(p.name,symbol,p,true);p.symbol=symbol;continue;
      }
      if(!p.pointer&&vectorLength(p.type)){
        const fields='xyzw'.slice(0,vectorLength(p.type)).split(''),element=vectorElement(p.type);
        for(const field of fields)scalars.push({name:p.name+'.'+field,type:element,sourceType:'vector-component',field:'cw_vector_'+p.name+'_'+field,offset:scalars.length*4});
        const symbol={name:p.name,type:p.type,code:`${p.type}(${fields.map(field=>`cw_params.cw_vector_${p.name}_${field}`).join(', ')})`,components:fields.map(field=>p.name+'.'+field),constant:true,atomic:false,kind:'uniform'};this.add(p.name,symbol,p,true);p.symbol=symbol;continue;
      }
      if(p.type==='surface2d'){if(p.pointer)this.fail('Surface objects must be passed by value.',p);const dimensions=new Set();walk(this.kernel.body,n=>{if(n.kind==='call'&&['surf1Dwrite','surf2Dwrite','surf3Dwrite','surf2DLayeredwrite'].includes(n.callee?.name)&&n.args[1]?.kind==='id'&&n.args[1].name===p.name)dimensions.add(n.callee.name==='surf2DLayeredwrite'?'2d-array':n.callee.name==='surf1Dwrite'?'1d':n.callee.name==='surf3Dwrite'?'3d':'2d');});if(dimensions.size>1)this.fail('A surface cannot mix 2D and 3D writes.',p);const dimension=dimensions.has('2d-array')?'2d-array':dimensions.has('3d')?'3d':'2d',binding=bufferCount+this.kernel.params.filter(p=>p.type==='texture3d').length*2+surfaces.length,symbol={name:p.name,type:p.type,code:'cw_surface_'+p.name,kind:'surface',dimension,constant:true};this.add(p.name,symbol,p,true);p.symbol=symbol;const metadata={name:p.name,binding,dimension,format:'r32float',access:'write-only',coordinates:dimension==='3d'?'global-xyz':'global-xy'};symbol.metadata=metadata;surfaces.push(metadata);continue;}
      if(p.type==='texture3d'){if(p.pointer)this.fail('Texture objects must be passed by value.',p);const binding=bufferCount+textures.length*2,samplerBinding=binding+1,sampling=p.textureSampling||'tex3D',{dimension,format}=textureShape(sampling);const symbol={name:p.name,type:p.type,code:'t_'+p.name,sampler:'s_'+p.name,coordinateScale:dimension==='3d'?`vec3<f32>(cw_params.cw_tex_${p.name}_sx, cw_params.cw_tex_${p.name}_sy, cw_params.cw_tex_${p.name}_sz)`:`vec2<f32>(cw_params.cw_tex_${p.name}_sx, cw_params.cw_tex_${p.name}_sy)`,pixelPoint:`cw_params.cw_tex_${p.name}_point`,...(sampling.startsWith('fetch_')?{linearFetch:sampling.slice(6),lengthCode:`cw_params.cw_tex_${p.name}_length`}:{}),dimension,format,kind:'texture',constant:true};this.add(p.name,symbol,p,true);p.symbol=symbol;textures.push({name:p.name,binding,samplerBinding,dimension,format,...(sampling==='tex2Duchar'?{coordinates:'pixel-byte'}:['tex2Dfloat2','tex2Dfloat4'].includes(sampling)?{coordinates:'2d'}:sampling.startsWith('fetch_')?{coordinates:'linear'}:{})});header.push(`@group(0) @binding(${binding}) var ${symbol.code}: texture_${dimension.replace('-','_')}<${['r32uint','r8uint'].includes(format)?'u32':'f32'}>;`,`@group(0) @binding(${samplerBinding}) var ${symbol.sampler}: sampler;`);continue;}
      if (p.pointer) {
        if(['cw_short','cw_ushort'].includes(p.type))this.fail('Short storage pointers need a packed 16-bit ABI; only short values are supported.',p);
        if(this.structs.has(p.type))this.fail('Struct buffer layout is not supported; use local struct values.',p);
        if (p.type === 'bool' || vectorLength(p.type)===3) this.fail('bool* and three-component vector pointers have incompatible CUDA/WGSL layouts. Use 32-bit scalars or two/four-component vectors.', p);
        const atomic = this.usage.atomic.has(p.name), readOnly = p.constant || !this.usage.writes.has(p.name);
        if (p.constant && this.usage.writes.has(p.name)) this.fail(`Cannot write through const buffer '${p.name}'.`, p);
        if (atomic && !['i32', 'u32','cw_uchar4','cw_uchar'].includes(p.type)) this.fail('Only 32-bit integer atomics are supported.', p);
        const binding = bindings.length;
        bindings.push({name: p.name, elementType: p.type, stride: p.type==='cw_uchar'?1:typeStride(p.type), binding, readOnly, atomic});
        const symbol = {name: p.name, rootBufferName:p.name, type: arrayOf(p.type), code: `b_${p.name}`, constant: p.constant, atomic, kind: 'buffer',...(shiftedPointers(this.kernel).has(p.name)?{offsetCode:'cw_pointer_'+p.name}:{})};
        this.add(p.name, symbol, p, true); p.symbol = symbol;this.bufferSymbols.set(p.name,symbol);
        header.push(`@group(0) @binding(${binding}) var<storage, ${readOnly ? 'read' : 'read_write'}> b_${p.name}: array<${atomic ? `atomic<${p.type}>` : p.type==='cw_uchar'?'u32':p.type}>;`);
      } else {
        if ((!numeric(p.type)&&!['bool','cw_uchar4'].includes(p.type))||p.type==='cw_uchar') this.fail('Scalar kernel parameters must be float, int, unsigned int, bool or packed uchar4. Put other vectors in buffers.', p);
        let defaultMetadata={};if(p.defaultValue!==undefined){let value=p.defaultValue.kind==='id'?Number(p.defaultValue.name==='true'):constantValue(p.defaultValue);if(!Number.isFinite(value)||p.type==='i32'&&(Math.trunc(value)<-2147483648||Math.trunc(value)>2147483647)||p.type==='u32'&&(Math.trunc(value)<0||Math.trunc(value)>4294967295))this.fail('Kernel default is outside its supported scalar range.',p);value=p.type==='f32'?Math.fround(value):p.type==='bool'?Number(!!value):p.type==='u32'?value>>>0:value|0;if(!Number.isFinite(value))this.fail('Kernel default overflows its scalar type.',p);defaultMetadata={defaultValue:value};}
        scalars.push({...defaultMetadata,name:p.name,type:p.type==='cw_short'?'i32':['bool','cw_uchar4','cw_ushort'].includes(p.type)?'u32':p.type,...(['bool','cw_uchar4','cw_short','cw_ushort'].includes(p.type)?{sourceType:p.type}:{}),offset:scalars.length*4});
        const symbol = {name: p.name, type: p.type, code: p.type==='bool'?`(cw_params.p_${p.name} != 0u)`:`cw_params.p_${p.name}`, constant: false, atomic: false, kind: 'uniform'};
        this.add(p.name, symbol, p, true); p.symbol = symbol;
      }
    }
    const moduleShared=new Map(),usedGlobalNames=new Set();for(const fn of [this.kernel,...this.helpers])walk(fn.body,n=>{if(n.kind==='id')usedGlobalNames.add(n.name);});
    for(const declaration of this.ast.sharedGlobals||[])if(usedGlobalNames.has(declaration.name)){this.declare(declaration);moduleShared.set(declaration.name,declaration.symbol);}
    const helperLines = [];
    // Helpers cannot capture kernel arguments; explicit scalar arguments only.
    const kernelScope = this.scopes;
    let emittedHelpers=0;
    const emitHelpers=()=>{while(emittedHelpers<this.helpers.length){const helper=this.helpers[emittedHelpers++];
      this.scopes = [new Map(moduleShared)]; this.currentFunction = helper;
      for (const p of helper.params) {
        if (p.shared || p.external || p.type === 'void'||p.type==='surface2d'||p.type==='cw_extent') this.fail('Invalid helper parameter.', p);
        if(p.type==='texture3d'){if(p.pointer||p.reference)this.fail('Texture helper parameters must be passed by value.',p);const {dimension,format}=textureShape(p.textureSampling);p.symbol=this.add(p.name,{name:p.name,type:p.type,code:'cw_texture_'+p.name,sampler:'cw_sampler_'+p.name,coordinateScale:'cw_scale_'+p.name,pixelPoint:'cw_point_'+p.name,...(p.textureSampling.startsWith('fetch_')?{linearFetch:p.textureSampling.slice(6),lengthCode:'cw_length_'+p.name}:{}),dimension,format,kind:'texture',constant:true},p);continue;}
        if(p.pointer){const base=p.boundShared?this.shared.find(s=>s.code===p.boundShared):this.bufferSymbols.get(p.boundBuffer);if(!base)this.fail('Helper buffer pointer was not specialized.',p);p.symbol=this.add(p.name,{...base,...(p.boundShared?{sharedPointer:p.boundShared}:{}),name:p.name,kind:'buffer-alias',constant:p.constant||p.boundConstant,offsetCode:'cw_buffer_offset_'+helper.params.indexOf(p)},p);continue;}
        if(p.type==='thread-block'){if(p.reference)this.fail('thread_block helper parameters must be passed by value.',p);p.symbol=this.add(p.name,{name:p.name,type:p.type,kind:'thread-block',constant:true},p);continue;}
        if(p.reference&&!numeric(p.type)&&!vectorLength(p.type)&&!(p.constant&&(p.type==='cw_uchar4'||vectorLength(p.type)||this.structs.has(p.type))))this.fail('Helper references require numeric scalars or vectors.',p);
        p.symbol = this.add(p.name, {name: p.name, type: p.type, code: p.reference?`(*v_${p.name})`:`v_${p.name}`,pointerCode:p.reference?`v_${p.name}`:undefined,...(p.reference?{referenceSpace:p.referenceSpace||'function'}:{}), constant:p.constant, atomic: false, kind: p.reference?'reference':'local'}, p);
      }
      const body = this.body(helper.body);
      helperLines.push(`fn f_${helper.name}(${[...helper.params.filter(p=>p.type!=='thread-block').map(p => p.type==='texture3d'?`cw_texture_${p.name}: texture_${textureShape(p.textureSampling).dimension.replace('-','_')}<${['fetch_uint','tex2Duchar'].includes(p.textureSampling)?'u32':'f32'}>, cw_sampler_${p.name}: sampler${p.textureSampling.startsWith('fetch_')?`, cw_length_${p.name}: u32`:''}${['tex2D','tex2Dfloat2','tex2Dfloat4','tex3D','tex3Dfloat4'].includes(p.textureSampling)?`, cw_scale_${p.name}: vec${p.textureSampling.startsWith('tex3D')?3:2}<f32>, cw_point_${p.name}: f32`:''}`:p.pointer?`cw_buffer_arg_${helper.params.indexOf(p)}: i32`:`${p.reference||p.constant?'v_':'cw_arg_'}${p.name}: ${p.reference?`ptr<${p.referenceSpace||'function'}, ${p.type}>`:p.type}`),'cw_thread: vec3<u32>','cw_block: vec3<u32>','cw_grid: vec3<u32>'].join(', ')})${helper.result === 'void' ? '' : ` -> ${helper.result}`} {`,...indent(helper.params.filter(p=>p.pointer).map(p=>`var cw_buffer_offset_${helper.params.indexOf(p)}: i32 = cw_buffer_arg_${helper.params.indexOf(p)};`)),...indent(helper.params.filter(p=>p.type!=='thread-block'&&p.type!=='texture3d'&&!p.pointer&&!p.reference&&!p.constant).map(p=>`var v_${p.name}: ${p.type} = cw_arg_${p.name};`)), ...indent(body), '}');
    }};
    emitHelpers();
    this.scopes = kernelScope; this.currentFunction = this.kernel;
    const main = [...this.kernel.params.filter(p=>p.pointer&&p.symbol.offsetCode).map(p=>`var ${p.symbol.offsetCode}: i32 = 0i;`),...this.body(this.kernel.body)];
    emitHelpers();this.scopes=kernelScope;this.currentFunction=this.kernel;
    for(const surface of surfaces)header.push(`@group(0) @binding(${surface.binding}) var cw_surface_${surface.name}: texture_storage_${surface.dimension.replace('-','_')}<${surface.format}, write>;`);
    for(const scalar of this.constantScalars)scalars.push({...scalar,offset:scalars.length*4});
    let uniformBytes=scalars.length*4;const textureScales=textures.filter(t=>t.format==='r32float'||t.coordinates==='2d'||t.dimension==='3d').map(t=>{const offset=uniformBytes;uniformBytes+=t.dimension==='3d'?16:12;return {name:t.name,offset,pointOffset:offset+(t.dimension==='3d'?12:8),...(t.dimension==='3d'?{dimension:'3d'}:{})};}),textureLengths=textures.filter(t=>t.coordinates==='linear').map(t=>{const offset=uniformBytes;uniformBytes+=4;return {name:t.name,offset};}),uniformSize=uniformBytes?Math.ceil(uniformBytes/16)*16:0;
    if(uniformSize){
      header.push('struct CWParams {',...scalars.map(s=>`  ${s.field||'p_'+s.name}: ${s.type},`));
      for(const scale of textureScales)header.push(`  cw_tex_${scale.name}_sx: f32,`,`  cw_tex_${scale.name}_sy: f32,`,...(scale.dimension==='3d'?[`  cw_tex_${scale.name}_sz: f32,`]:[]),`  cw_tex_${scale.name}_point: f32,`);
      for(const length of textureLengths)header.push(`  cw_tex_${length.name}_length: u32,`);
      for(let i=uniformBytes;i<uniformSize;i+=4)header.push(`  cw_pad_${i}: u32,`);
      header.push('}',`@group(0) @binding(${bindings.length+textures.length*2+surfaces.length}) var<uniform> cw_params: CWParams;`);
    }
    header.push(`const cw_block_size: vec3<u32> = vec3<u32>(${this.workgroupSize.map(x => `${x}u`).join(', ')});`);
    if(this.dynamicSharedBytes&&!this.dynamicSharedUsed)this.fail('sharedMemoryBytes was supplied but the kernel has no dynamic shared array.',this.kernel);
    // Runtime parameters preserve CUDA wraparound even when call arguments are literals;
    // WGSL rejects overflowing constant expressions in an inline multiply.
    if(this.integerIntrinsics.has('__mul24'))helperLines.unshift('fn cw_mul24(a: i32, b: i32) -> i32 { return ((a << 8u) >> 8u) * ((b << 8u) >> 8u); }');
    if(this.integerIntrinsics.has('__umul24'))helperLines.unshift('fn cw_umul24(a: u32, b: u32) -> u32 { return (a & 16777215u) * (b & 16777215u); }');
    // A float32 significand times a 16-bit integer fits exactly in 40 bits.
    // Integer limbs preserve the original double product before truncating to a byte.
    if(this.float64Used)helperLines.unshift(FLOAT64_WGSL);
    if(this.sizeMultiplyUsed)helperLines.unshift(`fn cw_size_multiply(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
  let a0 = a.x & 65535u; let a1 = a.x >> 16u;
  let b0 = b.x & 65535u; let b1 = b.x >> 16u;
  let p0 = a0 * b0;
  let p1 = a1 * b0 + (p0 >> 16u);
  let p2 = a0 * b1 + (p1 & 65535u);
  let low = (p2 << 16u) | (p0 & 65535u);
  let high = a1 * b1 + (p1 >> 16u) + (p2 >> 16u) + a.x * b.y + a.y * b.x;
  return vec2<u32>(low, high);
}`);
    if(this.extentUsed){header.unshift('alias cw_size64 = vec2<u32>;','struct cw_extent { width: vec2<u32>, height: vec2<u32>, depth: vec2<u32>, }');helperLines.unshift('fn cw_size_less(a: vec2<u32>, b: vec2<u32>) -> bool { return a.y < b.y || (a.y == b.y && a.x < b.x); }');}
    if(this.exactByteScaleUsed)helperLines.unshift(`fn cw_exact_byte_scale(value: f32, scale: u32) -> u32 {
  let bits = bitcast<u32>(value);
  let exponent = (bits >> 23u) & 255u;
  let mantissa = (bits & 8388607u) | select(0u, 8388608u, exponent != 0u);
  let lowProduct = (mantissa & 65535u) * scale;
  let upper = (mantissa >> 16u) * scale + (lowProduct >> 16u);
  let low = (lowProduct & 65535u) | (upper << 16u);
  let high = upper >> 16u;
  let shift = select(149i, 150i - i32(exponent), exponent != 0u);
  var result = 0u;
  if (exponent == 255u) { return 0u; }
  if (shift >= 64i) { result = 0u; }
  else if (shift >= 32i) { result = high >> u32(shift - 32i); }
  else if (shift > 0i) { result = (low >> u32(shift)) | (high << u32(32i - shift)); }
  else if (shift == 0i) { result = low; }
  else if (shift > -8i) { result = low << u32(-shift); }
  if ((bits & 2147483648u) != 0u) { result = 0u - result; }
  return result & 255u;
}`);
    if(this.packedAtomicUsed)helperLines.unshift('fn cw_store_byte(word: ptr<storage, atomic<u32>, read_write>, shift: u32, value: u32) { var old = atomicLoad(word); loop { let next = (old & ~(255u << shift)) | ((value & 255u) << shift); let result = atomicCompareExchangeWeak(word, old, next); if (result.exchanged) { return; } old = result.old_value; } }');
    for(const kind of this.linearFetchUsed||[]){const type=kind==='uint'?'u32':'f32';helperLines.unshift(`fn cw_fetch_${kind}(tex: texture_2d<${type}>, index: u32, length: u32) -> ${type} { if (index >= length) { return ${type}(0); } let width = textureDimensions(tex).x; return textureLoad(tex, vec2<i32>(i32(index % width), i32(index / width)), 0).r; }`);}
    if(this.float3DSamplingUsed)helperLines.unshift('fn cw_sample_float3d(tex: texture_3d<f32>, texSampler: sampler, coords: vec3<f32>, scale: vec3<f32>, pixelPoint: f32) -> f32 { if (pixelPoint > 0.0f) { let maximum = vec3<f32>(textureDimensions(tex)) - vec3<f32>(1.0f); let pixel = vec3<i32>(clamp(floor(coords), vec3<f32>(0.0f), maximum)); return textureLoad(tex, pixel, 0).r; } return textureSampleLevel(tex, texSampler, coords * scale, 0.0f).r; }');
    if(this.float4_3DSamplingUsed)helperLines.unshift('fn cw_sample_float4_3d(tex: texture_3d<f32>, texSampler: sampler, coords: vec3<f32>, scale: vec3<f32>, pixelPoint: f32) -> vec4<f32> { if (pixelPoint > 0.0f) { let maximum = vec3<f32>(textureDimensions(tex)) - vec3<f32>(1.0f); let pixel = vec3<i32>(clamp(floor(coords), vec3<f32>(0.0f), maximum)); return textureLoad(tex, pixel, 0); } return textureSampleLevel(tex, texSampler, coords * scale, 0.0f); }');
    if(this.float2DSamplingUsed)helperLines.unshift('fn cw_sample_float2d(tex: texture_2d<f32>, texSampler: sampler, coords: vec2<f32>, scale: vec2<f32>, pixelPoint: f32) -> f32 { if (pixelPoint > 0.0f) { let maximum = vec2<f32>(textureDimensions(tex)) - vec2<f32>(1.0f); let pixel = vec2<i32>(clamp(floor(coords), vec2<f32>(0.0f), maximum)); return textureLoad(tex, pixel, 0).r; } return textureSampleLevel(tex, texSampler, coords * scale, 0.0f).r; }');
    // Correct the approximate native WGSL quotient using its fused residual.
    // Nonfinite quotients retain ordinary WGSL behavior; this is not a software IEEE divider.
    if(this.compensatedDivisionUsed)helperLines.unshift('fn cw_divide_f32(a: f32, b: f32) -> f32 { let q = a / b; if ((bitcast<u32>(q) & 0x7f800000u) == 0x7f800000u || (bitcast<u32>(q) & 0x7fffffffu) == 0u || (bitcast<u32>(b) & 0x7f800000u) == 0x7f800000u) { return q; } let residual = fma(-q, b, a); return q + residual / b; }');
    if(this.roundAwayUsed)helperLines.unshift('fn cw_round_away(x: f32) -> f32 { let whole = trunc(x); let fraction = abs(x - whole); return select(whole, whole + select(-1.0f, 1.0f, x >= 0.0f), fraction >= 0.5f); }');
    if(this.byte2DSamplingUsed)helperLines.unshift('fn cw_sample_byte2d(tex: texture_2d<u32>, coords: vec2<f32>) -> u32 { let maximum = vec2<f32>(textureDimensions(tex)) - vec2<f32>(1.0f); let pixel = vec2<i32>(clamp(floor(coords), vec2<f32>(0.0f), maximum)); return textureLoad(tex, pixel, 0).x; }');
    if(this.rgba2DSamplingUsed)helperLines.unshift('fn cw_sample_rgba2d(tex: texture_2d<f32>, texSampler: sampler, coords: vec2<f32>, scale: vec2<f32>, pixelPoint: f32) -> vec4<f32> { if (pixelPoint > 0.0f) { let maximum = vec2<f32>(textureDimensions(tex)) - vec2<f32>(1.0f); let pixel = vec2<i32>(clamp(floor(coords), vec2<f32>(0.0f), maximum)); return textureLoad(tex, pixel, 0); } return textureSampleLevel(tex, texSampler, coords * scale, 0.0f); }');
    for (const s of this.shared) header.push(`var<workgroup> ${s.code}: ${s.atomic ? sharedAtomicType(s.type) : typeName(s.type)};`);
    const storageSize = this.shared.reduce((n, s) => n + Math.ceil(typeStride(s.type) / 16) * 16, 0);
    let usesPackedBytes=(this.ast.structs||[]).some(s=>s.fields.some(f=>['cw_uchar','cw_uchar4'].includes(f.type)));walk(this.ast,n=>{if(['cw_uchar','cw_uchar4'].includes(n.type)||['cw_uchar','cw_uchar4'].includes(n.result))usesPackedBytes=true;});let usesShort=false;walk(this.ast,n=>{if(['cw_short','cw_ushort'].includes(n.type)||['cw_short','cw_ushort'].includes(n.result))usesShort=true;});if(usesShort)header.unshift('alias cw_short = i32;','alias cw_ushort = u32;');if(usesPackedBytes)header.unshift('alias cw_uchar = u32;','alias cw_uchar4 = u32;');
    const wgsl = [...header, '', ...helperLines, '', `@compute @workgroup_size(${this.workgroupSize.join(', ')})`, 'fn main(', '  @builtin(local_invocation_id) cw_thread: vec3<u32>,', '  @builtin(workgroup_id) cw_block: vec3<u32>,', '  @builtin(num_workgroups) cw_grid: vec3<u32>', ') {', ...indent(main), '}', ''].join('\n');
    return {version: COMPILER_VERSION, name: this.kernel.name, entryPoint: 'main', wgsl, metadata: {workgroupSize: this.workgroupSize, bindings, scalars, uniformSize, uniformBinding: uniformSize ? bindings.length+textures.length*2+surfaces.length : null,...(textures.length?{textures}:{}),...(textureScales.length?{textureScales}:{}),...(textureLengths.length?{textureLengths}:{}),...(surfaces.length?{surfaces}:{}), workgroupStorageBytes: storageSize,...(this.dynamicSharedUsed?{dynamicSharedMemoryBytes:this.dynamicSharedBytes}:{}), barrier: this.usage.storageBarrier ? 'workgroup-and-storage' : 'workgroup'}, ast: this.ast, kernel: this.kernel};
  }
}
function resolveTraitTypes(fn, ast, parameter, argument) {
  const resolve=(type,node)=>{
    if(type?.kind!=='trait-type')return type;
    const key=type.argument===parameter?argument:type.argument;
    if(!key||!builtinType(key)||key==='void')throw new CompileError('Type-trait arguments must resolve to supported built-in value types.',node.token,ast.source);
    const trait=ast.typeTraits.find(t=>t.name===type.name),specialization=trait?.specializations.find(s=>s.argument===key);
    const member=(specialization?.members||trait?.members)?.find(m=>m.name===type.member);
    if(!member)throw new CompileError(`Unknown type-trait member '${type.name}::${type.member}'.`,node.token,ast.source);
    const name=!specialization&&member.type===trait.parameter?key:member.type,resolved=builtinType(name);
    if(!resolved||resolved==='void')throw new CompileError(`Type-trait member resolves to unsupported value type '${name}'.`,node.token,ast.source);
    return resolved;
  };
  fn.result=resolve(fn.result,fn);
  for(const p of fn.params)p.type=resolve(p.type,p);
  walk(fn.body,n=>{if(n.type)n.type=resolve(n.type,n);if(n.target)n.target=resolve(n.target,n);});
}
function instantiateHelperTemplates(ast, kernel) {
  const parameters=fn=>fn.templateParameters?.length?fn.templateParameters:[fn.templateParameter];
  const canonical=(fn,argument,node)=>{if(fn.templateKind==='int'){try{return String(integerExpression(argument));}catch(e){throw new CompileError(e.message,node.token,ast.source);}}const values=argument.split(',').map(x=>x.trim());if(values.length!==parameters(fn).length||values.some(v=>!v))throw new CompileError('Template argument count must match the helper type parameters.',node.token,ast.source);return values.join(',');};
  const definitions=new Map(),specializations=new Map(),instances=new Map(),visiting=new Set(),done=new Set(),clones=[];
  for(const fn of ast.functions){
    if(fn.specializationArgument!==undefined){
      const primary=definitions.get(fn.name);if(primary?.templateParameter)fn.specializationArgument=canonical(primary,fn.specializationArgument,fn);const key=fn.name+'<'+fn.specializationArgument+'>';
      if(!primary?.templateParameter||primary.qualifier!=='__device__')throw new CompileError('Declare a primary device helper template before its specialization.',fn.token,ast.source);
      if(specializations.has(key))throw new CompileError('Duplicate device helper specialization.',fn.token,ast.source);
      if(fn.params.length!==primary.params.length)throw new CompileError('Device helper specialization signature does not match its primary template.',fn.token,ast.source);
      specializations.set(key,fn);continue;
    }
    if(definitions.has(fn.name))throw new CompileError(`Duplicate function '${fn.name}'.`,fn.token,ast.source);
    definitions.set(fn.name,fn);
  }
  const fail=(message,node)=>{throw new CompileError(message,node.token,ast.source);};
  function process(fn){
    if(done.has(fn))return;
    if(visiting.has(fn))fail('Recursive helper calls are unsupported.',fn);
    visiting.add(fn);
    resolveTraitTypes(fn,ast);
    walk(fn.body,node=>{
      if(node.kind!=='call'||node.callee.kind!=='id')return;
      if(['tex3D','tex1D','tex1Dfetch','tex2D','tex2DLayered'].includes(node.callee.name))return;
      const callee=node.callee,definition=definitions.get(callee.name);let argument=callee.templateArgument;if(definition?.templateParameter&&argument!==undefined)argument=canonical(definition,argument,callee);
      if(!definition?.templateParameter){
        if(argument!==undefined)fail('Explicit template arguments require a templated device helper.',callee);
        if(definition?.qualifier==='__device__')process(definition);
        return;
      }
      if(definition.qualifier!=='__device__')fail('Device-side kernel launches are unsupported.',callee);
      if(argument===undefined)return; // Deduced after argument expression types are known to the emitter.
      const key=definition.name+'<'+argument+'>';
      let instance=instances.get(key);
      if(!instance){
        if(instances.size>=128)fail('At most 128 device helper template specializations are supported.',callee);
        const selected=specializations.get(key);
        instance=structuredClone(selected||definition);
        if(selected)for(let i=0;i<instance.params.length;i++)if(definition.params[i].defaultValue!==undefined)instance.params[i].defaultValue=structuredClone(definition.params[i].defaultValue);
        const names=parameters(definition),argumentsList=argument.split(','),replacements=new Map();
        for(let i=0;i<names.length;i++){const value=argumentsList[i],type=definition.templateKind==='type'?builtinType(value):null;if(definition.templateKind==='type'&&(!type||['void','texture3d','surface2d'].includes(type)))fail('Template type argument must be a supported built-in value type.',callee);if(!type&&(!Number.isSafeInteger(Number(value))||Number(value)<-2147483648||Number(value)>2147483647))fail('Template argument must be a signed 32-bit integer.',callee);replacements.set(names[i],{value,type});}
        const replaceType=type=>{if(typeof type==='string'&&type.startsWith('template:'))return replacements.get(type.slice(9))?.type??type;if(type?.kind==='trait-type')return {...type,argument:replacements.get(type.argument)?.value??type.argument};return type;};
        const substitute=fn=>{fn.result=replaceType(fn.result);for(const param of fn.params){if(replacements.has(param.name))fail('Template parameter shadowing is unsupported.',param);param.type=replaceType(param.type);}
          walk(fn.body,n=>{if(['decl','thread-block'].includes(n.kind)&&replacements.has(n.name))fail('Template parameter shadowing is unsupported.',n);if(n.type)n.type=replaceType(n.type);if(n.target)n.target=replaceType(n.target);if(n.kind==='call'&&n.callee.kind==='id'){const replacement=replacements.get(n.callee.name);if(replacement?.type)n.callee.name=replacement.value;}
            if(n.templateArgument!==undefined)n.templateArgument=n.templateArgument.replace(/[A-Za-z_]\w*/g,name=>{const r=replacements.get(name);return r?(r.type?r.value:'('+r.value+')'):name;});
            if(n.kind==='id'){const r=replacements.get(n.name);if(r&&!r.type){n.kind='literal';n.value=r.value;delete n.name;}}
          });resolveTraitTypes(fn,ast);
        };
        substitute(instance);
        if(selected){const expected={...definition,params:structuredClone(definition.params),body:{kind:'block',body:[]}};substitute(expected);
          if(instance.result!==expected.result||instance.params.some((p,i)=>p.type!==expected.params[i].type||p.pointer!==expected.params[i].pointer||p.reference!==expected.params[i].reference||(p.reference&&p.constant!==expected.params[i].constant)))fail('Device helper specialization signature does not match its primary template.',selected);
          walk(instance.body,n=>{if([n.type,n.target].some(t=>typeof t==='string'&&t.startsWith('unsupported:')))fail('Double-precision value types are unsupported in selected helper specializations.',n);});
        }
        let name='cw_specialized_'+instances.size;
        while(definitions.has(name))name+='_';
        instance.name=name;instance.templateParameter=null;instance.templateKind=null;delete instance.specializationArgument;
        instances.set(key,instance);definitions.set(name,instance);clones.push(instance);
      }
      process(instance);
      callee.name=instance.name;delete callee.templateArgument;
    });
    visiting.delete(fn);done.add(fn);
  }
  process(kernel);
  for(const fn of ast.functions)if(fn.qualifier==='__device__'&&!fn.templateParameter&&fn.specializationArgument===undefined)process(fn);
  ast.functions=ast.functions.filter(fn=>fn.qualifier!=='__device__'||(!fn.templateParameter&&fn.specializationArgument===undefined)).concat(clones);
  return {deduce(name,types,callee){
    const definition=definitions.get(name);
    if(!definition?.templateParameter)return null;
    if(definition.qualifier!=='__device__')fail('Device-side kernel launches are unsupported.',callee);
    if(definition.templateKind!=='type')fail('Integer helper templates require an explicit template argument.',callee);
    if(parameters(definition).length>1)fail('Multiple helper template types require explicit template arguments.',callee);
    if(types.length>definition.params.length||definition.params.slice(types.length).some(p=>p.defaultValue===undefined))fail(`Wrong number of arguments for '${name}'.`,callee);
    const matches=definition.params.flatMap((p,i)=>i<types.length&&p.type==='template:'+definition.templateParameter?[p.pointer?(isArray(types[i])?types[i].element:undefined):types[i]]:[]);
    if(!matches.length)fail('Cannot deduce helper template type from these parameters; supply an explicit argument.',callee);
    const type=matches[0];
    if(matches.some(t=>typeName(t)!==typeName(type)))fail('Conflicting deduced helper template argument types.',callee);
    const names=['float','int','uint','bool','uchar','uchar4',...['float','int','uint'].flatMap(p=>[2,3,4].map(n=>p+n))],argument=names.find(n=>builtinType(n)===type);
    if(!argument)fail('Deduced helper template argument must be a supported built-in value type.',callee);
    const call={kind:'call',token:callee.token,callee:{...callee,templateArgument:argument},args:[]};
    process({kind:'function',name:'deduction',token:callee.token,result:'void',params:[],body:{kind:'block',body:[call]}});
    for(const fn of clones)if(!ast.functions.includes(fn))ast.functions.push(fn);
    return definitions.get(call.callee.name);
  }};
}
export function compile(source, options = {},bufferUsage=null) {
  const ast = parse(source, options), kernels = ast.functions.filter(f => f.qualifier === '__global__');
  // Recognize same-allocation byte-address round trips before buffer usage analysis.
  // The emitter checks pointee type and constness before accepting the alias.
  walk(ast,n=>{
    if(n.kind!=='decl'||!n.pointer||n.init?.kind!=='pointer-cast')return;
    const outer=n.init,offset=outer.value,inner=offset?.left;
    if(outer.target!==n.type||offset?.kind!=='binary'||offset.op!=='+'||inner?.kind!=='pointer-cast'||inner.target!=='byte-address'||inner.value?.kind!=='id')return;
    if(outer.constant&&!n.constant)throw new CompileError('Cannot discard const through a pointer cast.',n.token,source);
    n.byteOffsetCast=true;n.init={...offset,left:inner.value};
  });
  walk(ast,n=>{if(n.kind==='unary'&&n.op==='*'){const base=n.value;delete n.value;delete n.op;Object.assign(n,{kind:'index',base,index:{kind:'literal',value:'0',token:n.token},dereference:true});}});
  const specialization=options.entry?.match(/^([A-Za-z_]\w*)<\s*(\d+|[A-Za-z_]\w*)\s*>$/),entry=specialization?specialization[1]:options.entry;if(specialization&&ast.typeAliases?.[specialization[2]])specialization[2]=['float','int','uint','bool','uchar','uchar4',...['float','int','uint'].flatMap(p=>[2,3,4].map(n=>p+n))].find(n=>builtinType(n)===ast.typeAliases[specialization[2]]);
  const kernel = entry ? kernels.find(k => k.name === entry) : kernels.length === 1 ? kernels[0] : null;
  if (!kernel) throw new CompileError(options.entry ? `Kernel '${options.entry}' was not found.` : 'Multiple kernels found; specify options.entry.');
  if(!!kernel.templateParameter!==!!specialization)throw new CompileError(kernel.templateParameter?'Specify a template entry, for example '+kernel.name+(kernel.templateKind==='type'?'<float>.':'<16>.'):'This kernel does not have a template parameter.',kernel.token,source);
  if(specialization&&kernel.templateKind==='type'){
    const type=builtinType(specialization[2]),name=kernel.templateParameter,placeholder='template:'+name;
    if(!type||type==='void')throw new CompileError('Template type argument must be a supported built-in value type.',kernel.token,source);
    for(const p of kernel.params){if(p.name===name)throw new CompileError('Template parameter shadowing is unsupported.',p.token,source);if(p.type===placeholder)p.type=type;}
    if(kernel.result===placeholder)kernel.result=type;
    walk(kernel.body,n=>{if(['decl','thread-block'].includes(n.kind)&&n.name===name)throw new CompileError('Template parameter shadowing is unsupported.',n.token,source);if(n.type===placeholder)n.type=type;if(n.target===placeholder)n.target=type;if(n.kind==='call'&&n.callee.kind==='id'&&n.callee.name===name)n.callee.name=specialization[2];});
  }else if(specialization){const value=Number(specialization[2]),name=kernel.templateParameter;
    if(!Number.isSafeInteger(value)||value>2147483647)throw new CompileError('Template argument must be a nonnegative 32-bit signed integer.',kernel.token,source);
    for(const p of kernel.params)if(p.name===name)throw new CompileError('Template parameter shadowing is unsupported.',p.token,source);
    walk(kernel.body,n=>{if(['decl','thread-block'].includes(n.kind)&&n.name===name)throw new CompileError('Template parameter shadowing is unsupported.',n.token,source);if(n.kind==='id'&&n.name===name){n.kind='literal';n.value=String(value);delete n.name;}});
  }
  if(specialization)walk(kernel.body,n=>{if(n.templateArgument!==undefined){n.templateArgument=n.templateArgument.replace(/[A-Za-z_]\w*/g,name=>name===kernel.templateParameter?(kernel.templateKind==='int'?'('+specialization[2]+')':specialization[2]):name);}});
  resolveTraitTypes(kernel,ast,kernel.templateParameter,specialization?.[2]);
  const scalarConstraints=uniformBlockGuards(kernel,options,walk,message=>{throw new CompileError(message,kernel.token,source);});
  const overloadGroups=new Map();for(const f of ast.functions)if(f.specializationArgument===undefined){const group=overloadGroups.get(f.name)||[];group.push(f);overloadGroups.set(f.name,group);}let overloadIndex=0;const occupied=new Set(ast.functions.map(f=>f.name));for(const [name,group]of overloadGroups)if(group.length>1){if(group.some(f=>f.qualifier!=='__device__'||f.templateParameter))throw new CompileError('Overloads support non-template device helpers only.',group[0].token,source);const signatures=new Set();for(const f of group){const signature=JSON.stringify(f.params.map(p=>[p.type,p.pointer,p.reference,(p.pointer||p.reference)&&p.constant]));if(signatures.has(signature))throw new CompileError('Duplicate function signature '+name,f.token,source);signatures.add(signature);let unique='cw_overload_'+overloadIndex+++'_'+name;while(occupied.has(unique))unique+='_';occupied.add(unique);f.overloadName=name;f.name=unique;}}
  const templates=instantiateHelperTemplates(ast,kernel);
  const emitter=new Emitter(ast,kernel,options,templates,bufferUsage),result=emitter.emit();
  if(scalarConstraints.length||emitter.pointerConstraints.length)result.metadata.scalarConstraints=[...scalarConstraints,...emitter.pointerConstraints];
  const changed=['reads','writes','atomic'].some(k=>[...emitter.usage[k]].some(name=>!emitter.initialBufferUsage[k].has(name)));
  if(changed){if(bufferUsage)throw new CompileError('Helper buffer access analysis did not converge.');return compile(source,options,Object.fromEntries(['reads','writes','atomic'].map(k=>[k,[...emitter.usage[k]]])));}if(specialization)result.metadata.templateArguments={[kernel.templateParameter]:kernel.templateKind==='type'?specialization[2]:Number(specialization[2])};return result;
}
export function serializableArtifact(compiled) {
  return {version: compiled.version, name: compiled.name, entryPoint: compiled.entryPoint, wgsl: compiled.wgsl, metadata: compiled.metadata};
}
