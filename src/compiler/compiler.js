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
const numeric = t => ['f32', 'i32', 'u32'].includes(t);
const indent = lines => lines.map(l => `  ${l}`);
const rootName = n => n?.kind === 'id' ? n.name : ['index', 'member'].includes(n?.kind) ? rootName(n.base) : null;
export function walk(node, visit) {
  if (!node || typeof node !== 'object') return;
  if (node.kind) visit(node);
  for (const [key, val] of Object.entries(node)) {
    if (['token', 'symbol', 'type', 'source', 'resolved','pointerBaseSymbol'].includes(key)) continue;
    if (Array.isArray(val)) val.forEach(n => walk(n, visit));
    else if (val && typeof val === 'object') walk(val, visit);
  }
}
function constantValue(n) {
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
    if(n.kind==='decl'){scan(n.init);const base=n.init?.kind==='id'?n.init:n.init?.kind==='binary'&&n.init.op==='+'?n.init.left:null;aliases.set(n.name,n.pointer&&base?.kind==='id'?resolve(base.name):null);return;}
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
    if (n.kind === 'member') { scan(n.base, mode); return; }
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
    this.functions = new Map(); this.shared = []; this.templates=templates;this.helperCalls=new Map();this.globalSymbols=new Map();this.constantScalars=[];
    for (const f of ast.functions) {
      if (this.functions.has(f.name)) this.fail(`Duplicate function '${f.name}'.`, f);
      this.functions.set(f.name, f);
    }
    this.helpers = ast.functions.filter(f => f.qualifier === '__device__'&&!f.params.some(p=>p.pointer));
    this.pointerHelpers=new Map();this.bufferSymbols=new Map();
    this.usage = analyse([kernel], kernel.params);this.usage.atomic=this.usage.atomicBindings;
    for(const kind of ['reads','writes','atomic'])for(const name of bufferUsage?.[kind]||[])this.usage[kind].add(name);
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
      if(!numeric(global.type))this.fail('Referenced constant globals require float, int or unsigned int scalar values.',n);
      if(!this.globalSymbols.has(name)){
        let value=0;
        if(global.init){const literal=global.init.kind==='unary'&&['+','-'].includes(global.init.op)?global.init.value:global.init;if(literal.kind!=='literal')this.fail('Constant global initializers must be numeric literals with an optional sign.',global);value=constantValue(global.init);}
        if(!Number.isFinite(value)||(global.type==='f32'&&!Number.isFinite(Math.fround(value)))||(global.type==='u32'&&(!Number.isInteger(value)||value<0||value>4294967295))||(global.type==='i32'&&(!Number.isInteger(value)||value<-2147483648||value>2147483647)))this.fail('Constant global initializer is outside its supported scalar range.',global);
        const scalarName='constant.'+name,symbol={name:scalarName,type:global.type,code:'cw_params.c_'+name,constant:true,atomic:false,kind:'constant-global'};
        this.globalSymbols.set(name,symbol);global.symbol=symbol;
        this.constantScalars.push({name:scalarName,type:global.type,origin:'constant',field:'c_'+name,defaultValue:global.type==='f32'?Math.fround(value):value});
      }
      return this.globalSymbols.get(name);
    }
    this.fail(`Unknown identifier '${name}'.`, n);
  }
  add(name, symbol, n, global = false) { const scope = global ? this.scopes[0] : this.scopes.at(-1); if (scope.has(name)) this.fail(`Duplicate identifier '${name}'.`, n); scope.set(name, symbol); return symbol; }
  common(a, b, n) {
    if (typeName(a) === typeName(b) && !isArray(a) && a !== 'void') return a;
    // C++ promotes a bool to int before the usual scalar arithmetic conversions.
    if(a==='bool'&&numeric(b))a='i32';
    if(b==='bool'&&numeric(a))b='i32';
    if (numeric(a) && numeric(b)) return a === 'f32' || b === 'f32' ? 'f32' : a === 'u32' || b === 'u32' ? 'u32' : 'i32';
    this.fail(`Incompatible operand types: ${typeName(a)} and ${typeName(b)}. Use explicit scalar/vector components.`, n);
  }
  convert(code, from, to, n) {
    if(to==='bool'&&isArray(from)&&from.length===null&&n){return 'true';} // All runtime storage bindings are required and non-null.
    if (typeName(from) === typeName(to) && !isArray(to)) return code;
    if (numeric(from) && numeric(to)) return `${to}(${code})`;
    if (to === 'bool' && numeric(from)) return `(${code} != ${from === 'f32' ? '0.0f' : from === 'u32' ? '0u' : '0i'})`;
    if (from === 'bool' && numeric(to)) return `select(${to}(0), ${to}(1), ${code})`;
    this.fail(`Cannot convert ${typeName(from)} to ${typeName(to)}.`, n);
  }
  result(n, type, code, pre = [], extra = {}) { n.type = type; return {type, code, pre, ...extra}; }
  expr(n, raw = false) {
    if (!n) this.fail('Missing expression.', this.kernel);
    switch (n.kind) {
      case 'initializer': {
        const width=vectorLength(n.target),element=vectorElement(n.target);
        if(!width||n.items.length>width)this.fail('Vector initializers require at most one scalar per component.',n);
        const values=n.items.map(item=>this.expr(item));
        if(values.some(v=>v.type!==element))this.fail('Vector initializer components must match the element type; use explicit casts for conversions.',n);
        const codes=values.map(v=>v.code);while(codes.length<width)codes.push(`${element}(0)`);
        return this.result(n,n.target,`${n.target}(${codes.join(', ')})`,values.flatMap(v=>v.pre));
      }
      case 'literal': {
        const isHex = /^0[xX]/.test(n.value), isFloat = !isHex && /[fF]$/.test(n.value), isUnsigned = /[uU]$/.test(n.value);
        if (!isFloat && /[.eE]/.test(n.value) && !/^0[xX]/.test(n.value)) this.fail('Double-precision literals are unsupported. Use an f suffix, for example 0.5f.', n);
        let value = Number(n.value.replace(isHex ? /[uU]$/ : /[fFuU]$/, ''));
        if (!Number.isFinite(value)) this.fail('Invalid numeric literal.', n);
        let type = isFloat ? 'f32' : isUnsigned || (/^0[xX]/.test(n.value) && value > 2147483647) ? 'u32' : 'i32';
        if (!isFloat && (value < 0 || value > (type === 'u32' ? 4294967295 : 2147483647))) this.fail('Integer literal is outside the supported 32-bit range.', n);
        if (isFloat && !Number.isFinite(Math.fround(value))) this.fail('Floating literal overflows f32.', n);
        n.numericValue = value;
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
        if (!isArray(base.type) || !['i32', 'u32'].includes(index.type)) this.fail('Indexing requires an array and a 32-bit integer index.', n);
        const offset=base.rootSymbol?.offsetCode,indexCode=offset?`(${offset} + ${this.convert(index.code,index.type,'i32',n)})`:index.code;
        const code = `${base.code}[${indexCode}]`, type = base.type.element;
        const atomic = base.atomicRoot && !isArray(type);
        return this.result(n, type, atomic && !raw ? `atomicLoad(&${code})` : code, [...base.pre, ...index.pre], {rootSymbol: base.rootSymbol, atomicRoot: base.atomicRoot, atomic});
      }
      case 'member': {
        if (n.base.kind === 'id' && ['threadIdx', 'blockIdx', 'blockDim', 'gridDim'].includes(n.base.name)) {
          if (!['x', 'y', 'z'].includes(n.member)) this.fail('CUDA dimensions have x, y and z components only.', n);
          const code = {threadIdx: 'cw_thread', blockIdx: 'cw_block', blockDim: 'cw_block_size', gridDim: 'cw_grid'}[n.base.name];
          return this.result(n, 'u32', `${code}.${n.member}`);
        }
        const base = this.expr(n.base, raw), size = vectorLength(base.type);
        if (!size || n.member.length !== 1 || 'xyzw'.indexOf(n.member) < 0 || 'xyzw'.indexOf(n.member) >= size) this.fail('Only valid single vector components (.x/.y/.z/.w) are supported.', n);
        return this.result(n, vectorElement(base.type), `${base.code}.${n.member}`, base.pre, {rootSymbol: base.rootSymbol});
      }
      case 'cast': { const value = this.expr(n.value); return this.result(n, n.target, this.convert(value.code, value.type, n.target, n), value.pre); }
      case 'unary': {
        if (['++', '--'].includes(n.op)) this.fail('Increment/decrement are supported as statements and for-loop updates, not inside expressions.', n);
        if (n.op === '&' || n.op === '*') this.fail('Pointers are supported only as kernel buffer parameters and &buffer[index] atomic targets.', n);
        const value = this.expr(n.value);
        if (n.op === '!') return this.result(n, 'bool', `(!${this.convert(value.code, value.type, 'bool', n)})`, value.pre);
        if (!numeric(value.type) || (n.op === '~' && value.type === 'f32')) this.fail('Invalid unary operator/type.', n);
        if (n.op === '-' && value.type === 'u32') return this.result(n, 'u32', `(0u - ${value.code})`, value.pre);
        return this.result(n, value.type, n.op === '+' ? value.code : `(${n.op}${value.code})`, value.pre);
      }
      case 'binary': {
        const a = this.expr(n.left), b = this.expr(n.right);
        if (['&&', '||'].includes(n.op)) {
          const ac = this.convert(a.code, a.type, 'bool', n), bc = this.convert(b.code, b.type, 'bool', n);
          if (!b.pre.length) return this.result(n, 'bool', `(${ac} ${n.op} ${bc})`, [...a.pre]);
          const tmp = `cw_tmp_${this.temp++}`;
          const pre = [...a.pre, `var ${tmp}: bool = ${ac};`, `if (${n.op === '&&' ? tmp : `!${tmp}`}) {`, ...indent([...b.pre, `${tmp} = ${bc};`]), '}'];
          return this.result(n, 'bool', tmp, pre);
        }
        let common = ['<<', '>>'].includes(n.op) ? a.type : this.common(a.type, b.type, n);
        if (vectorLength(common)) this.fail('CUDA vector arithmetic requires explicit components; operator overloads are outside this subset.', n);
        if (['&', '|', '^', '<<', '>>', '%'].includes(n.op) && !['i32', 'u32'].includes(common)) this.fail('Bitwise, shift and remainder operators require integers.', n);
        if (common === 'bool' && !['==', '!='].includes(n.op)) this.fail('Boolean values support only logical/equality operators.', n);
        const ac = this.convert(a.code, a.type, common, n), bc = this.convert(b.code, b.type, ['<<', '>>'].includes(n.op) ? 'u32' : common, n);
        const out = ['==', '!=', '<', '>', '<=', '>='].includes(n.op) ? 'bool' : common;
        n.operandType = common;
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
  argument(n){
    const address=n.kind==='unary'&&n.op==='&'&&n.value.kind==='index'?n.value:null;
    const base=address?address.base:n.kind==='binary'&&n.op==='+'?n.left:n.kind==='id'?n:null;
    if(base?.kind==='id'&&!['true','false'].includes(base.name)){
      const symbol=this.lookup(base.name,base);
      if(symbol.kind==='thread-block'&&n.kind==='id'){n.symbol=symbol;return {type:'thread-block',code:'',pre:[],rootSymbol:symbol};}
      if(['buffer','buffer-alias'].includes(symbol.kind)){
        const node=address?address.index:n.kind==='binary'?n.right:null,offset=node?this.expr(node):{type:'i32',code:'0i',pre:[]};
        if(!['i32','u32'].includes(offset.type))this.fail('Helper buffer offsets must be 32-bit integers.',n);
        n.pointerBaseSymbol=symbol;n.pointerOffset=node;
        const pointerCode=symbol.offsetCode?`(${symbol.offsetCode} + ${this.convert(offset.code,offset.type,'i32',n)})`:this.convert(offset.code,offset.type,'i32',n);
        return this.result(n,symbol.type,symbol.code,offset.pre,{rootSymbol:symbol,pointerCode});
      }
    }
    return this.expr(n);
  }
  bindPointerHelper(helper,args,n){
    const uses=analyse([helper],helper.params),roots=[];
    for(const [i,p]of helper.params.entries())if(p.pointer){
      const a=args[i],symbol=a?.rootSymbol,root=symbol?.rootBufferName;
      if(!root||!isArray(a.type)||a.type.element!==p.type)this.fail('Helper pointers require a same-type storage buffer or buffer offset.',n.args[i]);
      if(symbol.constant&&!p.constant)this.fail('Cannot discard const through a helper pointer argument.',n.args[i]);
      roots.push([i,root,!!symbol.constant]);
      if(uses.reads.has(p.name))this.usage.reads.add(root);
      if(uses.writes.has(p.name))this.usage.writes.add(root);
      if(uses.atomicBindings.has(p.name)){this.usage.atomic.add(root);this.bufferSymbols.get(root).atomic=true;}
    }
    this.usage.storageBarrier=[...this.usage.writes].some(x=>this.usage.reads.has(x));
    const key=JSON.stringify([helper.name,roots]);
    if(!this.pointerHelpers.has(key)){
      if(this.pointerHelpers.size>=128)this.fail('At most 128 helper buffer specializations are supported.',n);
      const clone=structuredClone(helper);let name='cw_buffer_helper_'+this.pointerHelpers.size;while(this.functions.has(name))name+='_';
      clone.name=name;clone.pointerOrigin=helper.name;
      for(const [i,root,constant]of roots){clone.params[i].boundBuffer=root;clone.params[i].boundConstant=constant;}
      this.pointerHelpers.set(key,clone);this.functions.set(name,clone);this.helpers.push(clone);this.ast.functions.push(clone);
    }
    return this.pointerHelpers.get(key);
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
    const casts = {float: 'f32', int: 'i32', uint: 'u32', bool: 'bool'};
    const args = n.args.map(a => this.argument(a)), pre = args.flatMap(a => a.pre);
    if(name==='__mul24'||name==='__umul24'){
      if(args.length!==2||args.some(a=>!['i32','u32'].includes(a.type)))this.fail(`${name} requires two 32-bit integer arguments.`,n);
      const signed=name==='__mul24',type=signed?'i32':'u32';
      this.integerIntrinsics.add(name);
      return this.result(n,type,`cw_${signed?'mul24':'umul24'}(${args.map(a=>this.convert(a.code,a.type,type,n)).join(', ')})`,pre);
    }
    if (casts[name]) { if (args.length !== 1) this.fail('Scalar casts require one argument.', n); return this.result(n, casts[name], this.convert(args[0].code, args[0].type, casts[name], n), pre); }
    if (/^make_(float|uint|int)[234]$/.test(name)) {
      const count = Number(name.at(-1)); if (args.length !== count) this.fail(`${name} needs ${count} arguments.`, n);
      const element=name.startsWith('make_uint')?'u32':name.startsWith('make_int')?'i32':'f32';
      return this.result(n, `vec${count}<${element}>`, `vec${count}<${element}>(${args.map(a => this.convert(a.code, a.type, element, n)).join(', ')})`, pre);
    }
    if(name==='__fdividef'){if(args.length!==2)this.fail('__fdividef requires two arguments.',n);return this.result(n,'f32',`(${args.map(a=>this.convert(a.code,a.type,'f32',n)).join(' / ')})`,pre);}
    if(name==='sqrt'){if(args.length!==1||args[0].type!=='f32')this.fail('sqrt supports the single float overload only; double/integer overloads are unavailable.',n);return this.result(n,'f32',`sqrt(${args[0].code})`,pre);}
    const unary = {sinf: 'sin', cosf: 'cos', tanf: 'tan', sqrtf: 'sqrt', rsqrtf: 'inverseSqrt', expf: 'exp', __expf:'exp', exp2f: 'exp2', logf: 'log', __logf:'log', log2f: 'log2', fabsf: 'abs', floorf: 'floor', ceilf: 'ceil', truncf: 'trunc'};
    const binary = {fminf: 'min', fmaxf: 'max', powf: 'pow', atan2f: 'atan2'};
    if (unary[name] || binary[name] || name === 'fmaf') {
      const count = unary[name] ? 1 : binary[name] ? 2 : 3;
      if (args.length !== count) this.fail(`${name} requires ${count} arguments.`, n);
      const target = unary[name] || binary[name] || 'fma';
      return this.result(n, 'f32', `${target}(${args.map(a => this.convert(a.code, a.type, 'f32', n)).join(', ')})`, pre);
    }
    if (['min', 'max'].includes(name)) {
      if (args.length !== 2) this.fail(`${name} needs two arguments.`, n);
      const type = this.common(args[0].type, args[1].type, n);
      if (!numeric(type)) this.fail('min/max accept scalars.', n);
      return this.result(n, type, `${name}(${args.map(a => this.convert(a.code, a.type, type, n)).join(', ')})`, pre);
    }
    let helper = this.functions.get(name);
    if(!helper&&this.templates){
      helper=this.templates.deduce(name,args.map(a=>a.type),n.callee);
      if(helper)for(const fn of this.ast.functions)if(fn.qualifier==='__device__'&&!this.functions.has(fn.name)){this.functions.set(fn.name,fn);if(!fn.params.some(p=>p.pointer))this.helpers.push(fn);}
    }
    if (!helper || helper.qualifier !== '__device__') this.fail(`Unsupported function '${name}'. CUDA host APIs, warp intrinsics, dynamic launches and libraries are not available.`, n);
    if (args.length !== helper.params.length) this.fail(`Wrong number of arguments for '${name}'.`, n);
    if(helper.params.some(p=>p.pointer))helper=this.bindPointerHelper(helper,args,n);
    const caller=this.currentFunction.name,edges=this.helperCalls.get(caller)||new Set();edges.add(helper.name);this.helperCalls.set(caller,edges);
    const reaches=(from,target,seen=new Set())=>{if(from===target)return true;if(seen.has(from))return false;seen.add(from);return [...(this.helperCalls.get(from)||[])].some(next=>reaches(next,target,seen));};
    if(reaches(helper.name,caller))this.fail('Recursive helper calls are unsupported.',n);
    n.callee.name=helper.name;n.callName=helper.name;
    const references=new Set();n.referenceArgs=helper.params.map(p=>!!p.reference);n.groupArgs=helper.params.map(p=>p.type==='thread-block');n.pointerArgs=helper.params.map(p=>!!p.pointer);
    const codes=args.map((a,i)=>{const p=helper.params[i];if(p.pointer)return a.pointerCode;if(p.type==='thread-block'){if(a.type!=='thread-block'||a.rootSymbol?.kind!=='thread-block')this.fail('thread_block arguments require a block handle.',n.args[i]);return null;}if(!p.reference)return this.convert(a.code,a.type,p.type,n);
      const node=n.args[i],s=a.rootSymbol;if(node.kind!=='id'||!s||!['local','reference'].includes(s.kind)||s.constant||isArray(a.type)||!numeric(a.type)||a.type!==p.type)this.fail('Reference arguments require a mutable named local scalar of the exact type.',node);
      if(references.has(s))this.fail('Aliased reference arguments are unsupported.',node);references.add(s);return s.kind==='reference'?s.pointerCode:`&${s.code}`;
    });
    return this.result(n, helper.result, `f_${helper.name}(${[...codes.filter(c=>c!==null),'cw_thread','cw_block','cw_grid'].join(', ')})`, pre);
  }
  writable(target, n) {
    const s = target.rootSymbol;
    if (!s || !['id', 'index', 'member'].includes(n.kind) || isArray(target.type)) this.fail('Assignment requires a scalar/vector variable or array element.', n);
    if (s.constant) this.fail(`Cannot write through const '${s.name}'.`, n);
    if (s.kind === 'uniform') this.fail('Scalar kernel parameters are read-only in this subset. Copy the parameter to a local variable first.', n);
  }
  effect(n) {
    if(n.kind==='sequence')return n.expressions.flatMap(e=>this.effect(e));
    if (n.kind === 'assign') {
      const target = this.expr(n.left, true); this.writable(target, n.left); const value = this.expr(n.right);
      let code = this.convert(value.code, value.type, target.type, n);
      if (n.op !== '=') {
        const op = n.op.slice(0, -1); if (target.atomic) this.fail('Use explicit atomicAdd/Min/Max/Exch rather than compound assignments to atomic arrays.', n);
        const type = ['<<', '>>'].includes(op) ? target.type : this.common(target.type, value.type, n), rhsType = ['<<', '>>'].includes(op) ? 'u32' : type;
        if (['<<', '>>', '%', '&', '|', '^'].includes(op) && !['i32', 'u32'].includes(type)) this.fail('Integer operator requires integer operands.', n);
        code = this.convert(`(${this.convert(target.code, target.type, type, n)} ${op} ${this.convert(value.code, value.type, rhsType, n)})`, type, target.type, n);
        n.operandType = type;
      }
      n.type = target.type;
      return [...target.pre, ...value.pre, target.atomic ? `atomicStore(&${target.code}, ${code});` : `${target.code} = ${code};`];
    }
    if (n.kind === 'unary' && ['++', '--'].includes(n.op)) {
      const target = this.expr(n.value, true); this.writable(target, n.value); if (target.atomic || !numeric(target.type)) this.fail('Increment/decrement require a non-atomic scalar.', n);
      n.type = target.type; return [...target.pre, `${target.code} ${n.op === '++' ? '+=' : '-='} ${target.type}(1);`];
    }
    const value = this.expr(n);
    if (n.kind !== 'call') this.fail('Only assignments, increments and function calls may stand alone as statements.', n);
    if (n.callName === '__syncthreads') return [...value.pre, 'workgroupBarrier();', ...(this.usage.storageBarrier ? ['storageBarrier();'] : [])];
    return [...value.pre, value.type === 'void' ? `${value.code};` : `_ = ${value.code};`];
  }
  declare(n) {
    if(n.init?.kind==='shared-conversion'){
      if(!n.pointer||n.reference||n.shared||n.external||n.dimensions.length||n.type!==n.init.target)this.fail('Shared conversion requires a matching local pointer declaration.',n);
      if(!n.init.conversions.includes(false)&&!n.constant)this.fail('Cannot discard const from shared wrapper conversion.',n);
      // A stateless conversion exposes the dispatch's dynamic shared allocation.
      n.pointer=false;n.shared=true;n.external=true;n.dimensions=[null];n.init=null;
      if(n.constant)this.fail('Const shared wrapper views are not yet supported.',n);
    }
    if(n.external&&(!n.shared||n.pointer||n.reference||n.constant||n.init||n.dimensions.length!==1||n.dimensions[0]!==null))this.fail('extern is supported only as extern __shared__ T name[].',n);
    if(n.reference)this.fail('References are supported only as helper parameters, not local declarations.',n);
    if(n.pointer){
      const baseNode=n.init?.kind==='id'?n.init:n.init?.kind==='binary'&&n.init.op==='+'?n.init.left:null,offsetNode=n.init?.kind==='binary'?n.init.right:null;
      if(n.shared||n.dimensions.length||baseNode?.kind!=='id')this.fail('Local pointers require a buffer alias with an optional integer offset.',n);
      const base=this.lookup(baseNode.name,baseNode);if(!['buffer','buffer-alias'].includes(base.kind)||base.type.element!==n.type)this.fail('Local pointers can alias only same-type storage buffers.',n);
      if(base.constant&&!n.constant)this.fail('Cannot discard const through a buffer alias.',n);
      const offset=offsetNode?this.expr(offsetNode):{type:'i32',code:'0i',pre:[]};if(!['i32','u32'].includes(offset.type))this.fail('Buffer alias offsets must be 32-bit integers.',n);
      const offsetCode=`cw_offset_${this.temp++}`,symbol={...base,name:n.name,constant:n.constant||base.constant,kind:'buffer-alias',offsetCode};this.add(n.name,symbol,n);n.symbol=symbol;n.aliasBase=base;n.aliasOffset=offsetNode;
      return [...offset.pre,`let ${offsetCode} = ${base.offsetCode?base.offsetCode+' + ':''}${this.convert(offset.code,offset.type,'i32',n)};`];
    }
    if (n.type === 'void') this.fail('Variables cannot have void type.', n);
    let type = n.type;const sharedOwner=(this.currentFunction.pointerOrigin||this.currentFunction.name)+':'+n.token.offset+':'+n.name;
    const dims = n.dimensions.map(d => {if(d===null){if(!n.external||!n.shared)this.fail('Unsized arrays require extern __shared__.',n);if(this.dynamicSharedUsed&&this.dynamicSharedOwner!==sharedOwner)this.fail('Only one dynamic shared array is supported; CUDA declarations alias the same allocation.',n);const stride=typeStride(n.type);if(n.type==='bool'||vectorLength(n.type)===3)this.fail('Dynamic shared arrays require 32-bit scalars or two/four-component vectors.',n);if(!this.dynamicSharedBytes||this.dynamicSharedBytes%stride)this.fail('Set sharedMemoryBytes to a positive multiple of the dynamic shared element size.',n);this.dynamicSharedUsed=true;this.dynamicSharedOwner=sharedOwner;return this.dynamicSharedBytes/stride;}const value = constantValue(d); if (!Number.isSafeInteger(value) || value < 1 || value > 65536) this.fail('Invalid fixed array dimension (1..65536).', n); return value; });
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
    const bindings = [], scalars = [], header = [`// CUDA WebShader ${COMPILER_VERSION}. Generated from kernel ${this.kernel.name}.`];
    const sharedAtomicType = t => isArray(t) ? `array<${sharedAtomicType(t.element)}, ${t.length}>` : `atomic<${t}>`;
    for (const p of this.kernel.params) {
      if (p.shared || p.reference || p.external || p.type === 'void') this.fail('Invalid kernel parameter type.', p);
      if (p.pointer) {
        if (p.type === 'bool' || vectorLength(p.type)===3) this.fail('bool* and three-component vector pointers have incompatible CUDA/WGSL layouts. Use 32-bit scalars or two/four-component vectors.', p);
        const atomic = this.usage.atomic.has(p.name), readOnly = p.constant || !this.usage.writes.has(p.name);
        if (p.constant && this.usage.writes.has(p.name)) this.fail(`Cannot write through const buffer '${p.name}'.`, p);
        if (atomic && !['i32', 'u32'].includes(p.type)) this.fail('Only 32-bit integer atomics are supported.', p);
        const binding = bindings.length;
        bindings.push({name: p.name, elementType: p.type, stride: typeStride(p.type), binding, readOnly, atomic});
        const symbol = {name: p.name, rootBufferName:p.name, type: arrayOf(p.type), code: `b_${p.name}`, constant: p.constant, atomic, kind: 'buffer'};
        this.add(p.name, symbol, p, true); p.symbol = symbol;this.bufferSymbols.set(p.name,symbol);
        header.push(`@group(0) @binding(${binding}) var<storage, ${readOnly ? 'read' : 'read_write'}> b_${p.name}: array<${atomic ? `atomic<${p.type}>` : p.type}>;`);
      } else {
        if (!numeric(p.type)) this.fail('Scalar kernel parameters must be float, int or unsigned int. Put vectors in buffers.', p);
        scalars.push({name: p.name, type: p.type, offset: scalars.length * 4});
        const symbol = {name: p.name, type: p.type, code: `cw_params.p_${p.name}`, constant: false, atomic: false, kind: 'uniform'};
        this.add(p.name, symbol, p, true); p.symbol = symbol;
      }
    }
    const helperLines = [];
    // Helpers cannot capture kernel arguments; explicit scalar arguments only.
    const kernelScope = this.scopes;
    let emittedHelpers=0;
    const emitHelpers=()=>{while(emittedHelpers<this.helpers.length){const helper=this.helpers[emittedHelpers++];
      this.scopes = [new Map()]; this.currentFunction = helper;
      for (const p of helper.params) {
        if (p.shared || p.external || p.type === 'void') this.fail('Invalid helper parameter.', p);
        if(p.pointer){const base=this.bufferSymbols.get(p.boundBuffer);if(!base)this.fail('Helper buffer pointer was not specialized.',p);p.symbol=this.add(p.name,{...base,name:p.name,kind:'buffer-alias',constant:p.constant||p.boundConstant,offsetCode:'v_'+p.name+'_offset'},p);continue;}
        if(p.type==='thread-block'){if(p.reference)this.fail('thread_block helper parameters must be passed by value.',p);p.symbol=this.add(p.name,{name:p.name,type:p.type,kind:'thread-block',constant:true},p);continue;}
        if(p.reference&&!numeric(p.type))this.fail('Helper references require a 32-bit numeric scalar.',p);
        p.symbol = this.add(p.name, {name: p.name, type: p.type, code: p.reference?`(*v_${p.name})`:`v_${p.name}`,pointerCode:p.reference?`v_${p.name}`:undefined, constant:p.constant, atomic: false, kind: p.reference?'reference':'local'}, p);
      }
      const body = this.body(helper.body);
      helperLines.push(`fn f_${helper.name}(${[...helper.params.filter(p=>p.type!=='thread-block').map(p => p.pointer?`v_${p.name}_offset: i32`:`${p.reference||p.constant?'v_':'cw_arg_'}${p.name}: ${p.reference?`ptr<function, ${p.type}>`:p.type}`),'cw_thread: vec3<u32>','cw_block: vec3<u32>','cw_grid: vec3<u32>'].join(', ')})${helper.result === 'void' ? '' : ` -> ${helper.result}`} {`,...indent(helper.params.filter(p=>p.type!=='thread-block'&&!p.pointer&&!p.reference&&!p.constant).map(p=>`var v_${p.name}: ${p.type} = cw_arg_${p.name};`)), ...indent(body), '}');
    }};
    emitHelpers();
    this.scopes = kernelScope; this.currentFunction = this.kernel;
    const main = this.body(this.kernel.body);
    emitHelpers();this.scopes=kernelScope;this.currentFunction=this.kernel;
    for(const scalar of this.constantScalars)scalars.push({...scalar,offset:scalars.length*4});
    const uniformSize=scalars.length?Math.ceil(scalars.length*4/16)*16:0;
    if(uniformSize){
      header.push('struct CWParams {',...scalars.map(s=>`  ${s.field||'p_'+s.name}: ${s.type},`));
      for(let i=scalars.length*4;i<uniformSize;i+=4)header.push(`  cw_pad_${i}: u32,`);
      header.push('}',`@group(0) @binding(${bindings.length}) var<uniform> cw_params: CWParams;`);
    }
    header.push(`const cw_block_size: vec3<u32> = vec3<u32>(${this.workgroupSize.map(x => `${x}u`).join(', ')});`);
    if(this.dynamicSharedBytes&&!this.dynamicSharedUsed)this.fail('sharedMemoryBytes was supplied but the kernel has no dynamic shared array.',this.kernel);
    // Runtime parameters preserve CUDA wraparound even when call arguments are literals;
    // WGSL rejects overflowing constant expressions in an inline multiply.
    if(this.integerIntrinsics.has('__mul24'))helperLines.unshift('fn cw_mul24(a: i32, b: i32) -> i32 { return ((a << 8u) >> 8u) * ((b << 8u) >> 8u); }');
    if(this.integerIntrinsics.has('__umul24'))helperLines.unshift('fn cw_umul24(a: u32, b: u32) -> u32 { return (a & 16777215u) * (b & 16777215u); }');
    for (const s of this.shared) header.push(`var<workgroup> ${s.code}: ${s.atomic ? sharedAtomicType(s.type) : typeName(s.type)};`);
    const storageSize = this.shared.reduce((n, s) => n + Math.ceil(typeStride(s.type) / 16) * 16, 0);
    const wgsl = [...header, '', ...helperLines, '', `@compute @workgroup_size(${this.workgroupSize.join(', ')})`, 'fn main(', '  @builtin(local_invocation_id) cw_thread: vec3<u32>,', '  @builtin(workgroup_id) cw_block: vec3<u32>,', '  @builtin(num_workgroups) cw_grid: vec3<u32>', ') {', ...indent(main), '}', ''].join('\n');
    return {version: COMPILER_VERSION, name: this.kernel.name, entryPoint: 'main', wgsl, metadata: {workgroupSize: this.workgroupSize, bindings, scalars, uniformSize, uniformBinding: uniformSize ? bindings.length : null, workgroupStorageBytes: storageSize,...(this.dynamicSharedUsed?{dynamicSharedMemoryBytes:this.dynamicSharedBytes}:{}), barrier: this.usage.storageBarrier ? 'workgroup-and-storage' : 'workgroup'}, ast: this.ast, kernel: this.kernel};
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
  const definitions=new Map(),specializations=new Map(),instances=new Map(),visiting=new Set(),done=new Set(),clones=[];
  for(const fn of ast.functions){
    if(fn.specializationArgument!==undefined){
      const primary=definitions.get(fn.name),key=fn.name+'<'+fn.specializationArgument+'>';
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
      const callee=node.callee,definition=definitions.get(callee.name),argument=callee.templateArgument;
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
        const parameter=definition.templateParameter,placeholder='template:'+parameter;
        let type;
        if(definition.templateKind==='type'){
          type=builtinType(argument);
          if(!type||type==='void')fail('Template type argument must be a supported built-in value type.',callee);
        }else if(!/^\d+$/.test(argument)||!Number.isSafeInteger(Number(argument))||Number(argument)>2147483647)fail('Template argument must be a nonnegative 32-bit signed integer.',callee);
        for(const param of instance.params){if(param.name===parameter)fail('Template parameter shadowing is unsupported.',param);if(param.type===placeholder)param.type=type;}
        if(instance.result===placeholder)instance.result=type;
        walk(instance.body,n=>{
          if(['decl','thread-block'].includes(n.kind)&&n.name===parameter)fail('Template parameter shadowing is unsupported.',n);
          if(n.templateArgument===parameter)n.templateArgument=argument;
          if(type){if(n.type===placeholder)n.type=type;if(n.target===placeholder)n.target=type;}
          else if(n.kind==='id'&&n.name===parameter){n.kind='literal';n.value=argument;delete n.name;}
        });
        resolveTraitTypes(instance,ast,parameter,argument);
        if(selected){
          const expected={...definition,params:structuredClone(definition.params),body:{kind:'block',body:[]}};
          if(expected.result===placeholder)expected.result=type;
          for(const p of expected.params)if(p.type===placeholder)p.type=type;
          resolveTraitTypes(expected,ast,parameter,argument);
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
    if(types.length!==definition.params.length)fail(`Wrong number of arguments for '${name}'.`,callee);
    const matches=definition.params.flatMap((p,i)=>p.type==='template:'+definition.templateParameter?[p.pointer?(isArray(types[i])?types[i].element:undefined):types[i]]:[]);
    if(!matches.length)fail('Cannot deduce helper template type from these parameters; supply an explicit argument.',callee);
    const type=matches[0];
    if(matches.some(t=>typeName(t)!==typeName(type)))fail('Conflicting deduced helper template argument types.',callee);
    const names=['float','int','uint','bool',...['float','int','uint'].flatMap(p=>[2,3,4].map(n=>p+n))],argument=names.find(n=>builtinType(n)===type);
    if(!argument)fail('Deduced helper template argument must be a supported built-in value type.',callee);
    const call={kind:'call',token:callee.token,callee:{...callee,templateArgument:argument},args:[]};
    process({kind:'function',name:'deduction',token:callee.token,result:'void',params:[],body:{kind:'block',body:[call]}});
    for(const fn of clones)if(!ast.functions.includes(fn))ast.functions.push(fn);
    return definitions.get(call.callee.name);
  }};
}
export function compile(source, options = {},bufferUsage=null) {
  const ast = parse(source, options), kernels = ast.functions.filter(f => f.qualifier === '__global__');
  const specialization=options.entry?.match(/^([A-Za-z_]\w*)<\s*(\d+|[A-Za-z_]\w*)\s*>$/),entry=specialization?specialization[1]:options.entry;
  const kernel = entry ? kernels.find(k => k.name === entry) : kernels.length === 1 ? kernels[0] : null;
  if (!kernel) throw new CompileError(options.entry ? `Kernel '${options.entry}' was not found.` : 'Multiple kernels found; specify options.entry.');
  if(!!kernel.templateParameter!==!!specialization)throw new CompileError(kernel.templateParameter?'Specify a template entry, for example '+kernel.name+(kernel.templateKind==='type'?'<float>.':'<16>.'):'This kernel does not have a template parameter.',kernel.token,source);
  if(specialization&&kernel.templateKind==='type'){
    const type=builtinType(specialization[2]),name=kernel.templateParameter,placeholder='template:'+name;
    if(!type||type==='void')throw new CompileError('Template type argument must be a supported built-in value type.',kernel.token,source);
    for(const p of kernel.params){if(p.name===name)throw new CompileError('Template parameter shadowing is unsupported.',p.token,source);if(p.type===placeholder)p.type=type;}
    if(kernel.result===placeholder)kernel.result=type;
    walk(kernel.body,n=>{if(['decl','thread-block'].includes(n.kind)&&n.name===name)throw new CompileError('Template parameter shadowing is unsupported.',n.token,source);if(n.type===placeholder)n.type=type;if(n.target===placeholder)n.target=type;});
  }else if(specialization){const value=Number(specialization[2]),name=kernel.templateParameter;
    if(!Number.isSafeInteger(value)||value>2147483647)throw new CompileError('Template argument must be a nonnegative 32-bit signed integer.',kernel.token,source);
    for(const p of kernel.params)if(p.name===name)throw new CompileError('Template parameter shadowing is unsupported.',p.token,source);
    walk(kernel.body,n=>{if(['decl','thread-block'].includes(n.kind)&&n.name===name)throw new CompileError('Template parameter shadowing is unsupported.',n.token,source);if(n.kind==='id'&&n.name===name){n.kind='literal';n.value=String(value);delete n.name;}});
  }
  if(specialization)walk(kernel.body,n=>{if(n.templateArgument===kernel.templateParameter)n.templateArgument=specialization[2];});
  resolveTraitTypes(kernel,ast,kernel.templateParameter,specialization?.[2]);
  const scalarConstraints=uniformBlockGuards(kernel,options,walk,message=>{throw new CompileError(message,kernel.token,source);});
  const templates=instantiateHelperTemplates(ast,kernel);
  const emitter=new Emitter(ast,kernel,options,templates,bufferUsage),result=emitter.emit();
  if(scalarConstraints.length)result.metadata.scalarConstraints=scalarConstraints;
  const changed=['reads','writes','atomic'].some(k=>[...emitter.usage[k]].some(name=>!emitter.initialBufferUsage[k].has(name)));
  if(changed){if(bufferUsage)throw new CompileError('Helper buffer access analysis did not converge.');return compile(source,options,Object.fromEntries(['reads','writes','atomic'].map(k=>[k,[...emitter.usage[k]]])));}if(specialization)result.metadata.templateArguments={[kernel.templateParameter]:kernel.templateKind==='type'?specialization[2]:Number(specialization[2])};return result;
}
export function serializableArtifact(compiled) {
  return {version: compiled.version, name: compiled.name, entryPoint: compiled.entryPoint, wgsl: compiled.wgsl, metadata: compiled.metadata};
}
