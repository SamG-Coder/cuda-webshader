import {parse, CompileError} from './parser.js';
export {CompileError, parse};
export const COMPILER_VERSION = '0.1.0';
export const isArray = t => !!t && typeof t === 'object' && t.kind === 'array';
export const arrayOf = (element, length = null) => ({kind: 'array', element, length});
export const typeName = t => isArray(t) ? `array<${typeName(t.element)}${t.length == null ? '' : `, ${t.length}`}>` : t;
export const vectorLength = t => typeof t === 'string' && /^vec[234]</.test(t) ? Number(t[3]) : 0;
export const typeStride = t => isArray(t) ? typeStride(t.element) * t.length : vectorLength(t) === 3 ? 16 : (vectorLength(t) || 1) * 4;
const numeric = t => ['f32', 'i32', 'u32'].includes(t);
const indent = lines => lines.map(l => `  ${l}`);
const rootName = n => n?.kind === 'id' ? n.name : ['index', 'member'].includes(n?.kind) ? rootName(n.base) : null;
export function walk(node, visit) {
  if (!node || typeof node !== 'object') return;
  if (node.kind) visit(node);
  for (const [key, val] of Object.entries(node)) {
    if (['token', 'symbol', 'type', 'source', 'resolved'].includes(key)) continue;
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
  const atomic = new Set(), reads = new Set(), writes = new Set(), bufferNames = new Set(params.filter(p => p.pointer).map(p => p.name));
  const scan = (n, mode = 'read') => {
    if (!n || !n.kind) return;
    if (n.kind === 'assign') { scan(n.left, n.op === '=' ? 'write' : 'both'); scan(n.right); return; }
    if (n.kind === 'unary' && ['++', '--'].includes(n.op)) { scan(n.value, 'both'); return; }
    if (n.kind === 'call' && n.callee.kind === 'id' && ['atomicAdd', 'atomicMin', 'atomicMax', 'atomicExch'].includes(n.callee.name)) {
      const target = n.args[0];
      if (target?.kind === 'unary' && target.op === '&') { const name = rootName(target.value); if (name) atomic.add(name); scan(target.value, 'both'); }
      n.args.slice(1).forEach(a => scan(a)); return;
    }
    if (n.kind === 'index') {
      const name = rootName(n); if (bufferNames.has(name)) { if (mode !== 'write') reads.add(name); if (mode !== 'read') writes.add(name); }
      scan(n.index); if (n.base.kind === 'index') scan(n.base, mode); return;
    }
    if (n.kind === 'member') { scan(n.base, mode); return; }
    for (const [key, val] of Object.entries(n)) {
      if (['token', 'source'].includes(key)) continue;
      if (Array.isArray(val)) val.forEach(x => scan(x)); else if (val?.kind) scan(val);
    }
  };
  functions.forEach(f => scan(f.body));
  return {atomic, reads, writes, storageBarrier: [...writes].some(x => reads.has(x))};
}
class Emitter {
  constructor(ast, kernel, options) {
    this.ast = ast; this.kernel = kernel; this.options = options; this.scopes = [new Map()]; this.temp = 0; this.loopDepth = 0; this.integerIntrinsics=new Set();
    this.functions = new Map(); this.shared = [];
    for (const f of ast.functions) {
      if (this.functions.has(f.name)) this.fail(`Duplicate function '${f.name}'.`, f);
      this.functions.set(f.name, f);
    }
    this.helpers = ast.functions.filter(f => f.qualifier === '__device__');
    this.usage = analyse([kernel, ...this.helpers], kernel.params);
    this.workgroupSize = [...(options.workgroupSize || [128, 1, 1])];
    while (this.workgroupSize.length < 3) this.workgroupSize.push(1);
    if (this.workgroupSize.length !== 3 || this.workgroupSize.some(v => !Number.isSafeInteger(v) || v < 1) || this.workgroupSize.reduce((a, b) => a * b, 1) > 1024) this.fail('Workgroup dimensions must be positive integers with at most 1024 total invocations.', kernel);
    this.currentFunction = kernel;
  }
  fail(message, n) { throw new CompileError(message, n?.token, this.ast.source); }
  lookup(name, n) { for (let i = this.scopes.length - 1; i >= 0; --i) { const s = this.scopes[i].get(name); if (s) return s; } this.fail(`Unknown identifier '${name}'.`, n); }
  add(name, symbol, n, global = false) { const scope = global ? this.scopes[0] : this.scopes.at(-1); if (scope.has(name)) this.fail(`Duplicate identifier '${name}'.`, n); scope.set(name, symbol); return symbol; }
  common(a, b, n) {
    if (typeName(a) === typeName(b) && !isArray(a) && a !== 'void') return a;
    if (numeric(a) && numeric(b)) return a === 'f32' || b === 'f32' ? 'f32' : a === 'u32' || b === 'u32' ? 'u32' : 'i32';
    this.fail(`Incompatible operand types: ${typeName(a)} and ${typeName(b)}. Use explicit scalar/vector components.`, n);
  }
  convert(code, from, to, n) {
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
        return this.result(n, s.type, s.code, [], {rootSymbol: s, atomicRoot: s.atomic});
      }
      case 'index': {
        const base = this.expr(n.base, true), index = this.expr(n.index);
        if (!isArray(base.type) || !['i32', 'u32'].includes(index.type)) this.fail('Indexing requires an array and a 32-bit integer index.', n);
        const code = `${base.code}[${index.code}]`, type = base.type.element;
        const atomic = base.atomicRoot && !isArray(type);
        return this.result(n, type, atomic && !raw ? `atomicLoad(&${code})` : code, [...base.pre, ...index.pre], {rootSymbol: base.rootSymbol, atomicRoot: base.atomicRoot, atomic});
      }
      case 'member': {
        if (n.base.kind === 'id' && ['threadIdx', 'blockIdx', 'blockDim', 'gridDim'].includes(n.base.name)) {
          if (this.currentFunction !== this.kernel) this.fail('Pass CUDA built-in indices as scalar arguments to __device__ helpers.', n);
          if (!['x', 'y', 'z'].includes(n.member)) this.fail('CUDA dimensions have x, y and z components only.', n);
          const code = {threadIdx: 'cw_thread', blockIdx: 'cw_block', blockDim: 'cw_block_size', gridDim: 'cw_grid'}[n.base.name];
          return this.result(n, 'u32', `${code}.${n.member}`);
        }
        const base = this.expr(n.base, raw), size = vectorLength(base.type);
        if (!size || n.member.length !== 1 || 'xyzw'.indexOf(n.member) < 0 || 'xyzw'.indexOf(n.member) >= size) this.fail('Only valid single vector components (.x/.y/.z/.w) are supported.', n);
        return this.result(n, 'f32', `${base.code}.${n.member}`, base.pre, {rootSymbol: base.rootSymbol});
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
  call(n) {
    const groupSync=n.callee.kind==='id'&&n.callee.name==='cooperative_groups::sync';
    const memberSync=n.callee.kind==='member'&&n.callee.member==='sync';
    if(groupSync||memberSync){
      const group=groupSync?n.args[0]:n.callee.base;
      if(n.args.length!==(groupSync?1:0)||group?.kind!=='id'||this.lookup(group.name,group).kind!=='thread-block')this.fail('Block sync requires a local thread_block handle from this_thread_block().',n);
      if(this.currentFunction!==this.kernel)this.fail('Barriers in helper functions are not supported.',n);
      n.callName='__syncthreads';return this.result(n,'void','workgroupBarrier()');
    }
    if (n.callee.kind !== 'id') this.fail('Only named functions are supported.', n);
    const name = n.callee.name; n.callName = name;
    if (name === '__syncthreads') { if (n.args.length) this.fail('__syncthreads takes no arguments.', n); if (this.currentFunction !== this.kernel) this.fail('Barriers in helper functions are not supported.', n); return this.result(n, 'void', 'workgroupBarrier()'); }
    const atomics = {atomicAdd: 'atomicAdd', atomicMin: 'atomicMin', atomicMax: 'atomicMax', atomicExch: 'atomicExchange'};
    if (atomics[name]) {
      if (n.args.length !== 2 || n.args[0].kind !== 'unary' || n.args[0].op !== '&' || n.args[0].value.kind !== 'index') this.fail(`${name} requires &buffer[index] and a scalar value.`, n);
      const target = this.expr(n.args[0].value, true), value = this.expr(n.args[1]);
      if (!target.atomic || !['u32', 'i32'].includes(target.type)) this.fail('Only integer buffer/shared-array atomics are supported; CUDA float atomicAdd is not silently emulated.', n);
      this.writable(target, n.args[0].value);
      return this.result(n, target.type, `${atomics[name]}(&${target.code}, ${this.convert(value.code, value.type, target.type, n)})`, [...target.pre, ...value.pre]);
    }
    const casts = {float: 'f32', int: 'i32', uint: 'u32', bool: 'bool'};
    const args = n.args.map(a => this.expr(a)), pre = args.flatMap(a => a.pre);
    if(name==='__mul24'||name==='__umul24'){
      if(args.length!==2||args.some(a=>!['i32','u32'].includes(a.type)))this.fail(`${name} requires two 32-bit integer arguments.`,n);
      const signed=name==='__mul24',type=signed?'i32':'u32';
      this.integerIntrinsics.add(name);
      return this.result(n,type,`cw_${signed?'mul24':'umul24'}(${args.map(a=>this.convert(a.code,a.type,type,n)).join(', ')})`,pre);
    }
    if (casts[name]) { if (args.length !== 1) this.fail('Scalar casts require one argument.', n); return this.result(n, casts[name], this.convert(args[0].code, args[0].type, casts[name], n), pre); }
    if (/^make_float[234]$/.test(name)) {
      const count = Number(name.at(-1)); if (args.length !== count) this.fail(`${name} needs ${count} arguments.`, n);
      return this.result(n, `vec${count}<f32>`, `vec${count}<f32>(${args.map(a => this.convert(a.code, a.type, 'f32', n)).join(', ')})`, pre);
    }
    const unary = {sinf: 'sin', cosf: 'cos', tanf: 'tan', sqrtf: 'sqrt', rsqrtf: 'inverseSqrt', expf: 'exp', exp2f: 'exp2', logf: 'log', log2f: 'log2', fabsf: 'abs', floorf: 'floor', ceilf: 'ceil', truncf: 'trunc'};
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
    const helper = this.functions.get(name);
    if (!helper || helper.qualifier !== '__device__') this.fail(`Unsupported function '${name}'. CUDA host APIs, warp intrinsics, dynamic launches and libraries are not available.`, n);
    if (args.length !== helper.params.length) this.fail(`Wrong number of arguments for '${name}'.`, n);
    return this.result(n, helper.result, `f_${name}(${args.map((a, i) => this.convert(a.code, a.type, helper.params[i].type, n)).join(', ')})`, pre);
  }
  writable(target, n) {
    const s = target.rootSymbol;
    if (!s || !['id', 'index', 'member'].includes(n.kind) || isArray(target.type)) this.fail('Assignment requires a scalar/vector variable or array element.', n);
    if (s.constant) this.fail(`Cannot write through const '${s.name}'.`, n);
    if (s.kind === 'uniform') this.fail('Scalar kernel parameters are read-only in this subset. Copy the parameter to a local variable first.', n);
  }
  effect(n) {
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
    if (n.pointer) this.fail('Local pointers are unsupported; use buffer parameters.', n);
    if (n.type === 'void') this.fail('Variables cannot have void type.', n);
    let type = n.type;
    const dims = n.dimensions.map(d => { const value = constantValue(d); if (!Number.isSafeInteger(value) || value < 1 || value > 65536) this.fail('Invalid fixed array dimension (1..65536).', n); return value; });
    for (let i = dims.length - 1; i >= 0; i--) type = arrayOf(type, dims[i]);
    if (n.shared && (!dims.length || n.init)) this.fail('__shared__ requires a fixed-size array without an initializer.', n);
    if (isArray(type) && n.init) this.fail('Array initializers are unsupported. Initialize elements explicitly.', n);
    const atomic = n.shared && this.usage.atomic.has(n.name);
    if (atomic && !['i32', 'u32'].includes(n.type)) this.fail('Shared atomics require int or unsigned int.', n);
    const init = n.init ? this.expr(n.init) : null;
    if (n.constant && !init && !n.shared) this.fail('A const local variable needs an initializer.', n);
    const code = `${n.shared ? 's' : 'v'}_${n.name}`;
    const symbol = {name: n.name, type, code, constant: n.constant, atomic, kind: n.shared ? 'shared' : 'local'};
    if (n.shared && this.currentFunction !== this.kernel) this.fail('Shared memory in helpers is unsupported.', n);
    this.add(n.name, symbol, n); n.symbol = symbol; n.resolvedDimensions = dims; n.resolvedType = type;
    if (n.shared) {
      if (this.shared.some(x => x.name === n.name)) this.fail('Shared array names must be unique.', n);
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
        if(this.currentFunction!==this.kernel)this.fail('thread_block handles are supported only inside a kernel.',n);
        n.symbol=this.add(n.name,{name:n.name,kind:'thread-block',constant:true},n);return [];
      case 'block': return ['{', ...indent(this.body(n)), '}'];
      case 'decl': return this.declare(n);
      case 'expr': return this.effect(n.value);
      case 'if': { const condition = this.expr(n.condition); return [...condition.pre, `if (${this.convert(condition.code, condition.type, 'bool', n)}) {`, ...indent(this.body(n.yes)), ...(n.no ? ['} else {', ...indent(this.body(n.no))] : []), '}']; }
      case 'for': case 'while': {
        this.scopes.push(new Map()); this.loopDepth++;
        const init = n.kind === 'for' && n.init ? (n.init.kind === 'decl' ? this.declare(n.init) : this.effect(n.init)) : [];
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
      if (p.shared || p.type === 'void') this.fail('Invalid kernel parameter type.', p);
      if (p.pointer) {
        if (p.type === 'bool' || p.type === 'vec3<f32>') this.fail('bool* and float3* have incompatible CUDA/WGSL layouts. Use unsigned int* or float4*.', p);
        const atomic = this.usage.atomic.has(p.name), readOnly = p.constant || !this.usage.writes.has(p.name);
        if (p.constant && this.usage.writes.has(p.name)) this.fail(`Cannot write through const buffer '${p.name}'.`, p);
        if (atomic && !['i32', 'u32'].includes(p.type)) this.fail('Only 32-bit integer atomics are supported.', p);
        const binding = bindings.length;
        bindings.push({name: p.name, elementType: p.type, stride: typeStride(p.type), binding, readOnly, atomic});
        const symbol = {name: p.name, type: arrayOf(p.type), code: `b_${p.name}`, constant: p.constant, atomic, kind: 'buffer'};
        this.add(p.name, symbol, p, true); p.symbol = symbol;
        header.push(`@group(0) @binding(${binding}) var<storage, ${readOnly ? 'read' : 'read_write'}> b_${p.name}: array<${atomic ? `atomic<${p.type}>` : p.type}>;`);
      } else {
        if (!numeric(p.type)) this.fail('Scalar kernel parameters must be float, int or unsigned int. Put vectors in buffers.', p);
        scalars.push({name: p.name, type: p.type, offset: scalars.length * 4});
        const symbol = {name: p.name, type: p.type, code: `cw_params.p_${p.name}`, constant: false, atomic: false, kind: 'uniform'};
        this.add(p.name, symbol, p, true); p.symbol = symbol;
      }
    }
    const uniformSize = scalars.length ? Math.ceil(scalars.length * 4 / 16) * 16 : 0;
    if (uniformSize) {
      header.push('struct CWParams {', ...scalars.map(s => `  p_${s.name}: ${s.type},`));
      for (let i = scalars.length * 4; i < uniformSize; i += 4) header.push(`  cw_pad_${i}: u32,`);
      header.push('}', `@group(0) @binding(${bindings.length}) var<uniform> cw_params: CWParams;`);
    }
    header.push(`const cw_block_size: vec3<u32> = vec3<u32>(${this.workgroupSize.map(x => `${x}u`).join(', ')});`);
    const helperLines = [];
    // Helpers cannot capture kernel arguments; explicit scalar arguments only.
    const kernelScope = this.scopes;
    for (const helper of this.helpers) {
      this.scopes = [new Map()]; this.currentFunction = helper;
      for (const p of helper.params) {
        if (p.pointer || p.shared || p.type === 'void') this.fail('Helper arguments must be scalar/vector values, not pointers.', p);
        p.symbol = this.add(p.name, {name: p.name, type: p.type, code: `v_${p.name}`, constant: true, atomic: false, kind: 'local'}, p);
      }
      const body = this.body(helper.body);
      helperLines.push(`fn f_${helper.name}(${helper.params.map(p => `v_${p.name}: ${p.type}`).join(', ')})${helper.result === 'void' ? '' : ` -> ${helper.result}`} {`, ...indent(body), '}');
    }
    this.scopes = kernelScope; this.currentFunction = this.kernel;
    const main = this.body(this.kernel.body);
    // Runtime parameters preserve CUDA wraparound even when call arguments are literals;
    // WGSL rejects overflowing constant expressions in an inline multiply.
    if(this.integerIntrinsics.has('__mul24'))helperLines.unshift('fn cw_mul24(a: i32, b: i32) -> i32 { return ((a << 8u) >> 8u) * ((b << 8u) >> 8u); }');
    if(this.integerIntrinsics.has('__umul24'))helperLines.unshift('fn cw_umul24(a: u32, b: u32) -> u32 { return (a & 16777215u) * (b & 16777215u); }');
    for (const s of this.shared) header.push(`var<workgroup> ${s.code}: ${s.atomic ? sharedAtomicType(s.type) : typeName(s.type)};`);
    const storageSize = this.shared.reduce((n, s) => n + Math.ceil(typeStride(s.type) / 16) * 16, 0);
    const wgsl = [...header, '', ...helperLines, '', `@compute @workgroup_size(${this.workgroupSize.join(', ')})`, 'fn main(', '  @builtin(local_invocation_id) cw_thread: vec3<u32>,', '  @builtin(workgroup_id) cw_block: vec3<u32>,', '  @builtin(num_workgroups) cw_grid: vec3<u32>', ') {', ...indent(main), '}', ''].join('\n');
    return {version: COMPILER_VERSION, name: this.kernel.name, entryPoint: 'main', wgsl, metadata: {workgroupSize: this.workgroupSize, bindings, scalars, uniformSize, uniformBinding: uniformSize ? bindings.length : null, workgroupStorageBytes: storageSize, barrier: this.usage.storageBarrier ? 'workgroup-and-storage' : 'workgroup'}, ast: this.ast, kernel: this.kernel};
  }
}
export function compile(source, options = {}) {
  const ast = parse(source, options), kernels = ast.functions.filter(f => f.qualifier === '__global__');
  const kernel = options.entry ? kernels.find(k => k.name === options.entry) : kernels.length === 1 ? kernels[0] : null;
  if (!kernel) throw new CompileError(options.entry ? `Kernel '${options.entry}' was not found.` : 'Multiple kernels found; specify options.entry.');
  return new Emitter(ast, kernel, options).emit();
}
export function serializableArtifact(compiled) {
  return {version: compiled.version, name: compiled.name, entryPoint: compiled.entryPoint, wgsl: compiled.wgsl, metadata: compiled.metadata};
}
