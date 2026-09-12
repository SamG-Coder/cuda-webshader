/** A deliberately bounded CUDA C frontend. No eval, regex transpilation, or source-specific rewrites. */
import {forwardingMacro} from './macros.js';
export class CompileError extends Error {
  constructor(message, token = {}, source = '') {
    const line = token.line || 1, column = token.column || 1;
    super(`${message} (${line}:${column})${source ? `\n${source.split('\n')[line - 1] || ''}\n${' '.repeat(column - 1)}^` : ''}`);
    this.name = 'CompileError'; this.line = line; this.column = column;
  }
}
const NUM = /^(?:0[xX][\da-fA-F]+(?:[uU][lL]?|[lL][uU])?|(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?(?:[uU][lL]?|[lL][uU]|[fF])?)/;
const WORD = /^[A-Za-z_]\w*/;
const OPERATORS = ['<<=', '>>=', '::', '++', '--', '+=', '-=', '*=', '/=', '%=', '==', '!=', '<=', '>=', '&&', '||', '<<', '>>', '&=', '|=', '^=', '->'];
const TYPES = new Set(['float', 'int', 'uint', 'unsigned', 'bool', 'void', 'float2', 'float3', 'float4']);
const QUALIFIERS = new Set(['const', '__shared__', '__restrict__', '__restrict', 'restrict','extern']);
const MAP = { float: 'f32', int: 'i32', uint: 'u32', bool: 'bool', void: 'void', float2: 'vec2<f32>', float3: 'vec3<f32>', float4: 'vec4<f32>' };
for(const [prefix,type] of [['uint','u32'],['int','i32']])for(const size of [2,3,4]){TYPES.add(prefix+size);MAP[prefix+size]=`vec${size}<${type}>`;}
export const builtinType = name => Object.hasOwn(MAP,name)?MAP[name]:null;
export function tokenize(source, defines = {}) {
  if (typeof source !== 'string' || source.length > 1_000_000) throw new CompileError('Source must be a string of at most 1 MB.');
  const macros = new Map(Object.entries(defines).map(([k, v]) => {
    if (!/^[A-Za-z_]\w*$/.test(k) || !Number.isFinite(v)) throw new CompileError('Defines must be named finite numbers.');
    return [k, String(v)];
  }));
  const tokens = [],forwarders=new Map(); let i = 0, line = 1, column = 1;
  const advance = str => { for (const c of str) { if (c === '\n') { line++; column = 1; } else column++; } i += str.length; };
  while (i < source.length) {
    const rest = source.slice(i), token = {line, column, offset: i};
    if (/^\s/.test(rest)) { advance(rest.match(/^\s+/)[0]); continue; }
    if (rest.startsWith('//')) { advance(rest.split('\n')[0]); continue; }
    if (rest.startsWith('/*')) { const end = rest.indexOf('*/'); if (end < 0) throw new CompileError('Unclosed comment.', token, source); advance(rest.slice(0, end + 2)); continue; }
    if (rest[0] === '#') {
      const directive = rest.split('\n')[0];
      if(/^#\s*pragma\s+unroll(?:\s+[1-9]\d*)?\s*(?:\/\/.*)?$/.test(directive.trimEnd())){advance(directive);continue;}
      const forward=forwardingMacro(directive.trimEnd());
      if(forward){if(macros.has(forward.name)||forwarders.has(forward.name))throw new CompileError('Macro redefinition is unsupported.',token,source);forwarders.set(forward.name,forward);advance(directive);continue;}
      const m = directive.trimEnd().match(/^#\s*define\s+([A-Za-z_]\w*)\s+([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?[fFuU]?)\s*(?:\/\/.*)?$/);
      if (!m) throw new CompileError('Only numeric object-like #define directives and direct function-forwarding macros are supported; preprocess other directives first.', token, source);
      if(forwarders.has(m[1]))throw new CompileError('Macro redefinition is unsupported.',token,source);
      if (!macros.has(m[1])) macros.set(m[1], m[2]); advance(directive); continue;
    }
    if(rest[0]==='"') {const literal=rest.match(/^"[^"\n\r\\]*"/);if(!literal)throw new CompileError('Unsupported or unterminated string literal.',token,source);tokens.push({...token,kind:'string',value:literal[0]});advance(literal[0]);continue;}
    const number = rest.match(NUM);
    if (number) { tokens.push({...token, kind: 'number', value: number[0].replace(/(?:[uU][lL]|[lL][uU])$/,'u')}); advance(number[0]); continue; }
    const word = rest.match(WORD);
    if (word) {
      const value = word[0], expanded = macros.get(value);
      if (expanded !== undefined) {
        let body = expanded;
        if (body.startsWith('-') || body.startsWith('+')) { tokens.push({...token, kind: 'symbol', value: body[0]}); body = body.slice(1); }
        tokens.push({...token, kind: 'number', value: body});
      } else {
        const chain=[];let target=value;const seen=new Set();
        while(forwarders.has(target)&&!seen.has(target)&&chain.length<32){seen.add(target);const f=forwarders.get(target);chain.push(f);target=f.target;}
        tokens.push({...token, kind: 'word', value,...(chain.length?{forward:{chain,target,tooDeep:forwarders.has(target)&&!seen.has(target),recursive:seen.has(target),numericTarget:macros.has(target)}}:{})});
      }
      advance(value); continue;
    }
    const op = OPERATORS.find(x => rest.startsWith(x));
    if (op) { tokens.push({...token, kind: 'symbol', value: op}); advance(op); continue; }
    if ('{}[]();,.?:+-*/%<>=!~&|^'.includes(rest[0])) { tokens.push({...token, kind: 'symbol', value: rest[0]}); advance(rest[0]); continue; }
    throw new CompileError(`Unsupported character ${JSON.stringify(rest[0])}.`, token, source);
  }
  tokens.push({kind: 'eof', value: '<eof>', line, column, offset: i}); return tokens;
}
const PRECEDENCE = {'=': 1, '+=': 1, '-=': 1, '*=': 1, '/=': 1, '%=': 1, '&=': 1, '|=': 1, '^=': 1, '<<=': 1, '>>=': 1, '||': 3, '&&': 4, '|': 5, '^': 6, '&': 7, '==': 8, '!=': 8, '<': 9, '>': 9, '<=': 9, '>=': 9, '<<': 10, '>>': 10, '+': 11, '-': 11, '*': 12, '/': 12, '%': 12};
export class Parser {
  constructor(source, defines) { this.source = source; this.tokens = tokenize(source, defines); this.i = 0; this.groupNamespaces = new Set(['cooperative_groups']); this.functionNames=new Set(); }
  peek(offset = 0) { return this.tokens[this.i + offset] || this.tokens.at(-1); }
  is(value) { return this.peek().value === value; }
  take(value) { if (value && !this.is(value)) this.fail(`Expected '${value}', found '${this.peek().value}'.`); return this.tokens[this.i++]; }
  match(value) { if (this.is(value)) { this.i++; return true; } return false; }
  fail(message, token = this.peek()) { throw new CompileError(message, token, this.source); }
  name() { const t = this.take(); if (t.kind !== 'word') this.fail('Expected an identifier.', t); return t.value; }
  qualifiedName() {
    const token=this.peek(),name=this.name();
    if(!this.match('::'))return name;
    if(!this.groupNamespaces.has(name))this.fail(`Unsupported namespace '${name}'. Only cooperative_groups namespace aliases are supported.`,token);
    return 'cooperative_groups::'+this.name();
  }
  startsType() { return TYPES.has(this.peek().value) || this.peek().value===this.templateTypeName || QUALIFIERS.has(this.peek().value); }
  type() {
    let constant = false, shared = false,external=false;
    while (QUALIFIERS.has(this.peek().value)) { const q = this.take().value; constant ||= q === 'const'; shared ||= q === '__shared__';external ||= q==='extern'; }
    const tok = this.take(); let type;
    if (tok.value === 'unsigned') { this.match('int'); type = 'u32'; } else type = tok.value===this.templateTypeName?'template:'+tok.value:builtinType(tok.value);
    if (!type) this.fail(`Unsupported type '${tok.value}'. Use float, int, unsigned int, bool or float2/3/4.`, tok);
    if (this.match('const')) constant = true;
    const pointer = this.match('*');
    const reference=this.match('&');
    if(pointer&&reference)this.fail('Pointer references are unsupported.');
    while (['__restrict__', '__restrict', 'restrict'].includes(this.peek().value)) this.take();
    if (this.is('*')) this.fail('Pointer-to-pointer types are not supported.');
    return {type, constant, shared, pointer,reference,external};
  }
  parse() {
    const functions = [];
    while (this.peek().kind !== 'eof') {
      const token = this.peek();
      let templateParameter=null,templateKind=null;this.templateTypeName=null;this.templateParameterName=null;
      if(this.match('extern')){const linkage=this.take();if(linkage.kind!=='string'||linkage.value!=='"C"')this.fail('Only extern "C" linkage on a single device function definition is supported.',linkage);if(!['__global__','__device__'].includes(this.peek().value))this.fail('extern "C" must precede a single __global__ or __device__ function definition; linkage blocks and templates are unsupported.');}
      if(this.match('template')){this.take('<');const kind=this.take();if(!['int','class','typename'].includes(kind.value))this.fail('Only one integer or built-in type template parameter is supported.',kind);templateKind=kind.value==='int'?'int':'type';templateParameter=this.name();if(TYPES.has(templateParameter))this.fail('Template parameter must have a distinct name.',token);this.take('>');if(templateKind==='type')this.templateTypeName=templateParameter;}
      this.templateParameterName=templateParameter;
      if(this.match('namespace')){const alias=this.name();this.take('=');const target=this.name();this.take(';');if(target!=='cooperative_groups'||this.groupNamespaces.has(alias))this.fail('Only distinct aliases of cooperative_groups are supported.',token);this.groupNamespaces.add(alias);continue;}
      while (['static','inline', '__forceinline__'].includes(this.peek().value)) this.take();
      let launchThreads=null;
      const launchBounds=()=>{this.take('__launch_bounds__');this.take('(');const t=this.take();if(t.kind!=='number'||!/^[0-9]+[uU]?$/.test(t.value))this.fail('Launch bounds require a positive integer thread count.',t);launchThreads=Number(t.value.replace(/[uU]$/,''));if(launchThreads<1||launchThreads>1024)this.fail('Launch bounds thread count must be in [1,1024].',t);this.take(')');};
      if(this.is('__launch_bounds__'))launchBounds();
      const qualifier = this.take().value;
      if(templateParameter&&!['__global__','__device__'].includes(qualifier))this.fail('Templates are supported only on kernels and device helpers.',token);
      if (!['__global__', '__device__'].includes(qualifier)) this.fail('Only __global__ kernels and __device__ helper functions are accepted. Host CUDA APIs, structs, templates and PTX are not supported.', token);
      while (['inline', '__forceinline__'].includes(this.peek().value)) this.take();
      if(this.is('__launch_bounds__')){if(launchThreads!==null)this.fail('Duplicate launch bounds.');launchBounds();}
      if(launchThreads!==null&&qualifier!=='__global__')this.fail('Launch bounds apply only to kernels.',token);
      const result = this.type();
      if (result.pointer || result.shared || result.reference || result.external) this.fail('Function return pointers/references/shared/extern qualifiers are unsupported.');
      if(this.peek().forward)this.fail('Function-forwarding macros are supported at call sites, not in function declarations.');
      const name = this.name(); this.functionNames.add(name); this.take('('); const params = [];
      if (!this.is(')')) do { const token = this.peek(), type = this.type(), name = this.name(); params.push({kind: 'param', token, name, ...type}); } while (this.match(','));
      this.take(')'); const body = this.block();
      functions.push({kind: 'function', token, name, qualifier, result: result.type, params, body,launchThreads,templateParameter,templateKind});
    }
    if (!functions.some(f => f.qualifier === '__global__')) this.fail('No __global__ kernel was found.');
    return {kind: 'module', functions, source: this.source};
  }
  block() { const token = this.take('{'), body = []; while (!this.is('}')) { if (this.peek().kind === 'eof') this.fail('Unclosed block.'); body.push(this.statement()); } this.take('}'); return {kind: 'block', token, body}; }
  declaration(semicolon = true) {
    const token = this.peek(), d = this.type(),declarations=[];
    do {const name=this.name(),dimensions=[];while(this.match('[')){dimensions.push(this.is(']')?null:this.expression(2));this.take(']');}const init=this.match('=')?this.expression(2):null;declarations.push({kind:'decl',token,name,...d,dimensions,init});if(this.is(',')&&(d.pointer||d.reference))this.fail('Pointer/reference declaration lists are unsupported.');}while(this.match(','));
    if (semicolon) this.take(';');return declarations.length===1?declarations[0]:{kind:'decls',token,declarations};
  }
  statement() {
    const token = this.peek();
    if(this.match('do')){const body=this.statement();this.take('while');this.take('(');const condition=this.expression();this.take(')');this.take(';');return {kind:'do',token,body,condition};}
    if(this.groupNamespaces.has(token.value)&&this.peek(1).value==='::'&&this.peek(2).value==='thread_block'){
      this.qualifiedName();const name=this.name();this.take('=');const factory=this.qualifiedName();this.take('(');this.take(')');this.take(';');
      if(factory!=='cooperative_groups::this_thread_block')this.fail('thread_block must be initialized with cooperative_groups::this_thread_block().',token);
      return {kind:'thread-block',token,name};
    }
    if (this.is('{')) return this.block();
    if (this.match(';')) return {kind: 'empty', token};
    if (this.match('if')) { this.take('('); const condition = this.expression(); this.take(')'); const yes = this.statement(), no = this.match('else') ? this.statement() : null; return {kind: 'if', token, condition, yes, no}; }
    if (this.match('for')) { this.take('('); const init = this.is(';') ? null : this.startsType() ? this.declaration(false) : this.expression(); this.take(';'); const condition = this.is(';') ? null : this.expression(); this.take(';'); const steps=[];if(!this.is(')'))do{steps.push(this.expression());}while(this.match(','));const step=steps.length>1?{kind:'sequence',token,expressions:steps}:steps[0]||null; this.take(')'); return {kind: 'for', token, init, condition, step, body: this.statement()}; }
    if (this.match('while')) { this.take('('); const condition = this.expression(); this.take(')'); return {kind: 'while', token, condition, body: this.statement()}; }
    if (this.match('return')) { const value = this.is(';') ? null : this.expression(); this.take(';'); return {kind: 'return', token, value}; }
    if (this.match('break') || this.match('continue')) { this.take(';'); return {kind: token.value, token}; }
    if (this.startsType()) return this.declaration();
    const value = this.expression(); this.take(';'); return {kind: 'expr', token, value};
  }
  expression(min = 1) {
    let left = this.unary();
    while (true) {
      const token = this.peek(), op = token.value;
      if (op === '?' && min <= 2) { this.take(); const yes = this.expression(); this.take(':'); const no = this.expression(2); left = {kind: 'conditional', token, condition: left, yes, no}; continue; }
      const p = PRECEDENCE[op]; if (p === undefined || p < min) break;
      this.take(); const right = this.expression(p === 1 ? p : p + 1); left = {kind: p === 1 ? 'assign' : 'binary', token, op, left, right};
    }
    return left;
  }
  unary() {
    const token = this.peek();
    if(this.match('static_cast')){this.take('<');const type=this.type();if(type.pointer||type.reference||type.shared||type.external)this.fail('static_cast supports value types only.',token);this.take('>');this.take('(');const value=this.expression();this.take(')');return {kind:'cast',token,target:type.type,value};}
    if (['+', '-', '!', '~', '&', '++', '--', '*'].includes(token.value)) { this.take(); return {kind: 'unary', token, op: token.value, value: this.unary(), prefix: true}; }
    if (this.is('(') && (TYPES.has(this.peek(1).value) || this.peek(1).value===this.templateTypeName || this.peek(1).value === 'const')) { this.take('('); const type = this.type(); if (type.pointer||type.reference) this.fail('Pointer/reference casts are unsupported.'); this.take(')'); return {kind: 'cast', token, target: type.type, value: this.unary()}; }
    let value;
    if (token.kind === 'number') { this.take(); value = {kind: 'literal', token, value: token.value}; }
    else if (this.match('(')) { value = this.expression(); this.take(')'); }
    else if (token.kind === 'word') { value = {kind: 'id', token, name: this.qualifiedName()}; }
    else this.fail('Expected an expression.', token);
    while (true) {
      if(value.kind==='id'&&this.functionNames.has(value.name)&&this.is('<')&&this.peek(2).value==='>'&&this.peek(3).value==='('&&(TYPES.has(this.peek(1).value)||this.peek(1).value===this.templateParameterName||this.peek(1).kind==='number')){this.take('<');value.templateArgument=this.take().value;this.take('>');}
      else if (this.match('[')) { const index = this.expression(); this.take(']'); value = {kind: 'index', token, base: value, index}; }
      else if (this.match('.')) { value = {kind: 'member', token, base: value, member: this.name()}; }
      else if (this.match('(')) { const args = []; if (!this.is(')')) do { args.push(this.expression(2)); } while (this.match(',')); this.take(')');
        if(value.kind==='id'&&value.token.forward){const f=value.token.forward;if(f.tooDeep)this.fail('Forwarding macro chains are limited to 32 calls.',value.token);if(f.recursive||f.numericTarget)this.fail('Recursive or non-function forwarding macro target is unsupported.',value.token);if(f.chain.some(m=>m.arity!==args.length))this.fail(`Wrong argument count for forwarding macro '${value.name}'.`,value.token);value={...value,name:f.target};}
        value = {kind: 'call', token, callee: value, args}; }
      else if (this.is('++') || this.is('--')) { value = {kind: 'unary', token, op: this.take().value, value, prefix: false}; }
      else break;
    }
    return value;
  }
}
export function parse(source, options = {}) { return new Parser(source, options.defines).parse(); }
