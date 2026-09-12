/** A deliberately bounded CUDA C frontend. No eval, regex transpilation, or source-specific rewrites. */
import {forwardingMacro,expressionMacro} from './macros.js';
import {integerExpression} from './integer-expression.js';
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
TYPES.add('cudaTextureObject_t');MAP.cudaTextureObject_t='texture3d';
TYPES.add('cudaSurfaceObject_t');MAP.cudaSurfaceObject_t='surface2d';
for(const [prefix,type] of [['uint','u32'],['int','i32']])for(const size of [2,3,4]){TYPES.add(prefix+size);MAP[prefix+size]=`vec${size}<${type}>`;}
export const builtinType = name => Object.hasOwn(MAP,name)?MAP[name]:null;
function unwrapCondition(text){for(let wraps=0;wraps<32&&text.startsWith('(')&&text.endsWith(')');wraps++){let depth=0,whole=true;for(let i=0;i<text.length;i++){if(text[i]==='(')depth++;if(text[i]===')')depth--;if(depth<0||(depth===0&&i<text.length-1))whole=false;}if(!whole||depth!==0)break;text=text.slice(1,-1).trim();}return text;}
export function tokenize(source, defines = {}) {
  if (typeof source !== 'string' || source.length > 1_000_000) throw new CompileError('Source must be a string of at most 1 MB.');
  const macros = new Map(Object.entries(defines).map(([k, v]) => {
    if (!/^[A-Za-z_]\w*$/.test(k) || !Number.isFinite(v)) throw new CompileError('Defines must be named finite numbers.');
    return [k, String(v)];
  }));
  const conditionals=[];let enabled=true;
  const tokens = [],forwarders=new Map(),expressions=new Map();let numericMacroSnapshot=null; let i = 0, line = 1, column = 1;
  const advance = str => { for (const c of str) { if (c === '\n') { line++; column = 1; } else column++; } i += str.length; };
  while (i < source.length) {
    const rest = source.slice(i), token = {line, column, offset: i};
    if (/^\s/.test(rest)) { advance(rest.match(/^\s+/)[0]); continue; }
    if (rest.startsWith('//')) { advance(rest.split('\n')[0]); continue; }
    if (rest.startsWith('/*')) { const end = rest.indexOf('*/'); if (end < 0) throw new CompileError('Unclosed comment.', token, source); advance(rest.slice(0, end + 2)); continue; }
    if (rest[0] === '#') {
      const directive = rest.split('\n')[0];
      const conditional=directive.trimEnd().match(/^#\s*(if|else|endif)\b(.*)$/);
      if(conditional){const [,kind,tail]=conditional,expression=kind==='if'?unwrapCondition(tail.replace(/\/\/.*$/,'').trim()):tail.replace(/\/\/.*$/,'').trim();
        if(kind==='if'){const match=expression.match(/^(!)?\s*([A-Za-z_]\w*|[0-9]+)$/);if(!match)throw new CompileError('Conditional preprocessing supports an integer literal or numeric macro with optional !.',token,source);const raw=/^[0-9]+$/.test(match[2])?match[2]:macros.get(match[2])??'0',number=Number(raw.replace(/[uU]$/,''));if(!Number.isSafeInteger(number))throw new CompileError('Conditional macro must be an integer.',token,source);const selected=match[1]?!number:!!number;conditionals.push({parent:enabled,selected,otherwise:false});enabled=enabled&&selected;}
        else{const frame=conditionals.at(-1);if(!frame||expression)throw new CompileError('Unmatched or malformed conditional directive.',token,source);if(kind==='else'){if(frame.otherwise)throw new CompileError('Duplicate #else.',token,source);frame.otherwise=true;enabled=frame.parent&&!frame.selected;}else{conditionals.pop();enabled=frame.parent;}}
        advance(directive);continue;
      }
      if(/^#\s*(elif|ifdef|ifndef)\b/.test(directive))throw new CompileError('Unsupported conditional directive; preprocess it first.',token,source);
      if(!enabled){advance(directive);continue;}
      if(/^#\s*pragma\s+unroll(?:\s+[1-9]\d*)?\s*(?:\/\/.*)?$/.test(directive.trimEnd())){advance(directive);continue;}
      const expression=expressionMacro(directive.trimEnd());if(expression){if(macros.has(expression.name)||forwarders.has(expression.name)||expressions.has(expression.name))throw new CompileError('Macro redefinition is unsupported.',token,source);expressions.set(expression.name,expression);advance(directive);continue;}
      const forward=forwardingMacro(directive.trimEnd());
      if(forward){if(macros.has(forward.name)||forwarders.has(forward.name)||expressions.has(forward.name))throw new CompileError('Macro redefinition is unsupported.',token,source);forwarders.set(forward.name,forward);advance(directive);continue;}
      const m = directive.trimEnd().match(/^#\s*define\s+([A-Za-z_]\w*)\s+([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?[fFuU]?)\s*(?:\/\/.*)?$/);
      if(!m){const object=directive.trimEnd().match(/^#\s*define\s+([A-Za-z_]\w*)\s+(.+?)\s*(?:\/\/.*)?$/);if(object){try{const value=integerExpression(object[2].replace(/[A-Za-z_]\w*/g,name=>macros.has(name)?'('+macros.get(name)+')':name));if(forwarders.has(object[1])||expressions.has(object[1]))throw Error('Macro redefinition is unsupported.');if(!macros.has(object[1])){macros.set(object[1],String(value));numericMacroSnapshot=null;}advance(directive);continue;}catch(error){throw new CompileError(error.message,token,source);}}}
      if (!m) throw new CompileError('Only numeric object-like #define directives and direct function-forwarding macros are supported; preprocess other directives first.', token, source);
      if(forwarders.has(m[1])||expressions.has(m[1]))throw new CompileError('Macro redefinition is unsupported.',token,source);
      if (!macros.has(m[1])){macros.set(m[1], m[2]);numericMacroSnapshot=null;} advance(directive); continue;
    }
    if(!enabled){advance(rest.split('\n')[0]);continue;}
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
        tokens.push({...token, kind: 'word', value,...(expressions.has(value)?{expressionMacro:{...expressions.get(value),defines:(numericMacroSnapshot??=Object.fromEntries(macros)),forbidden:[...expressions.keys(),...forwarders.keys()]}}:{}),...(chain.length?{forward:{chain,target,tooDeep:forwarders.has(target)&&!seen.has(target),recursive:seen.has(target),numericTarget:macros.has(target)}}:{})});
      }
      advance(value); continue;
    }
    const op = OPERATORS.find(x => rest.startsWith(x));
    if (op) { tokens.push({...token, kind: 'symbol', value: op}); advance(op); continue; }
    if ('{}[]();,.?:+-*/%<>=!~&|^'.includes(rest[0])) { tokens.push({...token, kind: 'symbol', value: rest[0]}); advance(rest[0]); continue; }
    throw new CompileError(`Unsupported character ${JSON.stringify(rest[0])}.`, token, source);
  }
  if(conditionals.length)throw new CompileError('Unclosed #if directive.',{line,column},source);
  tokens.push({kind: 'eof', value: '<eof>', line, column, offset: i}); return tokens;
}
const PRECEDENCE = {'=': 1, '+=': 1, '-=': 1, '*=': 1, '/=': 1, '%=': 1, '&=': 1, '|=': 1, '^=': 1, '<<=': 1, '>>=': 1, '||': 3, '&&': 4, '|': 5, '^': 6, '&': 7, '==': 8, '!=': 8, '<': 9, '>': 9, '<=': 9, '>=': 9, '<<': 10, '>>': 10, '+': 11, '-': 11, '*': 12, '/': 12, '%': 12};
export class Parser {
  constructor(source, defines) { this.source = source; this.tokens = tokenize(source, defines); this.i = 0; this.groupNamespaces = new Set(['cooperative_groups']); this.functionNames=new Set(['tex3D','tex1D','tex2D','tex2Dgather']); this.typeTraits=new Map();this.structs=new Map(); this.sharedWrappers=new Map();this.expandedMacroNodes=0; }
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
  deferredType(name){return this.deferUnsupportedTypes&&/^double[234]?$/.test(name);}
  startsType() { return this.structs.has(this.peek().value)||TYPES.has(this.peek().value) || this.deferredType(this.peek().value) || this.peek().value==='typename' || this.typeTraits.has(this.peek().value) || this.templateTypeNames?.has(this.peek().value) || QUALIFIERS.has(this.peek().value); }
  type() {
    let constant = false, shared = false,external=false;
    while (QUALIFIERS.has(this.peek().value)) { const q = this.take().value; constant ||= q === 'const'; shared ||= q === '__shared__';external ||= q==='extern'; }
    const tok = this.take(); let type;
    if(this.groupNamespaces.has(tok.value)){this.take('::');this.take('thread_block');type='thread-block';}
    else if(tok.value==='typename'||this.typeTraits.has(tok.value)){
      const name=tok.value==='typename'?this.name():tok.value;
      if(!this.typeTraits.has(name))this.fail(`Unknown type trait '${name}'.`,tok);
      this.take('<');const argument=this.name();this.take('>');this.take('::');const member=this.name();
      type={kind:'trait-type',name,argument,member};
    }else if (tok.value === 'unsigned') { this.match('int'); type = 'u32'; } else type = this.structs.has(tok.value)?this.structs.get(tok.value).type:this.templateTypeNames?.has(tok.value)?'template:'+tok.value:builtinType(tok.value);
    if(!type&&this.deferredType(tok.value))type='unsupported:'+tok.value;
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
    const functions = [],constantGlobals=[];
    while (this.peek().kind !== 'eof') {
      const token = this.peek();
      let templateParameter=null,templateKind=null,templateParameters=[];this.templateTypeNames=new Set();this.templateParameterName=null;this.deferUnsupportedTypes=false;
      if(this.is('typedef')&&this.peek(1).value==='struct'||this.is('struct')&&this.peek(2).value==='{'){
        const alias=this.match('typedef');this.take('struct');let name=this.is('{')?null:this.name();this.take('{');const fields=[];
        while(!this.is('}')){const fieldToken=this.peek(),spec=this.type(),fieldName=this.name(),dimensions=[];if(spec.pointer||spec.reference||spec.shared||spec.external||spec.constant||(['void','texture3d','surface2d','thread-block'].includes(spec.type)||spec.type.startsWith('cw_struct_')))this.fail('Struct fields require plain scalar/vector value types.',fieldToken);while(this.match('[')){dimensions.push(this.expression(2));this.take(']');}this.take(';');if(dimensions.length>1||fields.length>=64)this.fail('Structs support at most 64 fields and one-dimensional field arrays.',fieldToken);if(fields.some(f=>f.name===fieldName))this.fail('Duplicate struct field.',fieldToken);fields.push({name:fieldName,type:spec.type,dimensions,token:fieldToken});}
        this.take('}');if(alias){const aliasName=this.name();if(name&&name!==aliasName)this.fail('Distinct struct tag/typedef aliases are unsupported.',token);name=aliasName;}this.take(';');if(!name||!fields.length||this.structs.has(name)||TYPES.has(name)||this.typeTraits.has(name))this.fail('Structs require a distinct name and at least one field.',token);if(this.structs.size>=64)this.fail('At most 64 plain structs are supported.',token);this.structs.set(name,{name,type:'cw_struct_'+name,fields,token});continue;
      }
      if(this.match('__constant__')){
        this.deferUnsupportedTypes=true;const valueType=this.type(),name=this.name();
        if(valueType.pointer||valueType.reference||valueType.shared||valueType.external)this.fail('Constant globals support scalar values and fixed scalar arrays only.',token);
        const dimensions=[];while(this.match('[')){dimensions.push(this.expression(2));this.take(']');}if(dimensions.length>1)this.fail('Constant arrays must be one-dimensional.',token);
        const init=this.match('=')?this.initializer():null;this.take(';');
        if(constantGlobals.some(g=>g.name===name))this.fail('Duplicate constant global.',token);
        constantGlobals.push({kind:'constant-global',token,name,type:valueType.type,dimensions,init});continue;
      }
      if(this.match('extern')){const linkage=this.take();if(linkage.kind!=='string'||linkage.value!=='"C"')this.fail('Only extern "C" linkage on a single device function definition is supported.',linkage);if(!['__global__','__device__'].includes(this.peek().value))this.fail('extern "C" must precede a single __global__ or __device__ function definition; linkage blocks and templates are unsupported.');}
      if(this.match('template')){
        this.take('<');
        if(this.match('>'))templateKind='specialization';
        else{do{const kind=this.take();if(!['int','class','typename'].includes(kind.value))this.fail('Template parameters require int, class or typename.',kind);const parameterKind=kind.value==='int'?'int':'type',name=this.name();if(TYPES.has(name)||templateParameters.includes(name))this.fail('Template parameters must have distinct names.',token);if(templateParameters.length>=4)this.fail('At most four helper template type parameters are supported.',token);if(templateParameters.length&&(templateKind!=='type'||parameterKind!=='type'))this.fail('Multiple template parameters currently require type parameters only.',kind);templateKind=parameterKind;templateParameters.push(name);if(parameterKind==='type')this.templateTypeNames.add(name);}while(this.match(','));this.take('>');templateParameter=templateParameters[0];}
      }
      this.templateParameterName=templateParameter;
      if(this.match('struct')){
        if(templateParameters.length>1)this.fail('Type-trait and shared-wrapper structs require one template parameter.',token);
        if(!['type','specialization'].includes(templateKind))this.fail('Only type-trait template structs containing typedef members are supported.',token);
        const name=this.name();let argument=null;
        if(templateKind==='specialization'){this.take('<');argument=this.name();this.take('>');}
        this.take('{');const members=[];
        if(this.is('__device__')){
          if(templateKind!=='type')this.fail('Shared-memory conversion wrappers require one type parameter.',token);
          const conversions=[];
          while(!this.is('}')){
            this.take('__device__');while(['inline','__forceinline__'].includes(this.peek().value))this.take();
            this.take('operator');const constant=this.match('const');this.take(templateParameter);this.take('*');this.take('(');this.take(')');const methodConst=this.match('const');
            this.take('{');this.take('extern');this.take('__shared__');const storage=this.type();const storageName=this.name();this.take('[');this.take(']');this.take(';');
            if(!['i32','u32','f32'].includes(storage.type)||storage.pointer||storage.reference||storage.constant)this.fail('Shared wrapper backing storage must be an unsized 32-bit scalar array.',token);
            this.take('return');this.take('(');this.take(templateParameter);this.take('*');this.take(')');this.take(storageName);this.take(';');this.take('}');
            if(constant!==methodConst||conversions.includes(constant))this.fail('Shared wrapper conversions require distinct mutable and const overloads.',token);conversions.push(constant);
          }
          this.take('}');this.take(';');if(this.sharedWrappers.has(name)||this.typeTraits.has(name)||TYPES.has(name))this.fail('Duplicate or reserved shared wrapper name.',token);
          this.sharedWrappers.set(name,{name,conversions});continue;
        }
        while(!this.is('}')){
          this.take('typedef');let type=this.name();if(type==='unsigned'){this.match('int');type='uint';}
          const member=this.name();this.take(';');if(members.some(m=>m.name===member))this.fail('Duplicate type-trait member.',token);members.push({name:member,type});
        }
        this.take('}');this.take(';');
        if(!members.length)this.fail('Type traits require at least one typedef member.',token);
        if(argument===null){if(this.typeTraits.has(name)||TYPES.has(name))this.fail('Duplicate or reserved type-trait name.',token);this.typeTraits.set(name,{name,parameter:templateParameter,members,specializations:[]});}
        else{const trait=this.typeTraits.get(name);if(!trait)this.fail('Declare the primary type trait before its specializations.',token);if(trait.specializations.some(s=>s.argument===argument))this.fail('Duplicate type-trait specialization.',token);trait.specializations.push({argument,members});}
        continue;
      }
      this.deferUnsupportedTypes=templateKind==='specialization';
      if(this.match('namespace')){const alias=this.name();this.take('=');const target=this.name();this.take(';');if(target!=='cooperative_groups'||this.groupNamespaces.has(alias))this.fail('Only distinct aliases of cooperative_groups are supported.',token);this.groupNamespaces.add(alias);continue;}
      while (['static','inline', '__forceinline__'].includes(this.peek().value)) this.take();
      let launchThreads=null;
      const launchBounds=()=>{this.take('__launch_bounds__');this.take('(');const t=this.take();if(t.kind!=='number'||!/^[0-9]+[uU]?$/.test(t.value))this.fail('Launch bounds require a positive integer thread count.',t);launchThreads=Number(t.value.replace(/[uU]$/,''));if(launchThreads<1||launchThreads>1024)this.fail('Launch bounds thread count must be in [1,1024].',t);this.take(')');};
      if(this.is('__launch_bounds__'))launchBounds();
      const qualifier = this.take().value;
      if(templateKind==='specialization'&&qualifier!=='__device__')this.fail('Explicit function specializations support only device helpers.',token);
      if(templateParameters.length>1&&qualifier!=='__device__')this.fail('Multiple template type parameters are supported on device helpers only.',token);
      if(templateParameter&&!['__global__','__device__'].includes(qualifier))this.fail('Templates are supported only on kernels and device helpers.',token);
      if (!['__global__', '__device__'].includes(qualifier)) this.fail('Only __global__ kernels and __device__ helper functions are accepted. Host CUDA APIs, structs, templates and PTX are not supported.', token);
      while (['inline', '__forceinline__'].includes(this.peek().value)) this.take();
      if(this.is('__launch_bounds__')){if(launchThreads!==null)this.fail('Duplicate launch bounds.');launchBounds();}
      if(launchThreads!==null&&qualifier!=='__global__')this.fail('Launch bounds apply only to kernels.',token);
      const result = this.type();
      if (result.pointer || result.shared || result.reference || result.external) this.fail('Function return pointers/references/shared/extern qualifiers are unsupported.');
      if(this.peek().forward||this.peek().expressionMacro)this.fail('Function-like macros are supported at call sites, not in function declarations.');
      const name = this.name();this.functionNames.add(name);let specializationArgument;
      if(templateKind==='specialization')specializationArgument=this.templateArgument();
      this.take('('); const params = [];
      if (!this.is(')')) do { const token = this.peek(), type = this.type(), name = this.name(),defaultValue=this.match('=')?this.expression(2):undefined;
        if(defaultValue!==undefined){const literal=defaultValue.kind==='unary'&&['+','-'].includes(defaultValue.op)?defaultValue.value:defaultValue;if(qualifier!=='__device__'||templateKind==='specialization')this.fail('Default arguments belong on primary device helper definitions only.',token);if(type.pointer||type.reference||(!['f32','i32','u32','bool'].includes(type.type)&&!String(type.type).startsWith('template:')))this.fail('Default arguments require scalar value parameters.',token);if(literal.kind!=='literal'&&!(literal===defaultValue&&literal.kind==='id'&&['true','false'].includes(literal.name)))this.fail('Default arguments support numeric or boolean literals with an optional numeric sign.',defaultValue.token);}
        else if(params.some(p=>p.defaultValue!==undefined))this.fail('Parameters after a default argument must also have defaults.',token);
        params.push({kind: 'param', token, name, ...type,...(defaultValue!==undefined?{defaultValue}:{})});
      } while (this.match(','));
      this.take(')'); const body = this.block();
      functions.push({kind: 'function', token, name, qualifier, result: result.type, params, body,launchThreads,templateParameter,templateParameters,templateKind,...(specializationArgument!==undefined?{specializationArgument}:{})});
    }
    if (!functions.some(f => f.qualifier === '__global__')) this.fail('No __global__ kernel was found.');
    return {kind: 'module', functions, constantGlobals,structs:[...this.structs.values()],typeTraits:[...this.typeTraits.values()], source: this.source};
  }
  block() { const token = this.take('{'), body = []; while (!this.is('}')) { if (this.peek().kind === 'eof') this.fail('Unclosed block.'); body.push(this.statement()); } this.take('}'); return {kind: 'block', token, body}; }
  initializer(){
    if(this.sharedWrappers.has(this.peek().value)){
      const token=this.take(),wrapper=this.sharedWrappers.get(token.value);this.take('<');const type=this.type();this.take('>');this.take('(');this.take(')');
      if(type.pointer||type.reference||type.shared||type.external||type.constant)this.fail('Shared wrapper arguments must be value types.',token);
      return {kind:'shared-conversion',token,target:type.type,conversions:wrapper.conversions};
    }
    if(!this.is('{'))return this.expression(2);
    const token=this.take('{'),items=[];
    while(!this.is('}')){items.push(this.expression(2));if(!this.match(','))break;}
    this.take('}');return {kind:'initializer',token,items};
  }
  declaration(semicolon = true) {
    const token = this.peek(), d = this.type(),declarations=[];
    do {const name=this.name(),dimensions=[];while(this.match('[')){dimensions.push(this.is(']')?null:this.expression(2));this.take(']');}const init=this.match('=')?this.initializer():null;declarations.push({kind:'decl',token,name,...d,dimensions,init});if(this.is(',')&&(d.pointer||d.reference))this.fail('Pointer/reference declaration lists are unsupported.');}while(this.match(','));
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
  expandExpressionMacro(macro,args,token){
    if(args.length!==macro.params.length)this.fail('Wrong argument count for expression macro '+macro.name,token);
    const parser=new Parser([...Object.entries(macro.defines).map(([name,value])=>'#define '+name+' '+value),macro.body].join('\n'));parser.functionNames=new Set(this.functionNames);
    if(parser.tokens.some(t=>t.kind==='word'&&macro.forbidden.includes(t.value)))this.fail('Nested or recursive expression macros are unsupported.',token);
    const expression=parser.expression();if(parser.peek().kind!=='eof')this.fail('Expression macro must contain one expression.',token);
    const copy=(node,substitute)=>{if(!node||typeof node!=='object')return node;if(node.kind&&++this.expandedMacroNodes>65536)this.fail('Expression macro expansion exceeds 65,536 AST nodes.',token);if(substitute&&node.kind==='id'&&macro.params.includes(node.name))return copy(args[macro.params.indexOf(node.name)],false);if(Array.isArray(node))return node.map(n=>copy(n,substitute));return Object.fromEntries(Object.entries(node).map(([key,value])=>[key,key==='token'?(substitute?token:value):copy(value,substitute)]));};return copy(expression,true);
  }
  templateCallAhead(){for(let offset=1;offset<=65;offset++){const token=this.peek(offset);if(token.value==='>')return this.peek(offset+1).value==='(';if(!['word','number'].includes(token.kind)&&!['+','-','*','/','%','(',')',','].includes(token.value))return false;}return false;}
  templateArgument(){this.take('<');const parts=[];while(!this.is('>')){const token=this.peek();if(parts.length>=64||(!['word','number'].includes(token.kind)&&!['+','-','*','/','%','(',')',','].includes(token.value)))this.fail('Template arguments support type lists or bounded integer arithmetic.',token);parts.push(this.take().value);}this.take('>');if(!parts.length)this.fail('Missing template argument.');return parts.join(' ');}
  unary() {
    const token = this.peek();
    if(this.match('static_cast')){this.take('<');const type=this.type();if(type.pointer||type.reference||type.shared||type.external)this.fail('static_cast supports value types only.',token);this.take('>');this.take('(');const value=this.expression();this.take(')');return {kind:'cast',token,target:type.type,value};}
    if (['+', '-', '!', '~', '&', '++', '--', '*'].includes(token.value)) { this.take(); return {kind: 'unary', token, op: token.value, value: this.unary(), prefix: true}; }
    if (this.is('(') && this.peek(2).value!=='(' && (TYPES.has(this.peek(1).value) || this.deferredType(this.peek(1).value) || this.peek(1).value==='typename' || this.typeTraits.has(this.peek(1).value) || this.templateTypeNames?.has(this.peek(1).value) || this.peek(1).value === 'const')) { this.take('('); const type = this.type(); if (type.pointer||type.reference) this.fail('Pointer/reference casts are unsupported.'); this.take(')'); return {kind: 'cast', token, target: type.type, value: this.unary()}; }
    let value;
    if (token.kind === 'number') { this.take(); value = {kind: 'literal', token, value: token.value}; }
    else if (this.match('(')) { value = this.expression(); this.take(')'); }
    else if (token.kind === 'word') { value = {kind: 'id', token, name: this.qualifiedName()}; }
    else this.fail('Expected an expression.', token);
    while (true) {
      if(value.kind==='id'&&this.functionNames.has(value.name)&&this.is('<')&&this.templateCallAhead())value.templateArgument=this.templateArgument();
      else if (this.match('[')) { const index = this.expression(); this.take(']'); value = {kind: 'index', token, base: value, index}; }
      else if (this.match('.')) { value = {kind: 'member', token, base: value, member: this.name()}; }
      else if (this.match('(')) { const args = []; if (!this.is(')')) do { args.push(this.expression(2)); } while (this.match(',')); this.take(')');
        if(value.kind==='id'&&value.token.forward){const f=value.token.forward;if(f.tooDeep)this.fail('Forwarding macro chains are limited to 32 calls.',value.token);if(f.recursive||f.numericTarget)this.fail('Recursive or non-function forwarding macro target is unsupported.',value.token);if(f.chain.some(m=>m.arity!==args.length))this.fail(`Wrong argument count for forwarding macro '${value.name}'.`,value.token);value={...value,name:f.target};}
        value = value.kind==='id'&&value.token.expressionMacro?this.expandExpressionMacro(value.token.expressionMacro,args,value.token):{kind: 'call', token, callee: value, args}; }
      else if (this.is('++') || this.is('--')) { value = {kind: 'unary', token, op: this.take().value, value, prefix: false}; }
      else break;
    }
    return value;
  }
}
export function parse(source, options = {}) { return new Parser(source, options.defines).parse(); }
