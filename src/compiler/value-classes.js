// Parse plain public value classes into storage records and device helpers.
// Constructors return a value; const methods receive a read-only value copy.
export function parseValueClass(p) {
  const token=p.take('class'),name=p.name();
  if(p.structs.size>=64)p.fail('At most 64 value record types are supported.',token);
  if(p.structs.has(name)||p.typeAliases.has(name)||p.typeTraits.has(name))p.fail('Duplicate value class name.',token);
  p.take('{');
  const record={name,type:'cw_struct_'+name,fields:[],methods:[],constructors:[],token,valueClass:true};
  p.structs.set(name,record);
  const functions=[];let access='private';
  while(!p.is('}')) {
    if(['public','private','protected'].includes(p.peek().value)){access=p.take().value;p.take(':');continue;}
    if(access!=='public')p.fail('Value classes currently require public fields and methods.');
    const start=p.peek();let device=false;
    while(['__host__','__device__','inline','__forceinline__'].includes(p.peek().value)){if(p.take().value==='__device__')device=true;}
    const constructor=p.is(name)&&p.peek(1).value==='(';
    const spec=constructor?{type:record.type}:p.type();
    let member=p.name(),operator=null;
    if(member==='operator'){
      operator=p.take().value;
      if(operator==='['){p.take(']');operator='[]';}
      if(!['+','-','[]'].includes(operator))p.fail('This class operator overload is not yet supported.',start);
      member='operator'+operator;
    }
    if(p.is('(')) {
      const selfReference=spec.reference&&spec.constant&&spec.type===record.type;
      if(selfReference&&!['+','-'].includes(operator))p.fail('Const self-reference returns currently require a unary class operator.',start);
      if(!device||spec.pointer||spec.reference&&!selfReference||spec.shared||spec.external)p.fail('Value-class methods require value returns or a const self reference.',start);
      p.take('(');const params=[];
      if(!p.is(')'))do{const t=p.peek(),type=p.type(),param=p.name();if(type.pointer||type.reference||type.shared||type.external)p.fail('Value-class method parameters currently require values.',t);params.push({kind:'param',token:t,name:param,...type});}while(p.match(','));
      p.take(')');const constant=!!p.match('const');
      if(constructor&&constant)p.fail('Constructors cannot be const.',start);
      if(functions.length>=128)p.fail('At most 128 methods per value class are supported.',start);
      if(!constructor&&!constant)p.fail('Mutable class methods are not yet supported.',start);
      if(operator&&params.length!==(operator==='[]'?1:0))p.fail('Supported class operators are unary +/-, or single-index access.',start);
      const body=p.block(),helper=constructor?'cw_ctor_'+name:'cw_method_'+name+'_'+(operator?{'+':'positive','-':'negative','[]':'index'}[operator]:member);
      if(selfReference){const ret=body.body[0];if(body.body.length!==1||ret?.kind!=='return'||ret.value?.kind!=='unary'||ret.value.op!=='*'||ret.value.value?.kind!=='id'||ret.value.value.name!=='this')p.fail('Const reference methods currently require exactly return *this.',start);}
      if(constructor)record.constructors.push(helper);else record.methods.push({name:member,helper,selfReference});
      functions.push({kind:'function',token:start,name:helper,qualifier:'__device__',result:spec.type,params,body,classOwner:name,classConstructor:constructor,classMethod:constructor?null:member,classSelfReference:selfReference});
      p.match(';');
    } else {
      if(device||spec.pointer||spec.reference||spec.shared||spec.external||spec.constant||spec.type.startsWith('cw_struct_')||['void','texture3d','surface2d','thread-block','cw_extent','cw_size64'].includes(spec.type))p.fail('Value-class fields require plain scalar/vector values.',start);
      const dimensions=[];while(p.match('[')){dimensions.push(p.expression(2));p.take(']');}p.take(';');
      if(dimensions.length>1||record.fields.length>=64||record.fields.some(f=>f.name===member))p.fail('Invalid or duplicate value-class field.',start);
      record.fields.push({name:member,type:spec.type,dimensions,token:start});
    }
  }
  p.take('}');p.take(';');
  if(!record.fields.length)p.fail('Value classes require at least one field.',token);
  const fields=new Set(record.fields.map(f=>f.name));
  for(const fn of functions) {
    const self='cw_object_'+name,locals=new Set(fn.params.map(v=>v.name));
    if(locals.has(self))p.fail('Reserved value-class receiver parameter name.',fn.token);
    const inspect=n=>{if(!n||typeof n!=='object')return;if(n.kind==='decl'&&(fields.has(n.name)||n.name===self))p.fail('Value-class method locals cannot shadow fields or generated receiver storage.',n.token);if(n.kind==='id'&&n.name===self)p.fail('Reserved value-class receiver name.',n.token);for(const [key,v]of Object.entries(n))if(key!=='token')Array.isArray(v)?v.forEach(inspect):inspect(v);};inspect(fn.body);
    const rewrite=n=>{
      if(!n||typeof n!=='object')return n;
      if(fn.classSelfReference&&n.kind==='unary'&&n.op==='*'&&n.value?.name==='this')return {kind:'id',token:n.token,name:self};
      if(n.kind==='id'&&fields.has(n.name)&&!locals.has(n.name))return {kind:'member',token:n.token,base:{kind:'id',token:n.token,name:self},member:n.name};
      if(fn.classConstructor&&n.kind==='return'){if(n.value)p.fail('Constructors cannot return an explicit value.',n.token);n.value={kind:'id',name:self,token:n.token};return n;}
      for(const [key,v]of Object.entries(n))if(key!=='token')n[key]=Array.isArray(v)?v.map(rewrite):rewrite(v);
      return n;
    };fn.body=rewrite(fn.body);
    if(fn.classConstructor){
      fn.body.body.unshift({kind:'decl',token:fn.token,name:self,type:record.type,pointer:false,reference:false,constant:false,shared:false,external:false,dimensions:[],init:null});
      fn.body.body.push({kind:'return',token:fn.token,value:{kind:'id',name:self,token:fn.token}});
    } else fn.params.unshift({kind:'param',token:fn.token,name:self,type:record.type,constant:true,pointer:false,reference:false,shared:false,external:false});
    p.functionNames.add(fn.name);
  }
  return functions;
}
