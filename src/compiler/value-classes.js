// Parse plain public value classes into storage records and device helpers.
// Constructors return values; mutable methods receive a reference to the receiver.
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
      if(!['+','-','[]','+=','-=','*=','/='].includes(operator))p.fail('This class operator overload is not yet supported.',start);
      member='operator'+operator;
    }
    if(p.is('(')) {
      const mutableSelf=spec.reference&&!spec.constant&&spec.type===record.type;
      const selfReference=spec.reference&&spec.constant&&spec.type===record.type;
      if(mutableSelf&&!['+=','-=','*=','/='].includes(operator))p.fail('Mutable self references currently require compound operators.',start);
      const indexedReference=spec.reference&&!spec.constant&&operator==='[]';
      if(selfReference&&!['+','-'].includes(operator))p.fail('Const self-reference returns currently require a unary class operator.',start);
      if(!device||spec.pointer||spec.reference&&!selfReference&&!indexedReference&&!mutableSelf||spec.shared||spec.external)p.fail('Value-class methods require value returns or supported receiver/array references.',start);
      p.take('(');const params=[];
      if(!p.is(')'))do{const t=p.peek(),type=p.type(),param=p.name();if(type.pointer||(type.reference&&!type.constant)||type.shared||type.external)p.fail('Value-class method parameters require values or const references.',t);params.push({kind:'param',token:t,name:param,...type});}while(p.match(','));
      p.take(')');const constant=!!p.match('const');
      if(constructor&&constant)p.fail('Constructors cannot be const.',start);
      if(functions.length>=128)p.fail('At most 128 methods per value class are supported.',start);
      if(!constructor&&!constant&&!indexedReference&&!mutableSelf&&spec.type!=='void')p.fail('Mutable methods currently require void or self-reference returns.',start);
      if(operator&&params.length!==(['[]','+=','-=','*=','/='].includes(operator)?1:0))p.fail('Class operators require zero unary arguments or one index/compound argument.',start);
      const initializers=[];
      if(p.match(':')){if(!constructor)p.fail('Member initializer lists require a constructor.',start);do{const token=p.peek(),field=p.name();p.take('(');const args=[];if(!p.is(')'))do{args.push(p.expression(2));}while(p.match(','));p.take(')');if(initializers.some(i=>i.field===field))p.fail('Duplicate member initializer.',token);initializers.push({field,args,token});}while(p.match(','));}
      const declaration=!!p.match(';'),body=declaration?null:p.block(),helper=constructor?'cw_ctor_'+name:'cw_method_'+name+'_'+(operator?{'+':'positive','-':'negative','[]':'index','+=':'add_assign','-=':'subtract_assign','*=':'multiply_assign','/=':'divide_assign'}[operator]:member);
      if(indexedReference){
        if(!body)p.fail('Reference index accessors require an inline definition.',start);
        const ret=body.body[0],value=ret?.value;
        if(constant||body.body.length!==1||ret?.kind!=='return'||value?.kind!=='index'||value.base?.kind!=='id'||value.index?.kind!=='id'||value.index.name!==params[0].name||!['i32','u32'].includes(params[0].type))p.fail('Reference indexing requires exactly return field[index] with one integer index.',start);
        record.methods.push({name:member,indexedReference:true,field:value.base.name,result:spec.type,token:start});p.match(';');continue;
      }
      if(constructor)record.constructors.push(helper);else record.methods.push({name:member,helper,selfReference,mutableSelf});
      functions.push({kind:'function',token:start,name:helper,qualifier:'__device__',result:spec.type,params,body,classOwner:name,classConstructor:constructor,classMethod:constructor?null:member,classSelfReference:selfReference,classMutableSelf:mutableSelf,classConstant:constant,classDeclaration:declaration,classResultSpec:spec,classInitializers:initializers});
      p.match(';');
    } else {
      if(device||spec.pointer||spec.reference||spec.shared||spec.external||spec.constant||['void','texture3d','surface2d','thread-block','cw_extent','cw_size64'].includes(spec.type))p.fail('Value-class fields require plain scalar/vector values.',start);
      const dimensions=[];while(p.match('[')){dimensions.push(p.expression(2));p.take(']');}p.take(';');
      if(dimensions.length>1||record.fields.length>=64||record.fields.some(f=>f.name===member))p.fail('Invalid or duplicate value-class field.',start);
      if(spec.type.startsWith('cw_struct_')){const nested=[...p.structs.values()].find(r=>r.type===spec.type);if(!nested?.complete||dimensions.length)p.fail('Nested class fields require a previously completed class and no array dimensions.',start);}
      record.fields.push({name:member,type:spec.type,dimensions,token:start});
    }
  }
  p.take('}');p.take(';');
  if(!record.fields.length)p.fail('Value classes require at least one field.',token);
  for(const method of record.methods.filter(m=>m.indexedReference)){const field=record.fields.find(f=>f.name===method.field);if(!field||field.dimensions.length!==1||field.type!==method.result)p.fail('Reference indexing must return an element of a matching array field.',method.token);}
  if(!functions.some(f=>f.classConstructor)&&record.fields.some(f=>f.type.startsWith('cw_struct_'))){
    const helper='cw_ctor_'+name;record.constructors.push(helper);
    functions.push({kind:'function',token,name:helper,qualifier:'__device__',result:record.type,params:[],body:{kind:'block',token,body:[]},classOwner:name,classConstructor:true,classMethod:null,classConstant:false});
  }
  record.complete=true;
  (p.valueClassFunctions??=[]).push(...functions);
  return functions;
}

export function finishValueClasses(p) {
 for(const fn of p.valueClassFunctions||[]) {
    const record=p.structs.get(fn.classOwner),name=record.name,fields=new Set(record.fields.map(f=>f.name));
    fn.classOriginalArity=fn.params.length;
    if(!fn.body)p.fail('Missing definition for class method '+fn.classMethod,fn.token);
    if(fn.classSelfReference){const ret=fn.body.body[0];if(fn.body.body.length!==1||ret?.kind!=='return'||ret.value?.kind!=='unary'||ret.value.op!=='*'||ret.value.value?.kind!=='id'||ret.value.value.name!=='this')p.fail('Const reference methods currently require exactly return *this.',fn.token);}

    if(fn.classMutableSelf){const check=n=>{if(!n||typeof n!=='object')return;if(n.kind==='return'&&(n.value?.kind!=='unary'||n.value.op!=='*'||n.value.value?.name!=='this'))p.fail('Mutable operators must return *this.',n.token);for(const [k,v]of Object.entries(n))if(k!=='token')Array.isArray(v)?v.forEach(check):check(v);};check(fn.body);if(fn.body.body.at(-1)?.kind!=='return')p.fail('Mutable operators require a final return *this.',fn.token);}
    const self='cw_object_'+name,locals=new Set(fn.params.map(v=>v.name));
    if(locals.has(self))p.fail('Reserved value-class receiver parameter name.',fn.token);
    const inspect=n=>{if(!n||typeof n!=='object')return;if(n.kind==='decl'&&(fields.has(n.name)||n.name===self))p.fail('Value-class method locals cannot shadow fields or generated receiver storage.',n.token);if(n.kind==='id'&&n.name===self)p.fail('Reserved value-class receiver name.',n.token);for(const [key,v]of Object.entries(n))if(key!=='token')Array.isArray(v)?v.forEach(inspect):inspect(v);};inspect(fn.body);for(const init of fn.classInitializers||[])init.args.forEach(inspect);
    const rewrite=n=>{
      if(!n||typeof n!=='object')return n;
      if((fn.classSelfReference||fn.classMutableSelf)&&n.kind==='unary'&&n.op==='*'&&n.value?.name==='this')return {kind:'id',token:n.token,name:self};
      if(n.kind==='id'&&fields.has(n.name)&&!locals.has(n.name))return {kind:'member',token:n.token,base:{kind:'id',token:n.token,name:self},member:n.name};
      if(fn.classConstructor&&n.kind==='return'){if(n.value)p.fail('Constructors cannot return an explicit value.',n.token);n.value={kind:'id',name:self,token:n.token};return n;}
      for(const [key,v]of Object.entries(n))if(key!=='token')n[key]=Array.isArray(v)?v.map(rewrite):rewrite(v);
      return n;
    };fn.body=rewrite(fn.body);
    if(fn.classConstructor){
      fn.body.body.unshift({kind:'decl',token:fn.token,name:self,type:record.type,pointer:false,reference:false,constant:false,shared:false,external:false,dimensions:[],init:null});
      const nestedInitializers=[],initializers=fn.classInitializers||[];
      for(const init of initializers)if(!record.fields.some(f=>f.name===init.field))p.fail('Unknown member initializer '+init.field,init.token);
      for(const field of record.fields){
        const nested=[...p.structs.values()].find(r=>r.type===field.type),init=initializers.find(i=>i.field===field.name);let value;
        if(init){
          if(field.dimensions.length)p.fail('Array member initializers are unsupported.',init.token);
          const args=init.args.map(rewrite);
          if(nested)value={kind:'call',callee:{kind:'id',name:nested.name,token:init.token},args,token:init.token,classMemberInit:true};
          else{if(args.length!==1)p.fail('Scalar/vector member initializers require one value.',init.token);value=args[0];}
        }else if(nested){
          const constructors=(p.valueClassFunctions||[]).filter(f=>f.classOwner===nested.name&&f.classConstructor);
          if(!constructors.length)continue;
          if(!constructors.some(f=>(f.classOriginalArity??f.params.length)===0))p.fail('Nested class members require a default constructor or explicit initializer.',field.token);
          value={kind:'call',callee:{kind:'id',name:'cw_ctor_'+nested.name,token:field.token},args:[],token:field.token};
        }else continue;
        nestedInitializers.push({kind:'expr',token:field.token,value:{kind:'assign',op:'=',token:field.token,left:{kind:'member',base:{kind:'id',name:self,token:field.token},member:field.name,token:field.token},right:value}});
      }
      fn.body.body.splice(1,0,...nestedInitializers);
      fn.body.body.push({kind:'return',token:fn.token,value:{kind:'id',name:self,token:fn.token}});
    } else fn.params.unshift({kind:'param',token:fn.token,name:self,type:record.type,constant:fn.classConstant,pointer:false,reference:!fn.classConstant,shared:false,external:false});
    p.functionNames.add(fn.name);
  }
}

export function parseExternalValueMethod(p,result,qualifier,token) {
 const owner=p.name();p.take('::');let member=p.name();
 if(member==='operator')member+=p.take().value;
 p.take('(');const params=[];
 if(!p.is(')'))do{const token=p.peek(),type=p.type(),name=p.name();params.push({kind:'param',token,name,...type});}while(p.match(','));
 p.take(')');const constant=!!p.match('const');
 const signature=values=>JSON.stringify(values.map(v=>[v.type,!!v.reference,!!v.pointer,!!v.constant]));
 const fn=(p.valueClassFunctions||[]).find(f=>f.classOwner===owner&&f.classMethod===member&&f.classConstant===constant&&signature(f.params)===signature(params));
 if(qualifier!=='__device__'||!fn||fn.body||JSON.stringify(fn.classResultSpec)!==JSON.stringify(result))p.fail('Class method definition must match one previously declared signature.',token);
 fn.params=params;fn.body=p.block();
}
