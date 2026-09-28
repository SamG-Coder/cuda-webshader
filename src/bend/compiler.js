import * as Bend from './vendor/bend.js';
import baseSource from './vendor/base.js';
import {compile, serializableArtifact} from '../compiler/compiler.js';

export const BEND_REVISION = '3378e6237ed431d17629efd36d24c96241815b7e';
export const VALUE = {U32:1, Nat:2, F32:3};
// Opcodes describe checked Bend terms, not a second source-language parser.
const OP = {literal:1, variable:2, lambda:3, apply:4, global:5, match:6, primitive:7, constructor:8, input:9, fail:10};
const primitiveNames = [
  'U32.add','U32.sub','U32.mul','U32.div','U32.mod','U32.is_eq','U32.is_lt',
  'U32.and','U32.or','U32.xor','U32.shl','U32.shr',
  'F32.add','F32.sub','F32.mul','F32.div','F32.is_eq','F32.is_lt',
  'F32.neg','F32.abs','F32.sqrt','F32.sin','F32.cos',
  'U32.shln','U32.shrn',
];
const strip = term => {while(term?.$ === 'Ann') term=term.x; return term;};
const annotation = term => term?.$ === 'Ann' ? term.T.v ?? term.T : null;
let checkedBase;

/** Check a single source against the pinned, unmodified upstream language core. */
export function checkBend(source) {
  if(typeof source !== 'string' || source.length > 100000) throw Error('Bend source must be at most 100,000 characters.');
  try {
    if(!checkedBase) {
      const book=Bend.book_nil();
      Bend.parse_book(book,'',baseSource,'',Object.create(null));
      for(const t of Object.values(book.tlds)) t.b=true;
      Bend.book_valid(book);
      checkedBase=book;
    }
    const book=Bend.book_nil();
    for(const [k,v] of Object.entries(checkedBase.tlds)) book.tlds[k]={...v};
    Object.assign(book.ctrs,checkedBase.ctrs);
    for(const [k,v] of Object.entries(checkedBase.tmps)) book.tmps[k]={...v};
    book.order.push(...checkedBase.order);
    // Spaces preserve upstream diagnostic source positions. There is no network loader.
    const text=source.replace(/^\s*import\s+Base\s*(?:#[^\n]*)?$/gm, line=>line.replace(/[^\r\n]/g,' '));
    if(/^\s*import\b/m.test(text)) throw Error('This tool supports import Base only; put local definitions in the editor.');
    Bend.parse_book(book,'',text,'',Object.create(null));
    for(const [name,t] of Object.entries(book.tlds)) {
      if(!t.b && (t.u || t.i)) throw Error(`Unsafe/foreign definition '${name}' is not supported by this runtime.`);
    }
    Bend.book_valid(book,checkedBase.order.length);
    if(book.hols) throw Error(`Bend source contains ${book.hols} unfilled laws or TODOs.`);
    return book;
  } catch(error) {
    throw Error(error?.$ === 'Err' ? Bend.err_show(error) : error.message ?? String(error));
  }
}

/** Lower the checked terms to a CUDA-resident closure/constructor machine. */
export function lowerBend(source,{entry='main'}={}) {
  const book=checkBend(source), definition=book.tlds[entry];
  if(!definition || definition.$!=='Def' || definition.b) throw Error(`Select a user definition as entry: '${entry}'.`);
  const signature=Bend.tele_unbind(book,definition.T);
  const scalarType=t=>{const n=Bend.term_wnf(book,t); const k=n.k; if(!['Ref','ADT'].includes(n.$)||!(k in VALUE)) throw Error('Entry arguments and result must be U32, Nat or F32.'); return k;};
  const parameters=signature.doms.map(([q,name,type])=>{
    if(q.$==='None') throw Error('Entry parameters cannot be erased or generic.');
    return {name,type:scalarType(type)};
  });
  if(parameters.length>8) throw Error('At most eight scalar entry parameters are supported.');
  const resultType=scalarType(signature.ret);
  const nodes=[[0,0,0,0]], globals=new Map(), constructors=new Map([['Zero',1],['Succ',2],['False',3],['True',4]]);
  const node=(op,a=0,b=0,c=0)=>{if(nodes.length>=4096) throw Error('Bend program exceeds the 4,096-node CUDA constant-memory limit.');nodes.push([op,a>>>0,b>>>0,c>>>0]);return nodes.length-1;};
  const tag=k=>{if(!constructors.has(k)) constructors.set(k,constructors.size+1);return constructors.get(k);};
  function quantities(k) {
    const ctr=book.ctrs[k];
    if(!ctr) throw Error(`Unknown constructor ${k}`);
    if(['U32','F32','WCon','WNil'].includes(k)) throw Error(`Bit representation pattern '${k}' is not supported; use numeric operations.`);
    const {doms}=Bend.tele_unbind(book,ctr.T);
    // Checked Ctr.x contains fields only; datatype parameters live in its type.
    return (ctr.n?doms.slice(-ctr.n):[]).map(([q])=>q.$!=='None');
  }
  function global(k) {
    const p=primitiveNames.indexOf(k);
    if(p>=0) return node(OP.primitive,p+1,([10,11].includes(p)||(p>=18&&p<=22))?1:2);
    if(globals.has(k)) return globals.get(k);
    const d=book.tlds[k];
    if(!d?.e || d.u || d.i) throw Error(`Runtime does not implement '${k}'.`);
    const at=node(OP.global); globals.set(k,at);
    nodes[at][1]=emit(d.e);
    return at;
  }
  function emit(annotated) {
    const type=annotation(annotated), t=strip(annotated);
    if(!t) throw Error('Missing checked Bend term.');
    switch(t.$) {
      case 'Ann': return emit(t.x);
      case 'Lit': {
        if(!(t.k in VALUE)) throw Error(`Literal ${t.k} is not supported by this runtime.`);
        if(!Number.isInteger(t.v)||t.v<0||t.v>0xffffffff) throw Error('This runtime represents Nat in 32 bits; larger literals are rejected.');
        return node(OP.literal,VALUE[t.k],t.v);
      }
      case 'Var': return node(OP.variable,t.i);
      case 'Ref': return global(t.k);
      case 'Lam': {
        const all=type && Bend.term_wnf(book,type);
        if(all?.$==='All' && all.q.$==='None') return emit(t.f);
        return node(OP.lambda,t.i,emit(t.f));
      }
      case 'App': {
        const ft=annotation(t.f), all=ft && Bend.term_wnf(book,ft);
        if(all?.$==='All' && all.q.$==='None') return emit(t.f);
        return node(OP.apply,emit(t.f),emit(t.x));
      }
      case 'Let': {
        let body=emit(t.f);
        for(let i=t.i.length-1;i>=0;i--) if(t.q[i].$!=='None') body=node(OP.lambda,t.i[i],body);
        for(let i=0;i<t.v.length;i++) if(t.q[i].$!=='None') body=node(OP.apply,body,emit(t.v[i]));
        return body;
      }
      case 'Ctr': {
        const qs=quantities(t.k), live=t.x.filter((_,i)=>qs[i]);
        if(live.length>8) throw Error('Constructors support at most eight live fields.');
        let body=node(OP.constructor,tag(t.k),live.length);
        for(const field of live) body=node(OP.apply,body,emit(field));
        return body;
      }
      case 'Mat': quantities(t.k); return node(OP.match,tag(t.k),emit(t.h),emit(t.m));
      case 'Efq': return node(OP.fail);
      case 'Rwt': return emit(t.f);
      case 'Rfl': return node(OP.constructor,tag('Unit'),0);
      default: throw Error(`Runtime does not yet support live Bend term '${t.$}'.`);
    }
  }
  let root=global(entry);
  parameters.forEach((p,i)=>{root=node(OP.apply,root,node(OP.input,VALUE[p.type],i));});
  return {nodes,root,parameters,resultType,entry,globals:[...globals.keys()],constructors:Object.fromEntries(constructors),source,revision:BEND_REVISION};
}

export function compileBend(source,runtimeSource,options={}) {
  const program=lowerBend(source,options);
  const constants=`// Bend 2 ${BEND_REVISION}; checked terms executed by the WebShader Bend runtime.\n#define BEND_ROOT ${program.root}u\n#define BEND_ARGS ${program.parameters.length}u\n#define BEND_RESULT ${VALUE[program.resultType]}u\n__constant__ unsigned int bend_code[${program.nodes.length*4}] = {\n${program.nodes.map(n=>'  '+n.map(x=>x+'u').join(', ')).join(',\n')}\n};\n`;
  const cuda=constants+runtimeSource;
  const artifact=serializableArtifact(compile(cuda,{entry:'bend_run',workgroupSize:[64,1,1]}));
  return {program,cuda,artifact,verification:{checker:'upstream TypeScript checker',formalVerdict:false,translationProven:false}};
}

export {Bend};
