import {tokens,pairs,apply,trimWgsl} from './wgsl-trim.js';
// Exact comparison of f32 values, including subnormals on FTZ hardware.
// No arithmetic or approximate replacement of actual double calculations.
const helpers={
 lt:`fn cw_trim_f32_lt(a:f32,b:f32)->bool {
 let x=bitcast<u32>(a); let y=bitcast<u32>(b);
 let ax=x&2147483647u; let ay=y&2147483647u;
 if(ax>2139095040u || ay>2139095040u || x==y || (ax==0u && ay==0u)){return false;}
 if((x>>31u)!=(y>>31u)){return (x>>31u)!=0u;}
 return select((x < y),(x > y),(x>>31u)!=0u);
}`,
 eq:`fn cw_trim_f32_eq(a:f32,b:f32)->bool {
 let x=bitcast<u32>(a); let y=bitcast<u32>(b);
 let ax=x&2147483647u; let ay=y&2147483647u;
 return ax<=2139095040u && ay<=2139095040u && (x==y || (ax==0u && ay==0u));
}`
};
export function optimizeFloatComparisons(input){
 let source=input.wgsl,replacements=0;const used=new Set();
 // Reserve our names; an unexpected collision is a refusal, not a rewrite.
 if(/\bcw_trim_f32_(?:lt|eq)\b/.test(source))return {artifact:input,replacements:0};
 const t=tokens(source),match=pairs(t);
 const slice=(a,b)=>source.slice(t[a].start,t[b-1].end);
 const lifted=(a,b)=>{
  while(t[a]?.value==='('&&match.get(a)===b-1){a++;b--;}
  if(t[a]?.value==='cw_d_from_f32'&&t[a+1]?.value==='('&&match.get(a+1)===b-1)return slice(a+2,b-1);
  let negate=false;
  if(t[a]?.value==='cw_d_neg'&&t[a+1]?.value==='('&&match.get(a+1)===b-1){negate=true;a+=2;b--;}
  const literal=/^vec2<u32>\((\d+)u,(\d+)u\)$/.exec(t.slice(a,b).map(x=>x.value).join(''));
  if(!literal)return null;
  const bytes=new ArrayBuffer(8),view=new DataView(bytes);view.setUint32(0,Number(literal[1]),true);view.setUint32(4,(Number(literal[2])^(negate?0x80000000:0))>>>0,true);
  const value=view.getFloat64(0,true);if(!Number.isFinite(value)||!Object.is(Math.fround(value),value))return null;
  view.setFloat32(0,value,true);return `bitcast<f32>(${view.getUint32(0,true)}u)`;
 };
 const edits=[];
 for(let i=0;i<t.length-3;i++){
  const op=t[i].value==='cw_d_lt'?'lt':t[i].value==='cw_d_eq'?'eq':null;
  if(!op||t[i-1]?.value==='fn'||t[i+1].value!=='(')continue;
  const end=match.get(i+1);let comma=-1;
  for(let j=i+2;j<end;j++){if(t[j].value==='('){j=match.get(j);continue;}if(t[j].value===','){if(comma!==-1){comma=-1;break;}comma=j;}}
  if(comma<0)continue;
  const left=lifted(i+2,comma),right=lifted(comma+1,end);if(left===null||right===null)continue;
  edits.push({start:t[i].start,end:t[end].end,text:`cw_trim_f32_${op}(${left}, ${right})`});used.add(op);replacements++;i=end;
 }
 source=apply(source,edits)+'\n'+[...used].map(op=>helpers[op]).join('\n');
 return {artifact:trimWgsl({...input,wgsl:source}).artifact,replacements};
}
