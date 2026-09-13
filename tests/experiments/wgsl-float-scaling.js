import {tokens,pairs,apply,trimWgsl} from './wgsl-trim.js';
// A promoted nonzero finite f32 has binary64 exponent 874..1150. Shifts
// -512..512 stay normal in binary64, so no significand rounding is necessary.
const helper=`fn cw_trim_scale_f32(a:f32,shift:i32)->cw_f64 {
 let d=cw_d_from_f32(a);
 if(cw_d_nan(d)){return vec2<u32>(0u,2146959360u);}
 let exponent=(d.y>>20u)&2047u;
 if(exponent==0u || exponent==2047u){return d;}
 return vec2<u32>(d.x,(d.y&2148532223u)|(u32(i32(exponent)+shift)<<20u));
}`;
export function optimizeFloatScaling(input){
 const source=input.wgsl;if(/\bcw_trim_scale_f32\b/.test(source))return {artifact:input,replacements:0};
 const t=tokens(source),match=pairs(t),edits=[];
 const scalar=(a,b)=>t[a]?.value==='cw_d_from_f32'&&t[a+1]?.value==='('&&match.get(a+1)===b-1?source.slice(t[a+2].start,t[b-2].end):null;
 const power=(a,b)=>{
  const m=/^vec2<u32>\(0u,(\d+)u\)$/.exec(t.slice(a,b).map(x=>x.value).join(''));if(!m)return null;
  const high=Number(m[1]);if(!Number.isInteger(high)||high<0||high>0x7fffffff||(high&0xfffff)!==0)return null;
  const shift=(high>>>20)-1023;return shift>=-512&&shift<=512?shift:null;
 };
 for(let i=0;i<t.length-3;i++){
  if(t[i].value!=='cw_d_mul'||t[i-1]?.value==='fn'||t[i+1].value!=='(')continue;
  const end=match.get(i+1);let comma=-1;for(let j=i+2;j<end;j++){if(t[j].value==='('){j=match.get(j);continue;}if(t[j].value===','){comma=j;break;}}
  if(comma<0)continue;
  let value=scalar(i+2,comma),shift=power(comma+1,end);if(value===null||shift===null){value=scalar(comma+1,end);shift=power(i+2,comma);}
  if(value===null||shift===null)continue;
  edits.push({start:t[i].start,end:t[end].end,text:`cw_trim_scale_f32(${value}, ${shift}i)`});i=end;
 }
 if(!edits.length)return {artifact:input,replacements:0};
 return {artifact:trimWgsl({...input,wgsl:apply(source,edits)+'\n'+helper}).artifact,replacements:edits.length};
}
