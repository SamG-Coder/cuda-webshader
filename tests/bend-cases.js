import {examples} from '../src/bend/examples.js';
export const bendCases=Object.entries(examples).map(([name,e])=>({name,...e,rows:JSON.parse(e.input),expected:JSON.parse(e.input).map(x=>name==='pow2'?2**x[0]:name==='tree'?3*2**x[0]:name==='closure'?Math.fround(x[0]+x[1]):x[0])}));
const binaryRows=[[0,0],[0xffffffff,1],[0x80000000,0xffffffff],[123456789,987654321],[1,32],[0xffffffff,0]];
for(const [op,fn] of Object.entries({add:(x,y)=>(x+y)>>>0,sub:(x,y)=>(x-y)>>>0,mul:(x,y)=>Math.imul(x,y)>>>0,div:(x,y)=>y?Math.floor(x/y):0,mod:(x,y)=>y?x%y:x,and:(x,y)=>(x&y)>>>0,or:(x,y)=>(x|y)>>>0,xor:(x,y)=>(x^y)>>>0})) {
  bendCases.push({name:'u32-'+op,entry:'main',source:`import Base\ndef main(x: U32, y: U32) -> U32:\n  U32.${op}(x,y)\n`,rows:binaryRows,expected:binaryRows.map(([x,y])=>fn(x,y))});
}
for(const [op,fn] of Object.entries({shl:x=>(x<<1)>>>0,shr:x=>x>>>1})) {
  const rows=[[0],[1],[0xffffffff],[0x80000000]];
  bendCases.push({name:'u32-'+op,entry:'main',source:`import Base\ndef main(x: U32) -> U32:\n  U32.${op}(x)\n`,rows,expected:rows.map(([x])=>fn(x))});
}
for(const [op,fn] of Object.entries({shln:(x,y)=>y>=32?0:(x<<y)>>>0,shrn:(x,y)=>y>=32?0:x>>>y})) {
  const rows=[[0xffffffff,0],[0xffffffff,31],[0xffffffff,32],[7,33]];
  bendCases.push({name:'u32-'+op,entry:'main',source:`import Base\ndef main(x: U32, y: Nat) -> U32:\n  U32.${op}(x,y)\n`,rows,expected:rows.map(([x,y])=>fn(x,y))});
}
bendCases.push({name:'boolean-match',entry:'main',source:`import Base
def choose(b: Bool, a: U32, c: U32) -> U32:
  match b:
    case True{}: a
    case False{}: c
def main(+a: U32, +b: U32) -> U32:
  choose(U32.is_lt(a,b),a,b)
`,rows:binaryRows,expected:binaryRows.map(([x,y])=>Math.min(x,y))});
bendCases.push({name:'nested-closures',entry:'main',source:`import Base
def combine(a: U32) -> U32 -> U32 -> U32:
  b => c => (a + b * c : U32)
def main(a: U32, b: U32, c: U32) -> U32:
  combine(a)(b)(c)
`,rows:[[3,5,7],[0xffffffff,9,8]],expected:[38,71]});
bendCases.push({name:'partial-primitive',entry:'main',source:`import Base
def main(a: U32, b: U32) -> U32:
  add = U32.add(a)
  add(b)
`,rows:binaryRows,expected:binaryRows.map(([x,y])=>(x+y)>>>0)});
bendCases.push({name:'generic-list',entry:'main',source:`import Base
def sum(xs: List<U32>) -> U32:
  match xs:
    case []: 0
    case x <> rest: (x + sum(rest) : U32)
def main(a: U32, b: U32) -> U32:
  sum([a,b,7])
`,rows:[[2,3],[0xffffffff,0]],expected:[12,6]});
bendCases.push({name:'nat-successor',entry:'main',source:`import Base
def main(n: Nat) -> Nat:
  1n+n
`,rows:[[0],[12],[4294967294]],expected:[1,13,4294967295]});
