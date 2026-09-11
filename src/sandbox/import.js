// Extract device functions from desktop .cu files without interpreting host code.
// Whitespace padding preserves original source locations for Monaco diagnostics.
import {forwardingMacro} from '../compiler/macros.js';
export function kernelSource(source){
 const masked=source.replace(/\/\*[\s\S]*?\*\/|\/\/[^\n]*|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'/g,text=>text.replace(/[^\n]/g,' '));
 if(!/^\s*#\s*include\b/m.test(masked)&&!masked.includes('<<<')&&!/\bmain\s*\(/.test(masked))return {source,extracted:false};
 const spans=[],pattern=/\b(?:template\s*<[^>]*>\s*)?(?:__launch_bounds__\s*\([^)]*\)\s*)?__(?:global|device)__\s+[^;{}]+?\([^;{}]*\)\s*\{/g;let match;
 while((match=pattern.exec(masked))){let depth=1,end=pattern.lastIndex;for(;end<masked.length&&depth;end++){if(masked[end]==='{')depth++;else if(masked[end]==='}')depth--;}if(depth)throw Error('Unclosed CUDA device function.');spans.push([match.index,end]);pattern.lastIndex=end;}
 if(!spans.length)throw Error('No standalone __global__ / __device__ functions found. Templates, classes and host-only CUDA files need a supported kernel entry.');
 const defines=/^\s*#\s*define\s+\w+\s+[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?[fFuU]?\s*$/gm;
 while((match=defines.exec(masked)))spans.push([match.index,match.index+match[0].length]);
 const forwards=/^[ \t]*#[ \t]*define[^\n]*/gm;
 while((match=forwards.exec(masked)))if(forwardingMacro(match[0].trim()))spans.push([match.index,match.index+match[0].length]);
 const aliases=/\bnamespace\s+\w+\s*=\s*cooperative_groups\s*;/g;
 while((match=aliases.exec(masked)))spans.push([match.index,match.index+match[0].length]);
 const chars=source.replace(/[^\n]/g,' ').split('');for(const [start,end]of spans)for(let i=start;i<end;i++)chars[i]=source[i];
 return {source:chars.join(''),extracted:true,functions:spans.filter(([start])=>/^(?:__|template\b)/.test(masked.slice(start))).length};
}
