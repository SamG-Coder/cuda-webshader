// Extract device functions from desktop .cu files without interpreting host code.
// Whitespace padding preserves original source locations for Monaco diagnostics.
import {forwardingMacro,expressionMacro} from '../compiler/macros.js';
export function kernelSource(source){
 const masked=source.replace(/\/\*[\s\S]*?\*\/|\/\/[^\n]*|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'/g,text=>text.replace(/[^\n]/g,' '));
 if(!/^\s*#\s*include\b/m.test(masked)&&!masked.includes('<<<')&&!/\bmain\s*\(/.test(masked))return {source,extracted:false};
 const spans=[],pattern=/\b(?:template\s*<[^>]*>\s*)?(?:static\s+)?(?:__launch_bounds__\s*\([^)]*\)\s*)?__(?:global|device)__\s+[^;{}]+?\([^;{}]*\)\s*\{/g;let match;
 while((match=pattern.exec(masked))){let depth=1,end=pattern.lastIndex;for(;end<masked.length&&depth;end++){if(masked[end]==='{')depth++;else if(masked[end]==='}')depth--;}if(depth)throw Error('Unclosed CUDA device function.');spans.push([match.index,end]);pattern.lastIndex=end;}
 if(!spans.length)throw Error('No standalone __global__ / __device__ functions found. Templates, classes and host-only CUDA files need a supported kernel entry.');
 const defines=/^\s*#\s*define\s+\w+\s+[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?[fFuU]?\s*$/gm;
 while((match=defines.exec(masked)))spans.push([match.index,match.index+match[0].length]);
 const forwards=/^[ \t]*#[ \t]*define[^\n]*/gm;
 while((match=forwards.exec(masked)))if(forwardingMacro(match[0].trim())||expressionMacro(match[0].trim())||/^\s*#\s*define\s+\w+\s+\([^#;{}]*\)\s*$/.test(match[0]))spans.push([match.index,match.index+match[0].length]);
 const aliases=/\bnamespace\s+\w+\s*=\s*cooperative_groups\s*;/g;
 while((match=aliases.exec(masked)))spans.push([match.index,match.index+match[0].length]);
 const wrappers=[],structures=/\btemplate\s*<[^>]*>\s*struct\s+\w+(?:\s*<[^>]*>)?\s*\{/g;
 while((match=structures.exec(masked))){let depth=1,end=structures.lastIndex;for(;end<masked.length&&depth;end++){if(masked[end]==='{')depth++;else if(masked[end]==='}')depth--;}
   if(!depth&&(/\boperator\s+(?:const\s+)?\w+\s*\*/.test(masked.slice(structures.lastIndex,end))||/\bstatic\b/.test(masked.slice(structures.lastIndex,end))||!masked.slice(structures.lastIndex,end-1).trim())){while(/\s/.test(masked[end]||'')&&end<masked.length)end++;if(masked[end]===';'){wrappers.push([match.index,end+1]);spans.push([match.index,end+1]);}}structures.lastIndex=end;
 }
 const traits=/\btemplate\s*<[^>]*>\s*struct\s+\w+(?:\s*<[^>]*>)?\s*\{([^{}]*)\}\s*;/g;
 while((match=traits.exec(masked)))if(/^(?:\s*typedef\s+(?:unsigned\s+)?\w+\s+\w+\s*;)+\s*$/.test(match[1]))spans.push([match.index,match.index+match[0].length]);
 const declarations=[];
 const plainStructs=/\b(?:typedef\s+)?struct\s*(?:\w+\s*)?\{[^{}]*\}\s*(?:\w+\s*)?;/g;
 while((match=plainStructs.exec(masked))){const names=/^(?:typedef\s+)?struct\s*(\w+)?\s*\{[\s\S]*\}\s*(\w+)?\s*;$/.exec(match[0]);declarations.push({names:[names?.[1],names?.[2]].filter(Boolean),range:[match.index,match.index+match[0].length]});}
 const valueAliases=/\btypedef\s+(?:unsigned\s+)?\w+\s+\w+\s*;/g;
 while((match=valueAliases.exec(masked))){const name=/([A-Za-z_]\w*)\s*;$/.exec(match[0])[1];declarations.push({names:[name],range:[match.index,match.index+match[0].length]});}
 const sharedDeclarations=/\b(?:extern\s+)?__shared__\s+[^;{}]+;/g;
 while((match=sharedDeclarations.exec(masked)))spans.push([match.index,match.index+match[0].length]);
 const constants=/\b__constant__\b[^;]*;/g;
 while((match=constants.exec(masked)))spans.push([match.index,match.index+match[0].length]);
 // Keep declaration dependencies of the extracted device code, not unrelated
 // host resource structs. Iterate so aliases and struct fields retain their types.
 const selected=new Set();let changed=true;
 while(changed){changed=false;const used=new Set(spans.flatMap(([a,b])=>masked.slice(a,b).match(/[A-Za-z_]\w*/g)||[]));for(const declaration of declarations)if(!selected.has(declaration)&&declaration.names.some(name=>used.has(name))){selected.add(declaration);spans.push(declaration.range);changed=true;}}
 // Keep conditional regions that contain extracted declarations. Without their
 // directives, mutually exclusive device implementations become simultaneous.
 const conditionalStack=[],conditionalGroups=[],directives=/^[ \t]*#[ \t]*(if|ifdef|ifndef|elif|else|endif)\b[^\n]*/gm;
 while((match=directives.exec(masked))){const kind=match[1],range=[match.index,match.index+match[0].length];if(['if','ifdef','ifndef'].includes(kind))conditionalStack.push({start:match.index,directives:[range]});else{const frame=conditionalStack.at(-1);if(!frame)throw Error('Unmatched conditional directive in CUDA source.');frame.directives.push(range);if(kind==='endif'){conditionalStack.pop();conditionalGroups.push({...frame,end:range[1]});}}}
 if(conditionalStack.length)throw Error('Unclosed conditional directive in CUDA source.');
 for(const group of conditionalGroups)if(spans.some(([a,b])=>a<group.end&&b>group.start))spans.push(...group.directives);
 const chars=source.replace(/[^\n]/g,' ').split('');for(const [start,end]of spans)for(let i=start;i<end;i++)chars[i]=source[i];
 // Retain single-function C linkage when extracting a desktop translation unit.
 for(const [start] of spans){const linkage=/\bextern\s+"[^"]*"\s*$/.exec(source.slice(0,start));if(linkage)for(let i=linkage.index;i<start;i++)chars[i]=source[i];}
 return {source:chars.join(''),extracted:true,functions:spans.filter(([start,end])=>!wrappers.some(([a,b])=>start>=a&&end<=b)&&/\b__(?:global|device)__\b/.test(masked.slice(start,end))).length};
}
