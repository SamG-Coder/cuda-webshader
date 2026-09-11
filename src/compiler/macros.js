// Supported function-like macros forward every argument once, in order, to a named call.
// No expression substitution, variadics, token pasting, stringification or directives.
export function forwardingMacro(directive) {
 const m=directive.match(/^#\s*define\s+([A-Za-z_]\w*)\(([^()]*)\)\s+([A-Za-z_]\w*)\(([^()]*)\)\s*(?:\/\/[^\n]*)?$/);
 if(!m)return null;
 const split=s=>s.trim()?s.split(',').map(p=>p.trim()):[],params=split(m[2]),args=split(m[4]);
 if(params.some(p=>!/^[A-Za-z_]\w*$/.test(p))||new Set(params).size!==params.length||params.length!==args.length||params.some((p,i)=>p!==args[i]))return null;
 return {name:m[1],target:m[3],arity:params.length};
}
