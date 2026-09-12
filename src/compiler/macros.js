// Supported function-like macros forward every argument once, in order, to a named call.
// No expression substitution, variadics, token pasting, stringification or directives.
export function forwardingMacro(directive) {
 const m=directive.match(/^#\s*define\s+([A-Za-z_]\w*)\(([^()]*)\)\s+([A-Za-z_]\w*)\(([^()]*)\)\s*(?:\/\/[^\n]*)?$/);
 if(!m)return null;
 const split=s=>s.trim()?s.split(',').map(p=>p.trim()):[],params=split(m[2]),args=split(m[4]);
 if(params.some(p=>!/^[A-Za-z_]\w*$/.test(p))||new Set(params).size!==params.length||params.length!==args.length||params.some((p,i)=>p!==args[i]))return null;
 return {name:m[1],target:m[3],arity:params.length};
}

// Parenthesized expression macros preserve argument precedence when expanded
// into the AST. Other textual C preprocessing remains outside this subset.
export function expressionMacro(directive){
 const m=directive.match(/^#\s*define\s+([A-Za-z_]\w*)\(([^()]*)\)\s+(.*?)\s*(?:\/\/[^\n]*)?$/);if(!m)return null;
 const params=m[2].trim()?m[2].split(',').map(p=>p.trim()):[],body=m[3];
 if(params.length>16||params.some(p=>!/^[A-Za-z_]\w*$/.test(p))||new Set(params).size!==params.length||body.length>1024||!body.startsWith('(')||!body.endsWith(')')||/[#;{}"'\\]/.test(body))return null;
 let depth=0;for(let i=0;i<body.length;i++){if(body[i]==='(')depth++;if(body[i]===')')depth--;if(depth<0||(depth===0&&i<body.length-1))return null;}if(depth)return null;
 for(const match of body.matchAll(/[A-Za-z_]\w*/g))if(params.includes(match[0])&&(!body.slice(0,match.index).trimEnd().endsWith('(')||!body.slice(match.index+match[0].length).trimStart().startsWith(')')))return null;
 return {name:m[1],params,body};
}
