// Supported function-like macros forward every argument once, in order, to a named call.
// Forwarders do not perform general textual substitution.
export function forwardingMacro(directive) {
 const m=directive.match(/^#\s*define\s+([A-Za-z_]\w*)\(([^()]*)\)\s+([A-Za-z_]\w*)\(([^()]*)\)\s*(?:\/\/[^\n]*)?$/);
 if(!m)return null;
 const split=s=>s.trim()?s.split(',').map(p=>p.trim()):[],params=split(m[2]),args=split(m[4]);
 if(params.some(p=>!/^[A-Za-z_]\w*$/.test(p))||new Set(params).size!==params.length||params.length!==args.length||params.some((p,i)=>p!==args[i]))return null;
 return {name:m[1],target:m[3],arity:params.length};
}

// Parenthesized expressions and indexed primary macros preserve precedence when expanded
// into the AST. Other textual C preprocessing remains outside this subset.
export function expressionMacro(directive){
 const m=directive.match(/^#\s*define\s+([A-Za-z_]\w*)\(([^()]*)\)\s+(.*?)\s*(?:\/\/[^\n]*)?$/);if(!m)return null;
 const params=m[2].trim()?m[2].split(',').map(p=>p.trim()):[],body=m[3];
 if(params.length>16||params.some(p=>!/^[A-Za-z_]\w*$/.test(p))||new Set(params).size!==params.length||body.length>1024||/[#;{}"'\\]/.test(body))return null;
 // A complete parenthesized expression or named subscript is a primary
 // expression in CUDA, so substituting its AST preserves surrounding precedence.
 const indexed=body.match(/^[A-Za-z_]\w*\s*\[/),begin=indexed?indexed[0].length-1:0;
 if(!indexed&&(!body.startsWith('(')||!body.endsWith(')')))return null;
 const stack=[];for(let i=begin;i<body.length;i++){const c=body[i];if(c==='('||c==='[')stack.push(c);else if(c===')'||c===']'){if(stack.pop()!==(c===')'?'(':'['))return null;}if(!stack.length&&i<body.length-1)return null;}if(stack.length)return null;
 for(const match of body.matchAll(/[A-Za-z_]\w*/g))if(params.includes(match[0])&&(!body.slice(0,match.index).trimEnd().endsWith('(')||!body.slice(match.index+match[0].length).trimStart().startsWith(')')))return null;
 return {name:m[1],params,body};
}
