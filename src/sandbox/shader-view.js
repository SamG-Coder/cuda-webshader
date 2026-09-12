export function createShaderView(sourceEditor,{download,onView=()=>{}}){
 const $=id=>document.getElementById(id);
 monaco.languages.register({id:'wgsl'});
 monaco.languages.setMonarchTokensProvider('wgsl',{
  keywords:['fn','var','let','const','override','struct','return','if','else','for','while','loop','continuing','break','continue','switch','case','default','discard','true','false','enable','requires','alias'],
  typeKeywords:['bool','f32','f16','i32','u32','vec2','vec3','vec4','mat2x2','mat3x3','mat4x4','array','atomic','ptr','sampler','texture_2d'],
  tokenizer:{root:[[/\/\/.*$/,'comment'],[/\/\*/,'comment','@comment'],[/@[a-zA-Z_]\w*/,'annotation'],[/[a-zA-Z_]\w*/,{cases:{'@keywords':'keyword','@typeKeywords':'type','@default':'identifier'}}],[/\b(?:0x[\da-fA-F]+|\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)[fhiu]?/,'number'],[/[{}()[\]]/,'@brackets'],[/[+\-*\/%=<>!&|^~]+/,'operator']],comment:[[/[^/*]+/,'comment'],[/\*\//,'comment','@pop'],[/[/*]/,'comment']]}
 });
 const generatedEditor=monaco.editor.create($('wgsl-editor'),{value:'// Compile CUDA to inspect the generated WebGPU shader.',language:'wgsl',theme:'cuda-dark',readOnly:true,domReadOnly:true,automaticLayout:true,fontSize:13,lineHeight:21,minimap:{enabled:false},scrollBeyondLastLine:false,padding:{top:16},wordWrap:'off',ariaLabel:'Generated WGSL source, read only',stickyScroll:{enabled:false}});
 let artifact=null,view='cuda',stale=true;
 const status=(text,old=false)=>{$('shader-status').textContent=text;$('shader-status').classList.toggle('stale',old);};
 function select(next){view=next;document.querySelectorAll('[data-code-view]').forEach(b=>{b.setAttribute('aria-selected',String(b.dataset.codeView===next));b.tabIndex=b.dataset.codeView===next?0:-1;});$('code-editors').dataset.view=next;document.querySelector('main').classList.toggle('comparing',next==='compare');requestAnimationFrame(()=>{sourceEditor.layout();generatedEditor.layout();});onView(next);}
 const tabs=[...document.querySelectorAll('[data-code-view]')];
 tabs.forEach((b,i)=>{b.onclick=()=>select(b.dataset.codeView);b.onkeydown=e=>{let index;if(e.key==='ArrowRight')index=(i+1)%tabs.length;else if(e.key==='ArrowLeft')index=(i+tabs.length-1)%tabs.length;else if(e.key==='Home')index=0;else if(e.key==='End')index=tabs.length-1;else return;e.preventDefault();tabs[index].focus();select(tabs[index].dataset.codeView);};});
 $('save-wgsl').onclick=()=>artifact&&download(`${artifact.name}${stale?'-last-compiled':''}.wgsl`,artifact.wgsl);
 select('cuda');
 return {generatedEditor,select,get view(){return view;},
  generated(next){artifact=next;stale=false;generatedEditor.setValue(next.wgsl);$('save-wgsl').disabled=false;status(`${next.name}.wgsl · generated from ${next.sourceLabel||'current CUDA'} · GPU validation pending`);},
  validated(){if(artifact&&!stale)status(`${artifact.name}.wgsl · generated from ${artifact.sourceLabel||'current CUDA'} · accepted by GPU`);},
  markStale(){stale=true;if(artifact)status('OUTDATED — CUDA or block settings changed. Showing the last generated shader; compile again to update.',true);},
  failed(){if(artifact)status(stale?'COMPILE FAILED — showing the previous generated shader.':'Run failed — generated WGSL is available; see the execution log.',true);else status('No generated shader — fix the CUDA error and compile again.',true);},
  dispose(){generatedEditor.dispose();}
 };
}
