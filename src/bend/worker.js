import {compileBend} from './compiler.js';
const runtime=fetch(new URL('./runtime.cu',import.meta.url)).then(r=>{if(!r.ok)throw Error('Cannot load Bend CUDA runtime');return r.text();});
self.onmessage=async({data})=>{
  try {postMessage({id:data.id,result:compileBend(data.source,await runtime,{entry:data.entry})});}
  catch(error){postMessage({id:data.id,error:error.message});}
};
