import {readdir,readFile,writeFile} from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';

// Keep every project module in the same release, including transitive imports.
// GitHub Pages can otherwise serve a newly published page with cached old JS.
export async function versionBuildAssets(root) {
  async function walk(directory) {
    const files=[];
    for(const entry of await readdir(directory,{withFileTypes:true})) {
      if(entry.name==='node_modules') continue;
      const file=path.join(directory,entry.name);
      if(entry.isDirectory()) files.push(...await walk(file));
      else files.push(file);
    }
    return files.sort();
  }
  const files=await walk(root),assets=files.filter(f=>/\.(js|css)$/.test(f));
  const hash=createHash('sha256');
  for(const file of assets) hash.update(path.relative(root,file).replaceAll('\\','/')).update(await readFile(file));
  const version=hash.digest('hex').slice(0,16);
  // Workers have their own module graph and do not inherit page import maps.
  for(const file of assets.filter(f=>f.endsWith('.js'))) {
    const source=await readFile(file,'utf8');
    await writeFile(file,source.replace(/(['"])(\.{1,2}\/[^'"\s?#]+\.js)\1/g,(_,quote,url)=>quote+url+'?v='+version+quote));
  }
  for(const file of files.filter(f=>f.endsWith('.html'))) {
    let html=await readFile(file,'utf8');
    html=html.replace(/\b(src|href)=(['"])([^'"?#]+\.(?:js|css))\2/g,(match,attribute,quote,url)=>{
      if(/^(?:[a-z]+:|\/\/)/i.test(url)||url.includes('node_modules/')) return match;
      return attribute+'='+quote+url+'?v='+version+quote;
    });
    await writeFile(file,html);
  }
  return version;
}
