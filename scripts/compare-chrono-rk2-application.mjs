import {readFileSync,writeFileSync} from 'node:fs';import {createHash} from 'node:crypto';
const [c]=JSON.parse(readFileSync('reports/chrono-selected-native.json')),n=c.n,fluid=16731;
const application=readFileSync('reports/chrono-rk2-application.bin'),adapter=readFileSync('reports/chrono-rk2-native.bin'),selected=readFileSync('reports/chrono-selected-native.bin');
if(n!==30327||application.length!==n*40||adapter.length!==n*44+4||adapter.readUInt32LE(n*44)!==0)throw Error('Incomplete application/adapter reference');
const sections=[];
for(const [name,ao,bo,aw,bw]of [['position',0,0,4,4],['velocity',n*4,n*4,3,3],['properties',n*7,n*7,3,4]]){
 let compared=0,unequal=0,maximumError=0;
 for(let i=0;i<n;i++){const original=selected.readUInt32LE((n+i)*4);if(original>=fluid)continue;
  for(let j=0;j<aw;j++){const a=application.readFloatLE((ao+original*aw+j)*4),b=adapter.readFloatLE((bo+i*bw+j)*4);if(!Number.isFinite(a)||!Number.isFinite(b))throw Error('Nonfinite application/adapter state');compared++;if(a!==b)unequal++;maximumError=Math.max(maximumError,Math.abs(a-b));}
 }
 if(compared!==fluid*aw||unequal)throw Error('Original application differs from kernel adapter: '+name);
 sections.push({name,compared,unequal,maximumError});
}
const hash=data=>createHash('sha256').update(data).digest('hex');
const report={markers:n,fluid,steps:1,nativeBitExact:true,sections,sha256:{application:hash(application),adapter:hash(adapter)}};
writeFileSync('reports/chrono-rk2-application-comparison.json',JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));
