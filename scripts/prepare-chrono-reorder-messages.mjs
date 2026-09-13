import {readFileSync,writeFileSync} from 'node:fs';
const source=readFileSync(new URL('../.local/chrono-reorder-native.log',import.meta.url),'utf8'),messages=[];let current;
for(const line of source.split(/\r?\n/)){const m=/^CASE (\d+)$/.exec(line);if(m){if(Number(m[1])!==messages.length)throw Error('Unexpected native case order');current=[];messages.push(current);}else if(line){if(!current||!line.startsWith('Error! reorderDataD_ActiveOnly:'))throw Error('Unexpected native diagnostic');current.push(line+'\n');}}
if(JSON.stringify(messages.map(m=>m.length))!=='[0,0,1,2]')throw Error('Incomplete native diagnostic capture');
writeFileSync(new URL('../reports/chrono-reorder-messages.json',import.meta.url),JSON.stringify(messages,null,2)+'\n');
writeFileSync(new URL('../reports/chrono-reorder-native.txt',import.meta.url),source);
