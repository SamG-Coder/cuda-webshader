// Prepare the exact original activity flags already validated by the selected-marker test.
import {readFileSync,writeFileSync} from 'node:fs';
const [c]=JSON.parse(readFileSync('reports/chrono-selected-native.json')),data=readFileSync('reports/chrono-selected-native.bin');
if(c.mode!==0||c.n!==30327||c.sections.length!==12||c.sections[11]!==c.n)throw Error('Unexpected initialized native marker selection');
const start=c.offset+c.sections.slice(0,11).reduce((a,b)=>a+b,0)*4,end=start+c.n*4;
if(data.length<end)throw Error('Incomplete selected-marker reference');
writeFileSync('reports/chrono-rk2-activity.bin',data.subarray(start,end));
