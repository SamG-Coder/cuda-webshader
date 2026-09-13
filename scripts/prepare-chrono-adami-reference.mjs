import {readFileSync,writeFileSync} from 'node:fs';
const c=JSON.parse(readFileSync('reports/chrono-selected-native.json'))[0],data=readFileSync('reports/chrono-selected-native.bin');
if(c.n!==30327||c.neighbors!==794643||c.sections.length!==12)throw Error('Unexpected native selected-marker reference');
let offset=c.offset;const sections=c.sections.map(words=>{const b=data.subarray(offset,offset+words*4);offset+=words*4;if(b.length!==words*4)throw Error('Incomplete selected reference');return b;});
// Neighbour offsets, neighbour IDs, positions, properties and velocities, all
// produced by the original native preparation kernels. Fixed walls have zero acceleration.
writeFileSync('reports/chrono-adami-input.bin',Buffer.concat([sections[5],sections[6],sections[7],sections[10],sections[9]]));
console.log('Prepared initialized dam-break boundary inputs');
