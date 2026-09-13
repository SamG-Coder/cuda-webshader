import {readFileSync,writeFileSync} from 'node:fs';
const read=path=>readFileSync(path),json=path=>JSON.parse(read(path));
const info=json('.local/chrono-marker-info.json'),params=json('reports/chrono-params.json'),captured=json('.local/chrono-params.json');
if(info.markers!==30327||info.fluid!==16731||JSON.stringify(params)!==JSON.stringify(captured))throw Error('Unexpected native dam-break capture');
if(!read('.local/chrono-marker-posrad.bin').equals(read('reports/chrono-search-input.bin')))throw Error('Property and position captures differ');
const properties=read('.local/chrono-marker-properties.bin'),velocities=read('.local/chrono-marker-velocities.bin');
if(properties.length!==info.markers*12||velocities.length!==info.markers*12)throw Error('Incomplete property capture');
for(const b of [properties,velocities])for(let i=0;i<b.length;i+=4)if(!Number.isFinite(b.readFloatLE(i)))throw Error('Nonfinite initial property');
const rho=Buffer.alloc(info.markers*16);
// GetProperties exposes density, pressure and viscosity. This initialized demo
// contains fluid markers followed by fixed boundary markers, with no rigid,
// flexible, ghost or helper particles. Restore only that marker-type component.
for(let i=0;i<info.markers;i++){properties.copy(rho,i*16,i*12,i*12+12);rho.writeFloatLE(i<info.fluid?-1:0,i*16+12);}
writeFileSync('reports/chrono-marker-rhopremu.bin',rho);
writeFileSync('reports/chrono-marker-velocities.bin',velocities);
console.log(`Prepared native density, pressure, viscosity and velocity for ${info.markers} markers`);
