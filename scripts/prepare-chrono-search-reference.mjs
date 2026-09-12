import {readFileSync,writeFileSync} from 'node:fs';
const info=JSON.parse(readFileSync('.local/chrono-marker-info.json')),input=readFileSync('.local/chrono-marker-posrad.bin');
if(info.markers!==30327||info.fluid!==16731||input.length!==info.markers*16)throw Error('Unexpected original dam-break marker capture.');
const captured=JSON.parse(readFileSync('.local/chrono-params.json')),params=JSON.parse(readFileSync('reports/chrono-params.json'));
if(Object.keys(captured).length!==Object.keys(params).length||Object.keys(captured).some(k=>captured[k]!==params[k]))throw Error('Regenerate the hash parameter reference before preparing search inputs.');
writeFileSync('reports/chrono-search-input.bin',input);
console.log('Prepared 30,327 actual marker position/radius records, including 13,596 boundary markers.');
