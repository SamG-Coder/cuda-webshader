// Launch shapes are part of each kernel's contract, not automatic CUDA launch inference.
export const KERNELS = [
  {id:'saxpy', title:'SAXPY · fused scalar', file:'saxpy.cu', workgroupSize:[128,1,1], family:'bandwidth', description:'One coalesced fused multiply-add pass.'},
  {id:'saxpy_vec4', title:'SAXPY · float4', file:'saxpy_vec4.cu', workgroupSize:[128,1,1], family:'bandwidth', description:'Four values per invocation. 16-byte records.'},
  {id:'matmul_naive', title:'Matrix multiply · baseline', file:'matmul_naive.cu', workgroupSize:[8,8,1], family:'compute', description:'One output per lane; baseline global-memory loads.'},
  {id:'matmul_tiled', title:'Matrix multiply · shared tile', file:'matmul_tiled.cu', workgroupSize:[16,16,1], family:'compute', description:'16×16 cooperative workgroup tiles.'},
  {id:'matmul_register', title:'Matrix multiply · register tile', file:'matmul_register.cu', workgroupSize:[8,8,1], family:'compute', description:'2×2 outputs per lane, 16×16 shared tiles.'},
  {id:'reduce_sum', title:'Reduction · hierarchical', file:'reduce_sum.cu', workgroupSize:[128,1,1], family:'cooperative', description:'Two loads per lane; partial sums dispatched recursively.'},
  {id:'convolution', title:'Convolution · shared halo', file:'convolution.cu', workgroupSize:[128,1,1], family:'cooperative', description:'Five taps, zero-padding and cooperative halo loads.'},
  {id:'histogram', title:'Histogram · integer atomics', file:'histogram.cu', workgroupSize:[128,1,1], family:'atomic', description:'256 shared bins, then global atomic merge.'},
  {id:'transpose', title:'Transpose · padded tile', file:'transpose.cu', workgroupSize:[32,8,1], family:'bandwidth', description:'32×33 shared tile; rectangular and tail-safe.'},
  {id:'particles', title:'Particles · render interop', file:'particles.cu', workgroupSize:[128,1,1], family:'interop', description:'GPU-resident float4 position and velocity records.'}
];
export async function loadKernelSources(base = new URL('../kernels/', import.meta.url)) {
  return Object.fromEntries(await Promise.all(KERNELS.map(async k => {
    const response = await fetch(new URL(k.file, base));
    if (!response.ok) throw new Error(`Cannot load ${k.file}: HTTP ${response.status}`);
    return [k.id, await response.text()];
  })));
}
export function kernelOptions(id) { const k = KERNELS.find(k => k.id === id); if (!k) throw new Error(`Unknown kernel ${id}`); return {entry:id, workgroupSize:k.workgroupSize}; }
