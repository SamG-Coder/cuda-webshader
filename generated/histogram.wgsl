// CUDA WebShader 0.1.0. Generated from kernel histogram.
@group(0) @binding(0) var<storage, read> b_input: array<u32>;
@group(0) @binding(1) var<storage, read_write> b_bins: array<atomic<u32>>;
struct CWParams {
  p_n: u32,
  cw_pad_4: u32,
  cw_pad_8: u32,
  cw_pad_12: u32,
}
@group(0) @binding(2) var<uniform> cw_params: CWParams;
const cw_block_size: vec3<u32> = vec3<u32>(128u, 1u, 1u);
var<workgroup> s_localBins: array<atomic<u32>, 256>;


@compute @workgroup_size(128, 1, 1)
fn main(
  @builtin(local_invocation_id) cw_thread: vec3<u32>,
  @builtin(workgroup_id) cw_block: vec3<u32>,
  @builtin(num_workgroups) cw_grid: vec3<u32>
) {
  {
    var v_b: u32 = cw_thread.x;
    loop {
      if (!(v_b < u32(256i))) { break; }
      atomicStore(&s_localBins[v_b], 0u);
      continuing {
        v_b = (v_b + cw_block_size.x);
      }
    }
  }
  workgroupBarrier();
  storageBarrier();
  {
    var v_i: u32 = ((cw_block.x * cw_block_size.x) + cw_thread.x);
    loop {
      if (!(v_i < cw_params.p_n)) { break; }
      _ = atomicAdd(&s_localBins[(b_input[v_i] & 255u)], 1u);
      continuing {
        v_i = (v_i + (cw_block_size.x * cw_grid.x));
      }
    }
  }
  workgroupBarrier();
  storageBarrier();
  {
    var v_b: u32 = cw_thread.x;
    loop {
      if (!(v_b < u32(256i))) { break; }
      var v_value: u32 = atomicLoad(&s_localBins[v_b]);
      if ((v_value > 0u)) {
        _ = atomicAdd(&b_bins[v_b], v_value);
      }
      continuing {
        v_b = (v_b + cw_block_size.x);
      }
    }
  }
}
