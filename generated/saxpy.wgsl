// CUDA WebShader 0.1.0. Generated from kernel saxpy.
@group(0) @binding(0) var<storage, read> b_x: array<f32>;
@group(0) @binding(1) var<storage, read_write> b_y: array<f32>;
struct CWParams {
  p_a: f32,
  p_n: u32,
  cw_pad_8: u32,
  cw_pad_12: u32,
}
@group(0) @binding(2) var<uniform> cw_params: CWParams;
const cw_block_size: vec3<u32> = vec3<u32>(128u, 1u, 1u);


@compute @workgroup_size(128, 1, 1)
fn main(
  @builtin(local_invocation_id) cw_thread: vec3<u32>,
  @builtin(workgroup_id) cw_block: vec3<u32>,
  @builtin(num_workgroups) cw_grid: vec3<u32>
) {
  var v_i: u32 = ((cw_block.x * cw_block_size.x) + cw_thread.x);
  if ((v_i < cw_params.p_n)) {
    b_y[v_i] = fma(cw_params.p_a, b_x[v_i], b_y[v_i]);
  }
}
