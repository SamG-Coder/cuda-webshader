// CUDA WebShader 0.1.0. Generated from kernel simple_vbo_kernel.
@group(0) @binding(0) var<storage, read_write> b_pos: array<vec4<f32>>;
struct CWParams {
  p_width: u32,
  p_height: u32,
  p_time: f32,
  cw_pad_12: u32,
}
@group(0) @binding(1) var<uniform> cw_params: CWParams;
const cw_block_size: vec3<u32> = vec3<u32>(8u, 8u, 1u);


@compute @workgroup_size(8, 8, 1)
fn main(
  @builtin(local_invocation_id) cw_thread: vec3<u32>,
  @builtin(workgroup_id) cw_block: vec3<u32>,
  @builtin(num_workgroups) cw_grid: vec3<u32>
) {
  var v_x: u32 = ((cw_block.x * cw_block_size.x) + cw_thread.x);
  var v_y: u32 = ((cw_block.y * cw_block_size.y) + cw_thread.y);
  var v_u: f32 = (f32(v_x) / f32(cw_params.p_width));
  var v_v: f32 = (f32(v_y) / f32(cw_params.p_height));
  v_u = ((v_u * 2.0f) - 1.0f);
  v_v = ((v_v * 2.0f) - 1.0f);
  var v_freq: f32 = 4.0f;
  var v_w: f32 = ((sin(((v_u * v_freq) + cw_params.p_time)) * cos(((v_v * v_freq) + cw_params.p_time))) * 0.5f);
  b_pos[((v_y * cw_params.p_width) + v_x)] = vec4<f32>(v_u, v_w, v_v, 1.0f);
}
