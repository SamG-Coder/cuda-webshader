// CUDA WebShader 0.1.0. Generated from kernel convolution.
@group(0) @binding(0) var<storage, read> b_input: array<f32>;
@group(0) @binding(1) var<storage, read_write> b_output: array<f32>;
struct CWParams {
  p_n: u32,
  cw_pad_4: u32,
  cw_pad_8: u32,
  cw_pad_12: u32,
}
@group(0) @binding(2) var<uniform> cw_params: CWParams;
const cw_block_size: vec3<u32> = vec3<u32>(128u, 1u, 1u);
var<workgroup> s_tile: array<f32, 132>;


@compute @workgroup_size(128, 1, 1)
fn main(
  @builtin(local_invocation_id) cw_thread: vec3<u32>,
  @builtin(workgroup_id) cw_block: vec3<u32>,
  @builtin(num_workgroups) cw_grid: vec3<u32>
) {
  var v_lane: u32 = cw_thread.x;
  var v_i: u32 = ((cw_block.x * u32(128i)) + v_lane);
  var cw_tmp_0: f32;
  if ((v_i < cw_params.p_n)) {
    cw_tmp_0 = b_input[v_i];
  } else {
    cw_tmp_0 = 0.0f;
  }
  s_tile[(v_lane + u32(2i))] = cw_tmp_0;
  if ((v_lane < u32(2i))) {
    var v_left: i32 = (i32(((cw_block.x * u32(128i)) + v_lane)) - 2i);
    var v_right: u32 = (((cw_block.x * u32(128i)) + u32(128i)) + v_lane);
    var cw_tmp_1: f32;
    if (((v_left >= 0i) && (v_left < i32(cw_params.p_n)))) {
      cw_tmp_1 = b_input[v_left];
    } else {
      cw_tmp_1 = 0.0f;
    }
    s_tile[v_lane] = cw_tmp_1;
    var cw_tmp_2: f32;
    if ((v_right < cw_params.p_n)) {
      cw_tmp_2 = b_input[v_right];
    } else {
      cw_tmp_2 = 0.0f;
    }
    s_tile[(u32((128i + 2i)) + v_lane)] = cw_tmp_2;
  }
  workgroupBarrier();
  if ((v_i < cw_params.p_n)) {
    var v_sum: f32 = (s_tile[v_lane] * 0.0625f);
    v_sum = fma(s_tile[(v_lane + u32(1i))], 0.25f, v_sum);
    v_sum = fma(s_tile[(v_lane + u32(2i))], 0.375f, v_sum);
    v_sum = fma(s_tile[(v_lane + u32(3i))], 0.25f, v_sum);
    b_output[v_i] = fma(s_tile[(v_lane + u32(4i))], 0.0625f, v_sum);
  }
}
