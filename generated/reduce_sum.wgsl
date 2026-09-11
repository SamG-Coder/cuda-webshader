// CUDA WebShader 0.1.0. Generated from kernel reduce_sum.
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
var<workgroup> s_partial: array<f32, 128>;


@compute @workgroup_size(128, 1, 1)
fn main(
  @builtin(local_invocation_id) cw_thread: vec3<u32>,
  @builtin(workgroup_id) cw_block: vec3<u32>,
  @builtin(num_workgroups) cw_grid: vec3<u32>
) {
  var v_lane: u32 = cw_thread.x;
  var v_i: u32 = (((cw_block.x * u32(128i)) * u32(2i)) + v_lane);
  var v_sum: f32 = 0.0f;
  if ((v_i < cw_params.p_n)) {
    v_sum = b_input[v_i];
  }
  if (((v_i + u32(128i)) < cw_params.p_n)) {
    v_sum = (v_sum + b_input[(v_i + u32(128i))]);
  }
  s_partial[v_lane] = v_sum;
  workgroupBarrier();
  {
    var v_stride: u32 = u32((128i / 2i));
    loop {
      if (!(v_stride > u32(0i))) { break; }
      if ((v_lane < v_stride)) {
        s_partial[v_lane] = (s_partial[v_lane] + s_partial[(v_lane + v_stride)]);
      }
      workgroupBarrier();
      continuing {
        v_stride = (v_stride >> u32(1i));
      }
    }
  }
  if ((v_lane == u32(0i))) {
    b_output[cw_block.x] = s_partial[0i];
  }
}
