// CUDA WebShader 0.1.0. Generated from kernel transpose.
@group(0) @binding(0) var<storage, read> b_input: array<f32>;
@group(0) @binding(1) var<storage, read_write> b_output: array<f32>;
struct CWParams {
  p_width: u32,
  p_height: u32,
  cw_pad_8: u32,
  cw_pad_12: u32,
}
@group(0) @binding(2) var<uniform> cw_params: CWParams;
const cw_block_size: vec3<u32> = vec3<u32>(32u, 8u, 1u);
var<workgroup> s_tile: array<array<f32, 33>, 32>;


@compute @workgroup_size(32, 8, 1)
fn main(
  @builtin(local_invocation_id) cw_thread: vec3<u32>,
  @builtin(workgroup_id) cw_block: vec3<u32>,
  @builtin(num_workgroups) cw_grid: vec3<u32>
) {
  var v_x: u32 = ((cw_block.x * u32(32i)) + cw_thread.x);
  var v_y: u32 = ((cw_block.y * u32(32i)) + cw_thread.y);
  {
    var v_j: u32 = u32(0i);
    loop {
      if (!(v_j < u32(32i))) { break; }
      var cw_tmp_0: f32;
      if (((v_x < cw_params.p_width) && ((v_y + v_j) < cw_params.p_height))) {
        cw_tmp_0 = b_input[(((v_y + v_j) * cw_params.p_width) + v_x)];
      } else {
        cw_tmp_0 = 0.0f;
      }
      s_tile[(cw_thread.y + v_j)][cw_thread.x] = cw_tmp_0;
      continuing {
        v_j = (v_j + u32(8i));
      }
    }
  }
  workgroupBarrier();
  v_x = ((cw_block.y * u32(32i)) + cw_thread.x);
  v_y = ((cw_block.x * u32(32i)) + cw_thread.y);
  {
    var v_j: u32 = u32(0i);
    loop {
      if (!(v_j < u32(32i))) { break; }
      if (((v_x < cw_params.p_height) && ((v_y + v_j) < cw_params.p_width))) {
        b_output[(((v_y + v_j) * cw_params.p_height) + v_x)] = s_tile[cw_thread.x][(cw_thread.y + v_j)];
      }
      continuing {
        v_j = (v_j + u32(8i));
      }
    }
  }
}
