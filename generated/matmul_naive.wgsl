// CUDA WebShader 0.1.0. Generated from kernel matmul_naive.
@group(0) @binding(0) var<storage, read> b_A: array<f32>;
@group(0) @binding(1) var<storage, read> b_B: array<f32>;
@group(0) @binding(2) var<storage, read_write> b_C: array<f32>;
struct CWParams {
  p_M: u32,
  p_N: u32,
  p_K: u32,
  cw_pad_12: u32,
}
@group(0) @binding(3) var<uniform> cw_params: CWParams;
const cw_block_size: vec3<u32> = vec3<u32>(8u, 8u, 1u);


@compute @workgroup_size(8, 8, 1)
fn main(
  @builtin(local_invocation_id) cw_thread: vec3<u32>,
  @builtin(workgroup_id) cw_block: vec3<u32>,
  @builtin(num_workgroups) cw_grid: vec3<u32>
) {
  var v_row: u32 = ((cw_block.y * cw_block_size.y) + cw_thread.y);
  var v_col: u32 = ((cw_block.x * cw_block_size.x) + cw_thread.x);
  if (((v_row < cw_params.p_M) && (v_col < cw_params.p_N))) {
    var v_sum: f32 = 0.0f;
    {
      var v_k: u32 = u32(0i);
      loop {
        if (!(v_k < cw_params.p_K)) { break; }
        v_sum = fma(b_A[((v_row * cw_params.p_K) + v_k)], b_B[((v_k * cw_params.p_N) + v_col)], v_sum);
        continuing {
          v_k += u32(1);
        }
      }
    }
    b_C[((v_row * cw_params.p_N) + v_col)] = v_sum;
  }
}
