// CUDA WebShader 0.1.0. Generated from kernel matmul_tiled.
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
const cw_block_size: vec3<u32> = vec3<u32>(16u, 16u, 1u);
var<workgroup> s_tileA: array<array<f32, 16>, 16>;
var<workgroup> s_tileB: array<array<f32, 16>, 16>;


@compute @workgroup_size(16, 16, 1)
fn main(
  @builtin(local_invocation_id) cw_thread: vec3<u32>,
  @builtin(workgroup_id) cw_block: vec3<u32>,
  @builtin(num_workgroups) cw_grid: vec3<u32>
) {
  var v_tx: u32 = cw_thread.x;
  var v_ty: u32 = cw_thread.y;
  var v_row: u32 = ((cw_block.y * u32(16i)) + v_ty);
  var v_col: u32 = ((cw_block.x * u32(16i)) + v_tx);
  var v_sum: f32 = 0.0f;
  {
    var v_base: u32 = u32(0i);
    loop {
      if (!(v_base < cw_params.p_K)) { break; }
      var cw_tmp_0: f32;
      if (((v_row < cw_params.p_M) && ((v_base + v_tx) < cw_params.p_K))) {
        cw_tmp_0 = b_A[(((v_row * cw_params.p_K) + v_base) + v_tx)];
      } else {
        cw_tmp_0 = 0.0f;
      }
      s_tileA[v_ty][v_tx] = cw_tmp_0;
      var cw_tmp_1: f32;
      if ((((v_base + v_ty) < cw_params.p_K) && (v_col < cw_params.p_N))) {
        cw_tmp_1 = b_B[(((v_base + v_ty) * cw_params.p_N) + v_col)];
      } else {
        cw_tmp_1 = 0.0f;
      }
      s_tileB[v_ty][v_tx] = cw_tmp_1;
      workgroupBarrier();
      {
        var v_k: u32 = u32(0i);
        loop {
          if (!(v_k < u32(16i))) { break; }
          v_sum = fma(s_tileA[v_ty][v_k], s_tileB[v_k][v_tx], v_sum);
          continuing {
            v_k += u32(1);
          }
        }
      }
      workgroupBarrier();
      continuing {
        v_base = (v_base + u32(16i));
      }
    }
  }
  if (((v_row < cw_params.p_M) && (v_col < cw_params.p_N))) {
    b_C[((v_row * cw_params.p_N) + v_col)] = v_sum;
  }
}
