// CUDA WebShader 0.1.0. Generated from kernel matmul_register.
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
var<workgroup> s_tileA: array<array<f32, 16>, 16>;
var<workgroup> s_tileB: array<array<f32, 16>, 16>;


@compute @workgroup_size(8, 8, 1)
fn main(
  @builtin(local_invocation_id) cw_thread: vec3<u32>,
  @builtin(workgroup_id) cw_block: vec3<u32>,
  @builtin(num_workgroups) cw_grid: vec3<u32>
) {
  var v_tx: u32 = cw_thread.x;
  var v_ty: u32 = cw_thread.y;
  var v_row0: u32 = ((cw_block.y * u32(16i)) + v_ty);
  var v_row1: u32 = (v_row0 + u32(8i));
  var v_col0: u32 = ((cw_block.x * u32(16i)) + v_tx);
  var v_col1: u32 = (v_col0 + u32(8i));
  var v_c00: f32 = 0.0f;
  var v_c01: f32 = 0.0f;
  var v_c10: f32 = 0.0f;
  var v_c11: f32 = 0.0f;
  {
    var v_base: u32 = u32(0i);
    loop {
      if (!(v_base < cw_params.p_K)) { break; }
      var cw_tmp_0: f32;
      if (((v_row0 < cw_params.p_M) && ((v_base + v_tx) < cw_params.p_K))) {
        cw_tmp_0 = b_A[(((v_row0 * cw_params.p_K) + v_base) + v_tx)];
      } else {
        cw_tmp_0 = 0.0f;
      }
      s_tileA[v_ty][v_tx] = cw_tmp_0;
      var cw_tmp_1: f32;
      if (((v_row0 < cw_params.p_M) && (((v_base + v_tx) + u32(8i)) < cw_params.p_K))) {
        cw_tmp_1 = b_A[((((v_row0 * cw_params.p_K) + v_base) + v_tx) + u32(8i))];
      } else {
        cw_tmp_1 = 0.0f;
      }
      s_tileA[v_ty][(v_tx + u32(8i))] = cw_tmp_1;
      var cw_tmp_2: f32;
      if (((v_row1 < cw_params.p_M) && ((v_base + v_tx) < cw_params.p_K))) {
        cw_tmp_2 = b_A[(((v_row1 * cw_params.p_K) + v_base) + v_tx)];
      } else {
        cw_tmp_2 = 0.0f;
      }
      s_tileA[(v_ty + u32(8i))][v_tx] = cw_tmp_2;
      var cw_tmp_3: f32;
      if (((v_row1 < cw_params.p_M) && (((v_base + v_tx) + u32(8i)) < cw_params.p_K))) {
        cw_tmp_3 = b_A[((((v_row1 * cw_params.p_K) + v_base) + v_tx) + u32(8i))];
      } else {
        cw_tmp_3 = 0.0f;
      }
      s_tileA[(v_ty + u32(8i))][(v_tx + u32(8i))] = cw_tmp_3;
      var cw_tmp_4: f32;
      if ((((v_base + v_ty) < cw_params.p_K) && (v_col0 < cw_params.p_N))) {
        cw_tmp_4 = b_B[(((v_base + v_ty) * cw_params.p_N) + v_col0)];
      } else {
        cw_tmp_4 = 0.0f;
      }
      s_tileB[v_ty][v_tx] = cw_tmp_4;
      var cw_tmp_5: f32;
      if ((((v_base + v_ty) < cw_params.p_K) && (v_col1 < cw_params.p_N))) {
        cw_tmp_5 = b_B[(((v_base + v_ty) * cw_params.p_N) + v_col1)];
      } else {
        cw_tmp_5 = 0.0f;
      }
      s_tileB[v_ty][(v_tx + u32(8i))] = cw_tmp_5;
      var cw_tmp_6: f32;
      if (((((v_base + v_ty) + u32(8i)) < cw_params.p_K) && (v_col0 < cw_params.p_N))) {
        cw_tmp_6 = b_B[((((v_base + v_ty) + u32(8i)) * cw_params.p_N) + v_col0)];
      } else {
        cw_tmp_6 = 0.0f;
      }
      s_tileB[(v_ty + u32(8i))][v_tx] = cw_tmp_6;
      var cw_tmp_7: f32;
      if (((((v_base + v_ty) + u32(8i)) < cw_params.p_K) && (v_col1 < cw_params.p_N))) {
        cw_tmp_7 = b_B[((((v_base + v_ty) + u32(8i)) * cw_params.p_N) + v_col1)];
      } else {
        cw_tmp_7 = 0.0f;
      }
      s_tileB[(v_ty + u32(8i))][(v_tx + u32(8i))] = cw_tmp_7;
      workgroupBarrier();
      {
        var v_k: u32 = u32(0i);
        loop {
          if (!(v_k < u32(16i))) { break; }
          var v_a0: f32 = s_tileA[v_ty][v_k];
          var v_a1: f32 = s_tileA[(v_ty + u32(8i))][v_k];
          var v_b0: f32 = s_tileB[v_k][v_tx];
          var v_b1: f32 = s_tileB[v_k][(v_tx + u32(8i))];
          v_c00 = fma(v_a0, v_b0, v_c00);
          v_c01 = fma(v_a0, v_b1, v_c01);
          v_c10 = fma(v_a1, v_b0, v_c10);
          v_c11 = fma(v_a1, v_b1, v_c11);
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
  if (((v_row0 < cw_params.p_M) && (v_col0 < cw_params.p_N))) {
    b_C[((v_row0 * cw_params.p_N) + v_col0)] = v_c00;
  }
  if (((v_row0 < cw_params.p_M) && (v_col1 < cw_params.p_N))) {
    b_C[((v_row0 * cw_params.p_N) + v_col1)] = v_c01;
  }
  if (((v_row1 < cw_params.p_M) && (v_col0 < cw_params.p_N))) {
    b_C[((v_row1 * cw_params.p_N) + v_col0)] = v_c10;
  }
  if (((v_row1 < cw_params.p_M) && (v_col1 < cw_params.p_N))) {
    b_C[((v_row1 * cw_params.p_N) + v_col1)] = v_c11;
  }
}
