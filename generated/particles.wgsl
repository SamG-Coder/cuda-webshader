// CUDA WebShader 0.1.0. Generated from kernel particles.
@group(0) @binding(0) var<storage, read_write> b_position: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> b_velocity: array<vec4<f32>>;
struct CWParams {
  p_n: u32,
  p_dt: f32,
  p_time: f32,
  p_attraction: f32,
}
@group(0) @binding(2) var<uniform> cw_params: CWParams;
const cw_block_size: vec3<u32> = vec3<u32>(128u, 1u, 1u);

fn f_softened_inverse_radius(cw_arg_radiusSquared: f32) -> f32 {
  var v_radiusSquared: f32 = cw_arg_radiusSquared;
  return inverseSqrt((v_radiusSquared + 0.35f));
}

@compute @workgroup_size(128, 1, 1)
fn main(
  @builtin(local_invocation_id) cw_thread: vec3<u32>,
  @builtin(workgroup_id) cw_block: vec3<u32>,
  @builtin(num_workgroups) cw_grid: vec3<u32>
) {
  var v_i: u32 = ((cw_block.x * cw_block_size.x) + cw_thread.x);
  if ((v_i < cw_params.p_n)) {
    var v_p: vec4<f32> = b_position[v_i];
    var v_v: vec4<f32> = b_velocity[v_i];
    var v_radiusSquared: f32 = ((v_p.x * v_p.x) + (v_p.z * v_p.z));
    var v_inv: f32 = f_softened_inverse_radius(v_radiusSquared);
    var v_radial: f32 = (cw_params.p_attraction * (1.4f - (0.11f * sqrt((v_radiusSquared + 0.001f)))));
    var v_ax: f32 = ((((-v_p.z) * 0.52f) - (v_p.x * v_radial)) * v_inv);
    var v_az: f32 = (((v_p.x * 0.52f) - (v_p.z * v_radial)) * v_inv);
    var v_ay: f32 = ((sin(((cw_params.p_time * 0.37f) + (v_p.w * 6.283185f))) * 0.18f) - (v_p.y * 0.22f));
    var v_drag: f32 = max(0.0f, (1.0f - (cw_params.p_dt * 0.12f)));
    v_v.x = (fma(v_ax, cw_params.p_dt, v_v.x) * v_drag);
    v_v.y = (fma(v_ay, cw_params.p_dt, v_v.y) * v_drag);
    v_v.z = (fma(v_az, cw_params.p_dt, v_v.z) * v_drag);
    v_p.x = fma(v_v.x, cw_params.p_dt, v_p.x);
    v_p.y = fma(v_v.y, cw_params.p_dt, v_p.y);
    v_p.z = fma(v_v.z, cw_params.p_dt, v_p.z);
    b_position[v_i] = v_p;
    b_velocity[v_i] = v_v;
  }
}
