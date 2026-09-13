// MIT sandbox camera-input test. Original path-tracer code above is unchanged.
// xyz is an orbit-camera position supplied by the sandbox mouse input.
// w enables this override; zero preserves the original create_world camera.
__global__ void sandbox_mouse_camera(camera **d_camera, const float *mouse_camera, int nx, int ny) {
    if (mouse_camera[3] > 0.0f) {
        delete *d_camera;
        *d_camera = new camera(vec3(mouse_camera[0], mouse_camera[1], mouse_camera[2]),
                              vec3(0,0,0), vec3(0,1,0), 20.0f,
                              float(nx)/float(ny), 0.1f, 10.0f);
    }
}
