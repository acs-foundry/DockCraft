struct CameraUniform {
    view_proj: mat4x4<f32>,
    eye_pos: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> cam: CameraUniform;

struct SphereVertexIn {
    @location(0) local_pos: vec3<f32>,
    @location(1) inst_pos_rad: vec4<f32>,
    @location(2) inst_color: vec4<f32>,
};

struct SphereVertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) world_pos: vec3<f32>,
    @location(2) color: vec4<f32>,
};

@vertex
fn vs_sphere(v: SphereVertexIn) -> SphereVertexOut {
    let world_pos = v.inst_pos_rad.xyz + v.local_pos * v.inst_pos_rad.w;
    var out: SphereVertexOut;
    out.clip_pos = cam.view_proj * vec4<f32>(world_pos, 1.0);
    out.world_normal = normalize(v.local_pos);
    out.world_pos = world_pos;
    out.color = v.inst_color;
    return out;
}

@fragment
fn fs_solid(in: SphereVertexOut) -> @location(0) vec4<f32> {
    let n = normalize(in.world_normal);
    let light_dir = normalize(vec3<f32>(0.45, 0.75, 0.65));
    let view_dir = normalize(cam.eye_pos.xyz - in.world_pos);
    let diff = max(dot(n, light_dir), 0.0);
    let half_vec = normalize(light_dir + view_dir);
    let spec = pow(max(dot(n, half_vec), 0.0), 36.0) * 0.32;
    let amb = 0.24;
    let rgb = in.color.rgb * (amb + 0.78 * diff) + vec3<f32>(spec);
    return vec4<f32>(rgb, in.color.a);
}

@fragment
fn fs_corona(in: SphereVertexOut) -> @location(0) vec4<f32> {
    let n = normalize(in.world_normal);
    let view_dir = normalize(cam.eye_pos.xyz - in.world_pos);
    let fresnel = pow(1.0 - max(dot(n, view_dir), 0.0), 1.65);
    let alpha = clamp(in.color.a * (0.28 + 0.85 * fresnel), 0.08, 0.78);
    let glow = in.color.rgb * (0.85 + 0.35 * fresnel);
    return vec4<f32>(glow, alpha);
}

struct CylinderVertexIn {
    @location(0) unit_pos: vec3<f32>, // x,y on unit circle, z in [0,1]
    @location(1) inst_a_rad: vec4<f32>,
    @location(2) inst_b_pad: vec4<f32>,
    @location(3) inst_color: vec4<f32>,
};

@vertex
fn vs_cylinder(v: CylinderVertexIn) -> SphereVertexOut {
    let a = v.inst_a_rad.xyz;
    let b = v.inst_b_pad.xyz;
    let r = v.inst_a_rad.w;
    let axis = b - a;
    let len = max(length(axis), 1e-4);
    let w = axis / len;
    var up = vec3<f32>(0.0, 1.0, 0.0);
    if abs(w.y) > 0.9 {
        up = vec3<f32>(1.0, 0.0, 0.0);
    }
    let u = normalize(cross(up, w));
    let ortho_v = cross(w, u);
    let radial = (u * v.unit_pos.x + ortho_v * v.unit_pos.y) * r;
    let world_pos = mix(a, b, v.unit_pos.z) + radial;
    let world_normal = normalize(u * v.unit_pos.x + ortho_v * v.unit_pos.y);

    var out: SphereVertexOut;
    out.clip_pos = cam.view_proj * vec4<f32>(world_pos, 1.0);
    out.world_normal = world_normal;
    out.world_pos = world_pos;
    out.color = v.inst_color;
    return out;
}

struct LineIn {
    @location(0) pos: vec3<f32>,
    @location(1) color: vec4<f32>,
};

struct LineOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_line(v: LineIn) -> LineOut {
    var out: LineOut;
    out.clip_pos = cam.view_proj * vec4<f32>(v.pos, 1.0);
    out.color = v.color;
    return out;
}

@fragment
fn fs_line(in: LineOut) -> @location(0) vec4<f32> {
    return in.color;
}

struct QuadOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_quad(@builtin(vertex_index) idx: u32) -> QuadOut {
    var pos = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, 1.0)
    );
    var uvs = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 0.0)
    );
    var out: QuadOut;
    out.clip_pos = vec4<f32>(pos[idx], 0.0, 1.0);
    out.uv = uvs[idx];
    return out;
}

@group(0) @binding(0)
var offscreen_tex: texture_2d<f32>;
@group(0) @binding(1)
var offscreen_samp: sampler;

@fragment
fn fs_quad(in: QuadOut) -> @location(0) vec4<f32> {
    return textureSample(offscreen_tex, offscreen_samp, in.uv);
}
