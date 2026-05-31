struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  var positions = array<vec2f, 6>(
    vec2f(-1.0, -1.0),
    vec2f(1.0, -1.0),
    vec2f(-1.0, 1.0),
    vec2f(-1.0, 1.0),
    vec2f(1.0, -1.0),
    vec2f(1.0, 1.0),
  );
  var uvs = array<vec2f, 6>(
    vec2f(0.0, 1.0),
    vec2f(1.0, 1.0),
    vec2f(0.0, 0.0),
    vec2f(0.0, 0.0),
    vec2f(1.0, 1.0),
    vec2f(1.0, 0.0),
  );
  var output: VertexOutput;
  output.position = vec4f(positions[vertexIndex], 0.0, 1.0);
  output.uv = uvs[vertexIndex];
  return output;
}

struct VideoUniforms {
  opacity: f32,
  rectMinX: f32,
  rectMinY: f32,
  rectMaxX: f32,
  rectMaxY: f32,
};

@group(0) @binding(0) var texSampler: sampler;
@group(0) @binding(1) var videoTexture: texture_external;
@group(0) @binding(2) var<uniform> layer: VideoUniforms;

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  let rectSize = vec2f(layer.rectMaxX - layer.rectMinX, layer.rectMaxY - layer.rectMinY);
  let videoUv = select(
    vec2f(0.5),
    (input.uv - vec2f(layer.rectMinX, layer.rectMinY)) / rectSize,
    rectSize.x > 0.0001 && rectSize.y > 0.0001,
  );
  let tSample = textureSampleBaseClampToEdge(videoTexture, texSampler, clamp(videoUv, vec2f(0.0), vec2f(1.0)));

  let inRectX = step(layer.rectMinX, input.uv.x) * step(input.uv.x, layer.rectMaxX);
  let inRectY = step(layer.rectMinY, input.uv.y) * step(input.uv.y, layer.rectMaxY);
  let inRect = inRectX * inRectY;
  let weight = inRect * layer.opacity;

  return vec4f(tSample.rgb, weight);
}
