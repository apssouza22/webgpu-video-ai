struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

struct DetectionBoxGpu {
  bounds: vec4f,
  color: vec4f,
};

struct DetectionUniforms {
  count: u32,
  lineWidth: f32,
  _pad: vec2f,
  boxes: array<DetectionBoxGpu, 32>,
};

@group(0) @binding(0) var<uniform> detection: DetectionUniforms;

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

fn onBorder(uv: vec2f, bounds: vec4f, lineWidth: f32) -> bool {
  let xmin = bounds.x;
  let ymin = bounds.y;
  let xmax = bounds.z;
  let ymax = bounds.w;

  let insideX = step(xmin, uv.x) * step(uv.x, xmax);
  let insideY = step(ymin, uv.y) * step(uv.y, ymax);
  let inside = insideX * insideY > 0.5;

  let nearLeft = abs(uv.x - xmin) < lineWidth;
  let nearRight = abs(uv.x - xmax) < lineWidth;
  let nearTop = abs(uv.y - ymin) < lineWidth;
  let nearBottom = abs(uv.y - ymax) < lineWidth;

  let verticalSpan = insideY > 0.5;
  let horizontalSpan = insideX > 0.5;

  return (nearLeft && verticalSpan) || (nearRight && verticalSpan) ||
    (nearTop && horizontalSpan) || (nearBottom && horizontalSpan) ||
    (inside && (
      abs(uv.x - xmin) < lineWidth ||
      abs(uv.x - xmax) < lineWidth ||
      abs(uv.y - ymin) < lineWidth ||
      abs(uv.y - ymax) < lineWidth
    ));
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  let uv = input.uv;
  let lineWidth = detection.lineWidth;
  var color = vec4f(0.0);

  for (var i = 0u; i < detection.count; i++) {
    let box = detection.boxes[i];
    if (onBorder(uv, box.bounds, lineWidth)) {
      color = vec4f(box.color.rgb, 1.0);
    }
  }

  return color;
}
