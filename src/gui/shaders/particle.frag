#version 330 core
in vec4 vColor;
out vec4 FragColor;

void main() {
    // 圆形点
    vec2 coord = gl_PointCoord - vec2(0.5);
    float dist = length(coord);
    if (dist > 0.5) discard;
    float alpha = smoothstep(0.5, 0.3, dist);
    FragColor = vec4(vColor.rgb, alpha);
}