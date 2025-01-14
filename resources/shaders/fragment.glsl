#version 330
in vec3 color;
out vec4 fragColor;

float pi = 3.14159;
float freq = 1.0 + 0*1/2.0;
vec2 c = vec2(1280/2, 720/2);
float X = 1280;
float Y = 720;
float n = 10;
uniform float time;

void main() {

    float vx = gl_FragCoord.x/X;
    float vy = gl_FragCoord.y/Y;
    float xn = sin(2*pi*freq*time*vx);
    float yn = sin(2*pi*freq*time*vy);
    float m = xn*yn;

    float r = length(gl_FragCoord.xy - c);
    float y = gl_FragCoord.y;
    float s = sin(2*pi*freq*y);
    s = floor(mod(y/7.2, 2))/2;
    vec3 offset = vec3(s,s,s);
    float L = pow(1 - r/X, 1);
    float mask = floor(n*L);
    fragColor = vec4(mask/n*color, 1.0);
    //fragColor = fragColor;//* m;
    fragColor = vec4(color, 1);

}
