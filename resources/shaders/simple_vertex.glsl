#version 330
in vec3 in_position;
in vec3 in_color;
in vec3 in_normal;
out vec3 color;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
//uniform float time;

//float pi = 3.14159;
//float freq = 1/4.0;
//float A = 0.25;

//vec3 zero_v3 = vec3(0,0,0);

void main() {
    //vec3 N = vec3(model * vec4(in_normal, 0.0));
    gl_Position = projection * view * model * vec4(in_position, 1.0);
    color = in_color;
}
