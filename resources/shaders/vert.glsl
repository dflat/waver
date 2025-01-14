#version 330

in vec3 in_position;
in vec3 in_color;
in vec3 in_normal;

out vec3 frag_normal;
out vec3 frag_world_pos;
out vec3 frag_eye_pos;
out vec3 frag_color;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

uniform vec3 eye_position; // Pass precomputed camera position as a uniform

void main() {
    // Transform position and normal
    vec4 world_pos = model * vec4(in_position, 1.0);
    //vec3 normal = mat3(transpose(inverse(model))) * in_normal; // Normal matrix transformation
    vec3 normal = mat3(model) * in_normal; // Assume model matrix has isometric basis

    // Pass data to fragment shader
    frag_normal = normal;
    frag_world_pos = vec3(world_pos);
    frag_eye_pos = eye_position;
    frag_color = in_color;

    // Compute final vertex position
    gl_Position = projection * view * world_pos;
}

