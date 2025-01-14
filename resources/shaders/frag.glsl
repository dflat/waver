#version 330

in vec3 frag_normal;
in vec3 frag_world_pos;
in vec3 frag_eye_pos;
in vec3 frag_color;

out vec4 final_color;

const int MAX_LIGHTS = 10;

struct Light {
    vec4 position;
    //float padding;
    vec4 color;
    //float padding2;
    float intensity;
};

//layout(std140) uniform LightBuffer {
uniform LightBuffer {
    Light lights[MAX_LIGHTS];
};

//uniform Light lights[MAX_LIGHTS];
uniform int light_count;
uniform float reflectivity;

void main() {
    vec3 normal = normalize(frag_normal);
    vec3 view_dir = normalize(frag_eye_pos - frag_world_pos);

    vec3 ambient = vec3(0.1) * frag_color;
    vec3 diffuse = vec3(0.0);
    vec3 specular = vec3(0.0);

    for (int i = 0; i < light_count; i++) {
        Light light = lights[i];
        vec3 light_dir = normalize(light.position.xyz - frag_world_pos);

        // Diffuse lighting
        float diff = max(dot(normal, light_dir), 0.0);
        diffuse += diff * light.color.rgb * light.intensity;

        // Specular lighting
        vec3 reflect_dir = reflect(-light_dir, normal);
        float spec = pow(max(dot(view_dir, reflect_dir), 0.0), 16.0);
        specular += spec * light.color.rgb * light.intensity * reflectivity;
    }

    vec3 result = ambient + diffuse * frag_color + specular;
    //final_color = vec4(lights[0].color,1); //vec4(clamp(result, 0.0, 1.0), 1.0);
    final_color = vec4(clamp(result, 0.0, 1.0), 1.0);
}

