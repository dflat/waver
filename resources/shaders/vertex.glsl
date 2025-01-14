#version 330

in vec3 in_position;
in vec3 in_color;
in vec3 in_normal;

out vec3 color;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

const int MAX_LIGHTS = 10;

struct Light {
    vec3 position;
    vec3 color;
    float intensity;
};

Light lights[MAX_LIGHTS];

float reflectivity = 1;//0.5;

int lightCount;

vec3 zero_v3 = vec3(0,0,0);

vec3 white = vec3(1.0, 1.0, 1.0);
vec3 blue = vec3(0.4, 0.1, 0.9);

void initializeLights() {
    float default_intensity = 1.0; //0.5;

    // Initialize light_a
    Light light_a = Light(vec3(3.0, 3.0, 3.0) / sqrt(3), white, default_intensity);
    Light light_b = Light(vec3(-1.0, 1, -1.0), blue, default_intensity);

    // Assign light_a to the first element of the array
    lights[0] = light_a;
    lights[1] = light_b;
    lightCount = 2;
}

vec3 calculateColor(vec3 normal, vec3 worldPos, vec3 eyePos, float reflectivity, vec3 baseColor, Light lights[MAX_LIGHTS], int lightCount) {
    // Ensure the normal is normalized
    normal = normalize(normal);

    // Compute the view direction
    vec3 viewDir = normalize(eyePos - worldPos);

    // Initialize the color components
    vec3 ambient = vec3(0.1) * baseColor; // Ambient light as a fraction of the base color
    vec3 diffuse = vec3(0.0);
    vec3 specular = vec3(0.0);

    // Loop through all lights
    for (int i = 0; i < lightCount; i++) {
        Light light = lights[i];

        // Calculate the light direction and normalize
        vec3 lightDir = normalize(light.position - worldPos);

        // Diffuse lighting
        float diff = max(dot(normal, lightDir), 0.0);
        diffuse += diff * light.color * light.intensity;

        // Specular lighting (Phong reflection model)
        vec3 reflectDir = reflect(-lightDir, normal);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), 16.0); // Shininess factor
        specular += spec * light.color * light.intensity * reflectivity;
    }

    // Combine ambient, diffuse, and specular components
    vec3 result = ambient + (diffuse * baseColor) + specular;

    // Clamp the result to ensure valid RGB output
    return clamp(result, 0.0, 1.0);
}

void main() {
    vec3 eyePos = vec3(inverse(view)[3]);
    vec3 normal = vec3(model * vec4(in_normal, 0.0));
    vec4 world_pos = model * vec4(in_position, 1.0);
    vec3 baseColor = in_color;

    initializeLights();

    color = baseColor;
    if (in_normal != zero_v3) {
        color = calculateColor(normal, vec3(world_pos), eyePos, reflectivity, baseColor, lights, lightCount);
    }

    gl_Position = projection * view * world_pos; 
}
