#version 330 core

in vec3 vsPos3;
in float radius;

out vec4 FragColor;
const float NEAR = 0.1f;
const float FAR = 100.0f;


void main()
{
	vec3 normal;
	normal.xy = gl_PointCoord * 2 - 1.0;
	float r_sqr = dot(normal.xy, normal.xy);
	if (r_sqr > 1.0 - 1e-6)
		discard;
	normal.z = -sqrt(1.0 - r_sqr);
	normal = normalize(normal);

	vec4 lightPos = vec4(100.0, 100.0, -100.0, 1.0);
	vec3 lightDir = normalize(lightPos.xyz - vsPos3);
	float diffuse = max(0.2f, dot(normal, lightDir));

	FragColor = diffuse * vec4(0.30f, 0.30f, 0.85f, 0.5);
}