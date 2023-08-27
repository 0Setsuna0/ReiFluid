#version 330 core
layout(location = 0) in vec3 aPos;

out vec3 vsPos3;
out float raduis;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;


void main()
{
	//view space position
	vec4 vsPosition = view * model * vec4(aPos.xyz, 1.0);
	//clip space，裁剪空间,投影至[-1,1]
	gl_Position = projection * view * model * vec4(aPos.xyz, 1.0);

	vec3 vsPos3 = vsPosition.xyz / vsPosition.w;
	float vsPositionLen = length(vsPos3);
	float radius = max(10.0, 5.0 / vsPositionLen);
	gl_PointSize = radius;
}