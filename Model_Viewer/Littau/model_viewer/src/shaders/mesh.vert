// Vertex shader
#version 330 core
#extension GL_ARB_explicit_attrib_location : require

layout(location = 0) in vec4 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec2 texcoord;
layout(location = 3) in vec3 a_tangent;
layout(location = 4) in vec3 a_bitangent;

out vec3 Normal;
out vec3 WorldPos;
out vec2 UV;

uniform mat4 u_mvp;
uniform mat4 u_mv; // ModelView matrix
uniform vec3 u_light_position; // The position of your light source
uniform mat3 mv3x3;

void main()
{
    gl_Position = u_mvp * a_position;

    vec3 position_eye = vec3(u_mv * a_position);

    UV = texcoord;
    WorldPos = -position_eye;
    Normal = normalize(mat3(u_mv) * a_normal);




}
