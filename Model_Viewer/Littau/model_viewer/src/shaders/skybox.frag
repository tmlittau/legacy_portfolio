// Fragment shader
#version 150
out vec4 FragColor;

in vec3 TexCoords;

uniform samplerCube skybox;
uniform bool cubemap_on;

void main()
{

  FragColor = texture(skybox, TexCoords);

}
