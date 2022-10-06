// Assignment 3, Part 1 and 2
//
// Modify this file according to the lab instructions.
//

#include "utils.h"
#include "utils2.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <imgui.h>
#include <imgui_impl_glfw_gl3.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <iostream>
#include <cstdlib>
#include <algorithm>

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/mesh.h>
#include <assimp/types.h>
#include <assimp/scene.h>

// The attribute locations we will use in the vertex shader
enum AttributeLocation {
    POSITION = 0,
    NORMAL = 1,
    TEXCOORD = 2,
    TAN = 3,
    BITAN = 4
};

// Struct for representing an indexed triangle mesh
struct Mesh {
    std::vector<glm::vec3> vertices;
    std::vector<glm::vec3> normals;
    std::vector<uint32_t> indices;
    std::vector<glm::vec2> texcoords;
    std::vector<glm::vec3> tangents;
    std::vector<glm::vec3> bitangents;
};

// Struct for representing a vertex array object (VAO) created from a
// mesh. Used for rendering.
struct MeshVAO {
    GLuint vao;
    GLuint vertexVBO;
    GLuint normalVBO;
    GLuint texVBO;
    GLuint tangentVBO;
    GLuint bitangentVBO;
    GLuint indexVBO;
    int numVertices;
    int numIndices;
};

// Struct for resources and state
struct Context {
    int width;
    int height;
    float aspect;
    GLFWwindow *window;
    GLuint program;
    GLuint sb_program;
    Trackball trackball;
    Mesh mesh;
    MeshVAO meshVAO;
    GLuint skyboxVAO;
    GLuint skyboxVBO;
    GLuint defaultVAO;
    GLuint cubemap;
    float elapsed_time;
    GLuint a_text;
    GLuint m_text;
    GLuint n_text;
    GLuint r_text;
    GLuint ao_text;
};

static float zoomFactor = 0.01f;

// Returns the value of an environment variable
std::string getEnvVar(const std::string &name)
{
    char const* value = std::getenv(name.c_str());
    if (value == nullptr) {
        return std::string();
    }
    else {
        return std::string(value);
    }
}

void createSkybox(Context &ctx)
{
    // MODIFY THIS PART: Define the six faces (front, back, left,
    // right, top, and bottom) of the cube. Each face should be
    // constructed from two triangles, and each triangle should be
    // constructed from three vertices. That is, you should define 36
    // vertices that together make up 12 triangles. One triangle is
    // given; you have to define the rest!
    const GLfloat vertices[] = {
      -1.0f,  1.0f, -1.0f,
      -1.0f, -1.0f, -1.0f,
      1.0f, -1.0f, -1.0f,
      1.0f, -1.0f, -1.0f,
      1.0f,  1.0f, -1.0f,
      -1.0f,  1.0f, -1.0f,

      -1.0f, -1.0f,  1.0f,
      -1.0f, -1.0f, -1.0f,
      -1.0f,  1.0f, -1.0f,
      -1.0f,  1.0f, -1.0f,
      -1.0f,  1.0f,  1.0f,
      -1.0f, -1.0f,  1.0f,

      1.0f, -1.0f, -1.0f,
      1.0f, -1.0f,  1.0f,
      1.0f,  1.0f,  1.0f,
      1.0f,  1.0f,  1.0f,
      1.0f,  1.0f, -1.0f,
      1.0f, -1.0f, -1.0f,

      -1.0f, -1.0f,  1.0f,
      -1.0f,  1.0f,  1.0f,
      1.0f,  1.0f,  1.0f,
      1.0f,  1.0f,  1.0f,
      1.0f, -1.0f,  1.0f,
      -1.0f, -1.0f,  1.0f,

      -1.0f,  1.0f, -1.0f,
      1.0f,  1.0f, -1.0f,
      1.0f,  1.0f,  1.0f,
      1.0f,  1.0f,  1.0f,
      -1.0f,  1.0f,  1.0f,
      -1.0f,  1.0f, -1.0f,

      -1.0f, -1.0f, -1.0f,
      -1.0f, -1.0f,  1.0f,
      1.0f, -1.0f, -1.0f,
      1.0f, -1.0f, -1.0f,
      -1.0f, -1.0f,  1.0f,
      1.0f, -1.0f,  1.0f
    };

    glGenBuffers(2, &ctx.skyboxVBO);
    glBindBuffer(GL_ARRAY_BUFFER, ctx.skyboxVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glGenVertexArrays(2, &ctx.skyboxVAO);
    glBindVertexArray(ctx.skyboxVAO);
    glBindBuffer(GL_ARRAY_BUFFER, ctx.skyboxVBO);
    glEnableVertexAttribArray(POSITION);
    glVertexAttribPointer(POSITION, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    glBindVertexArray(ctx.defaultVAO); // unbinds the VAO
}

// Returns the absolute path to the shader directory
std::string shaderDir(void)
{
    std::string rootDir = getEnvVar("LITTAU_ROOT");
    if (rootDir.empty()) {
        std::cout << "Error: LITTAU_ROOT is not set." << std::endl;
        std::exit(EXIT_FAILURE);
    }
    return rootDir + "/model_viewer/src/shaders/";
}

// Returns the absolute path to the 3D model directory
std::string modelDir(void)
{
    std::string rootDir = getEnvVar("LITTAU_ROOT");
    if (rootDir.empty()) {
        std::cout << "Error: LITTAU_ROOT is not set." << std::endl;
        std::exit(EXIT_FAILURE);
    }
    return rootDir + "/model_viewer/3d_models/";
}

std::string textureDir(void)
{
    std::string rootDir = getEnvVar("LITTAU_ROOT");
    if (rootDir.empty()) {
        std::cout << "Error: LITTAU_ROOT is not set." << std::endl;
        std::exit(EXIT_FAILURE);
    }
    return rootDir + "/model_viewer/Textures/";

}

// Returns the absolute path to the cubemap texture directory
std::string cubemapDir(void)
{
    std::string rootDir = getEnvVar("LITTAU_ROOT");
    if (rootDir.empty()) {
        std::cout << "Error: LITTAU_ROOT is not set." << std::endl;
        std::exit(EXIT_FAILURE);
    }
    return rootDir + "/model_viewer/cubemaps/";
}


void importMesh(const std::string &filename, Mesh &fino)
{

  Assimp::Importer importer;

  unsigned int importOptions = aiProcess_FlipUVs            |
    aiProcess_CalcTangentSpace              |
    aiProcess_GenNormals                    |
    aiProcess_JoinIdenticalVertices         |
    aiProcess_Triangulate                   |
    aiProcess_GenUVCoords                   |
    aiProcess_SortByPType;

  //const aiScene *scene = importer.ReadFile(filename, aiProcessPreset_TargetRealtime_Fast);
  const aiScene *scene = importer.ReadFile(filename, importOptions);

  aiMesh *mesh = scene->mMeshes[0];

  // Walk through each of the mesh's vertices
  for (GLuint i = 0; i < mesh->mNumVertices; i++)
    {
        //Vertex vertex;
        glm::vec3 norm;
        glm::vec3 pos;
        glm::vec2 tex;
        glm::vec3 tang;
        glm::vec3 bitan;
        glm::vec3 vector; // We declare a placeholder vector since assimp uses its own vector class that doesn't directly convert to glm's vec3 class so we transfer the data to this placeholder glm::vec3 first.

        // Positions
        vector.x = mesh->mVertices[i].x;
        vector.y = mesh->mVertices[i].y;
        vector.z = mesh->mVertices[i].z;
        pos = vector;

        // Normals
        vector.x = mesh->mNormals[i].x;
        vector.y = mesh->mNormals[i].y;
        vector.z = mesh->mNormals[i].z;
        norm = vector;

        // Tangents and Bitangents
        vector.x = mesh->mTangents[i].x;
        vector.y = mesh->mTangents[i].y;
        vector.z = mesh->mTangents[i].z;
        tang = vector;

        vector.x = mesh->mBitangents[i].x;
        vector.y = mesh->mBitangents[i].y;
        vector.z = mesh->mBitangents[i].z;
        bitan = vector;

        // Texture Coordinates
        if (mesh->mTextureCoords[0]) // Does the mesh contain texture coordinates?
        {
            glm::vec2 vec;
            // A vertex can contain up to 8 different texture coordinates. We thus make the assumption that we won't
            // use models where a vertex can have multiple texture coordinates so we always take the first set (0).
            vec.x = mesh->mTextureCoords[0][i].x;
            vec.y = mesh->mTextureCoords[0][i].y;
            tex = vec;
        }
        else
        {
            tex = glm::vec2(0.0f, 0.0f);
        }

        fino.vertices.push_back(pos);
        fino.normals.push_back(norm);
        fino.texcoords.push_back(tex);
        fino.tangents.push_back(tang);
        fino.bitangents.push_back(bitan);
      }

    for (GLuint i = 0; i < mesh->mNumFaces; i++)
    {
        aiFace face = mesh->mFaces[i];
        // Retrieve all indices of the face and store them in the indices vector
        for (GLuint j = 0; j < face.mNumIndices; j++)
        {
            fino.indices.push_back(face.mIndices[j]);
        }
    }

}

void loadMesh(const std::string &filename, Mesh *mesh)
{
    OBJMeshUV obj_mesh;
    //objMeshLoad(obj_mesh, filename);
    objMeshUVLoad(obj_mesh, filename);
    mesh->vertices = obj_mesh.vertices;
    mesh->normals = obj_mesh.normals;
    mesh->indices = obj_mesh.indices;
    mesh->texcoords = obj_mesh.texcoords;
}

void createMeshVAO(Context &ctx, const Mesh &mesh, MeshVAO *meshVAO)
{

    // Generates and populates a VBO for the vertices
    glGenBuffers(1, &(meshVAO->vertexVBO));
    glBindBuffer(GL_ARRAY_BUFFER, meshVAO->vertexVBO);
    auto verticesNBytes = mesh.vertices.size() * sizeof(mesh.vertices[0]);
    glBufferData(GL_ARRAY_BUFFER, verticesNBytes, mesh.vertices.data(), GL_STATIC_DRAW);

    // Generates and populates a VBO for the vertex normals
    glGenBuffers(1, &(meshVAO->normalVBO));
    glBindBuffer(GL_ARRAY_BUFFER, meshVAO->normalVBO);
    auto normalsNBytes = mesh.normals.size() * sizeof(mesh.normals[0]);
    glBufferData(GL_ARRAY_BUFFER, normalsNBytes, mesh.normals.data(), GL_STATIC_DRAW);

    // Generates and populates a VBO for the uv coordinates
    glGenBuffers(1, &(meshVAO->texVBO));
    glBindBuffer(GL_ARRAY_BUFFER, meshVAO->texVBO);
    auto texcoordsNBytes = mesh.texcoords.size() * sizeof(mesh.texcoords[0]);
    glBufferData(GL_ARRAY_BUFFER, texcoordsNBytes, mesh.texcoords.data(), GL_STATIC_DRAW);

    glGenBuffers(1, &(meshVAO->tangentVBO));
    glBindBuffer(GL_ARRAY_BUFFER, meshVAO->tangentVBO);
    auto tangentNBytes = mesh.tangents.size() * sizeof(mesh.tangents[0]);
    glBufferData(GL_ARRAY_BUFFER, tangentNBytes, mesh.tangents.data(), GL_STATIC_DRAW);

    glGenBuffers(1, &(meshVAO->bitangentVBO));
    glBindBuffer(GL_ARRAY_BUFFER, meshVAO->bitangentVBO);
    auto bitangentNBytes = mesh.bitangents.size() * sizeof(mesh.bitangents[0]);
    glBufferData(GL_ARRAY_BUFFER, bitangentNBytes, mesh.bitangents.data(), GL_STATIC_DRAW);

    // Generates and populates a VBO for the element indices
    glGenBuffers(1, &(meshVAO->indexVBO));
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, meshVAO->indexVBO);
    auto indicesNBytes = mesh.indices.size() * sizeof(mesh.indices[0]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indicesNBytes, mesh.indices.data(), GL_STATIC_DRAW);

    // Creates a vertex array object (VAO) for drawing the mesh
    glGenVertexArrays(1, &(meshVAO->vao));
    glBindVertexArray(meshVAO->vao);

    glBindBuffer(GL_ARRAY_BUFFER, meshVAO->vertexVBO);
    glEnableVertexAttribArray(POSITION);
    glVertexAttribPointer(POSITION, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    glBindBuffer(GL_ARRAY_BUFFER, meshVAO->normalVBO);
    glEnableVertexAttribArray(NORMAL);
    glVertexAttribPointer(NORMAL, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    glBindBuffer(GL_ARRAY_BUFFER, meshVAO->texVBO);
    glEnableVertexAttribArray(TEXCOORD);
    glVertexAttribPointer(TEXCOORD, 2, GL_FLOAT, GL_FALSE, 0, nullptr);

    glBindBuffer(GL_ARRAY_BUFFER, meshVAO->tangentVBO);
    glEnableVertexAttribArray(TAN);
    glVertexAttribPointer(TAN, 2, GL_FLOAT, GL_FALSE, 0, nullptr);

    glBindBuffer(GL_ARRAY_BUFFER, meshVAO->bitangentVBO);
    glEnableVertexAttribArray(BITAN);
    glVertexAttribPointer(BITAN, 2, GL_FLOAT, GL_FALSE, 0, nullptr);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, meshVAO->indexVBO);
    glBindVertexArray(ctx.defaultVAO); // unbinds the VAO

    // Additional information required by draw calls
    meshVAO->numVertices = mesh.vertices.size();
    meshVAO->numIndices = mesh.indices.size();
}

void initializeTrackball(Context &ctx)
{
    double radius = double(std::min(ctx.width, ctx.height)) / 2.0;
    ctx.trackball.radius = radius;
    glm::vec2 center = glm::vec2(ctx.width, ctx.height) / 2.0f;
    ctx.trackball.center = center;
}

void init(Context &ctx)
{

    GLenum eFormat;
    glDepthMask(GL_FALSE);

    ctx.sb_program = loadShaderProgram(shaderDir() + "skybox.vert",
                                    shaderDir() + "skybox.frag");
    createSkybox(ctx);


    glDepthMask(GL_TRUE);
    ctx.program = loadShaderProgram(shaderDir() + "mesh.vert",
                                    shaderDir() + "mesh.frag");


    importMesh((modelDir() + "/Cerberus_LP.FBX"), ctx.mesh);
    createMeshVAO(ctx, ctx.mesh, &ctx.meshVAO);

    ctx.cubemap = loadCubemap(cubemapDir()+"/RomeChurch/prefiltered/2048");
    ctx.a_text = load2DTexture(textureDir() + "/Cerberus/Cerberus_A.png");
    ctx.m_text = load2DTexture(textureDir() + "/Cerberus/Cerberus_M.png");
    ctx.n_text = load2DTexture(textureDir() + "/Cerberus/Cerberus_N.png");
    ctx.r_text = load2DTexture(textureDir() + "/Cerberus/Cerberus_R.png");
    ctx.ao_text = load2DTexture(textureDir() + "/Cerberus/Cerberus_AO.png");





    initializeTrackball(ctx);
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{

  if(yoffset > 0)
  {
    zoomFactor = zoomFactor + 0.001f;
  }
  if(yoffset < 0)
  {
    zoomFactor = zoomFactor - 0.001f;
  }
// here your response to cursor scrolling ...

// yoffset > 0 means you scrolled up, i think
}

void drawSkybox(Context &ctx, GLuint program)
{
  glUseProgram(program);
  glm::mat4 model = glm::mat4(1.0f);
  glm::mat4 view = glm::mat4(1.0f);
  glm::mat4 projection = glm::mat4(1.0f);


  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_CUBE_MAP, 1);

  glUniformMatrix4fv(glGetUniformLocation(program, "projection"), 1, GL_FALSE, &projection[0][0]);
  glUniformMatrix4fv(glGetUniformLocation(program, "view"), 1, GL_FALSE, &view[0][0]);
  glUniform1i(glGetUniformLocation(program, "skybox"), 1);


  glBindVertexArray(ctx.skyboxVAO);
  glDrawArrays(GL_TRIANGLES, 0, 36);
  glBindVertexArray(ctx.defaultVAO);


  glUseProgram(0);

}

// MODIFY THIS FUNCTION
void drawMesh(Context &ctx, GLuint program, const MeshVAO &meshVAO)
{

    //GLuint texture;
    // Define uniforms
    glm::mat4 model = trackballGetRotationMatrix(ctx.trackball);
    glm::mat4 view = glm::mat4();
    //glm::mat4 translat = glm::translate(glm::mat4(), glm::vec3(0.0f, 0.0f, -1.0f));

    glfwSetScrollCallback(ctx.window, scroll_callback);

    glm::mat4 projection = glm::ortho(-ctx.aspect, ctx.aspect, -1.0f, 1.0f, -10.0f, 10.0f);
    glm::mat4 mv = view * model;
    glm::vec3 zoom = glm::vec3(zoomFactor);
    mv = glm::scale(mv, zoom);
    glm::mat4 mvp = projection * mv;
    glm::mat3 ModelView3x3Matrix = glm::mat3(mv);
    // ...

    //glm::vec3 lpos = glm::vec3(1.2f, 1.5f, 3.0f);
    glm::vec3 lcol = glm::vec3(250.0f);

    std::vector<glm::vec3> lpos;

    lpos.push_back(glm::vec3(5.2f, 1.5f, 3.0f));
    lpos.push_back(glm::vec3(-5.2f, 2.5f, 3.0f));

    // Activate program
    glUseProgram(program);

    // Bind textures

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, ctx.a_text);

    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, ctx.m_text);

    glActiveTexture(GL_TEXTURE4);
    glBindTexture(GL_TEXTURE_2D, ctx.n_text);

    glActiveTexture(GL_TEXTURE5);
    glBindTexture(GL_TEXTURE_2D, ctx.r_text);

    glActiveTexture(GL_TEXTURE6);
    glBindTexture(GL_TEXTURE_2D, ctx.ao_text);





    // Pass uniforms
    glUniformMatrix4fv(glGetUniformLocation(program, "model"), 1, GL_FALSE, &model[0][0]);
    glUniformMatrix4fv(glGetUniformLocation(program, "u_mv"), 1, GL_FALSE, &mv[0][0]);
    glUniformMatrix4fv(glGetUniformLocation(program, "u_mvp"), 1, GL_FALSE, &mvp[0][0]);
    glUniformMatrix3fv(glGetUniformLocation(program, "mv3x3"), 1, GL_FALSE, &ModelView3x3Matrix[0][0]);
    glUniform1f(glGetUniformLocation(program, "u_time"), ctx.elapsed_time);
    glUniform3f(glGetUniformLocation(program, "lightPositions1"), lpos[0].x, lpos[0].y, lpos[0].z);
    glUniform3f(glGetUniformLocation(program, "lightPositions2"), lpos[1].x, lpos[1].y, lpos[1].z);
    glUniform3f(glGetUniformLocation(program, "lightColors"), lcol.x, lcol.y, lcol.z);
    glUniform1i(glGetUniformLocation(program, "albedoMap"), ctx.a_text);
    glUniform1i(glGetUniformLocation(program, "normalMap"), ctx.n_text);
    glUniform1i(glGetUniformLocation(program, "metallicMap"), ctx.m_text);
    glUniform1i(glGetUniformLocation(program, "roughnessMap"), ctx.r_text);
    glUniform1i(glGetUniformLocation(program, "aoMap"), ctx.ao_text);



    // ...

    // Draw!
    glBindVertexArray(meshVAO.vao);
    glDrawElements(GL_TRIANGLES, meshVAO.numIndices, GL_UNSIGNED_INT, 0);
    glBindVertexArray(ctx.defaultVAO);

}

void display(Context &ctx)
{
    glClearColor(0.2, 0.2, 0.2, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glEnable(GL_DEPTH_TEST); // ensures that polygons overlap correctly
    glDepthMask(GL_FALSE);
    drawSkybox(ctx, ctx.sb_program);

    glDepthMask(GL_TRUE);
    drawMesh(ctx, ctx.program, ctx.meshVAO);
}

void reloadShaders(Context *ctx)
{
    glDeleteProgram(ctx->program);
    ctx->program = loadShaderProgram(shaderDir() + "mesh.vert",
                                     shaderDir() + "mesh.frag");
}

void mouseButtonPressed(Context *ctx, int button, int x, int y)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        ctx->trackball.center = glm::vec2(x, y);
        trackballStartTracking(ctx->trackball, glm::vec2(x, y));
    }
}

void mouseButtonReleased(Context *ctx, int button, int x, int y)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        trackballStopTracking(ctx->trackball);
    }
}

void moveTrackball(Context *ctx, int x, int y)
{
    if (ctx->trackball.tracking) {
        trackballMove(ctx->trackball, glm::vec2(x, y));
    }
}

void errorCallback(int /*error*/, const char* description)
{
    std::cerr << description << std::endl;
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    // Forward event to GUI
    ImGui_ImplGlfwGL3_KeyCallback(window, key, scancode, action, mods);
    if (ImGui::GetIO().WantCaptureKeyboard) { return; }  // Skip other handling

    Context *ctx = static_cast<Context *>(glfwGetWindowUserPointer(window));
    if (key == GLFW_KEY_R && action == GLFW_PRESS) {
        reloadShaders(ctx);
    }
}

void charCallback(GLFWwindow* window, unsigned int codepoint)
{
    // Forward event to GUI
    ImGui_ImplGlfwGL3_CharCallback(window, codepoint);
    if (ImGui::GetIO().WantTextInput) { return; }  // Skip other handling
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    // Forward event to GUI
    ImGui_ImplGlfwGL3_MouseButtonCallback(window, button, action, mods);
    if (ImGui::GetIO().WantCaptureMouse) { return; }  // Skip other handling

    double x, y;
    glfwGetCursorPos(window, &x, &y);

    Context *ctx = static_cast<Context *>(glfwGetWindowUserPointer(window));
    if (action == GLFW_PRESS) {
        mouseButtonPressed(ctx, button, x, y);
    }
    else {
        mouseButtonReleased(ctx, button, x, y);
    }
}

void cursorPosCallback(GLFWwindow* window, double x, double y)
{
    if (ImGui::GetIO().WantCaptureMouse) { return; }  // Skip other handling

    Context *ctx = static_cast<Context *>(glfwGetWindowUserPointer(window));
    moveTrackball(ctx, x, y);
}

void resizeCallback(GLFWwindow* window, int width, int height)
{
    Context *ctx = static_cast<Context *>(glfwGetWindowUserPointer(window));
    ctx->width = width;
    ctx->height = height;
    ctx->aspect = float(width) / float(height);
    ctx->trackball.radius = double(std::min(width, height)) / 2.0;
    ctx->trackball.center = glm::vec2(width, height) / 2.0f;
    glViewport(0, 0, width, height);
}


int main(void)
{
    Context ctx;

    bool show_spec = true;
    bool show_diff = true;
    bool show_amb = true;
    bool show_normals = true;
    bool gamma_on = true;
    bool alb_on = true;

    glm::vec3 ambcol = glm::vec3(0.05);
    glm::vec3 speccol = glm::vec3(0.04);
    float specpow = 1.0f;
    float metalv = 1.0f;




    // Create a GLFW window
    glfwSetErrorCallback(errorCallback);
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    ctx.width = 500;
    ctx.height = 500;
    ctx.aspect = float(ctx.width) / float(ctx.height);
    ctx.window = glfwCreateWindow(ctx.width, ctx.height, "Model viewer", nullptr, nullptr);
    glfwMakeContextCurrent(ctx.window);
    glfwSetWindowUserPointer(ctx.window, &ctx);
    glfwSetKeyCallback(ctx.window, keyCallback);
    glfwSetCharCallback(ctx.window, charCallback);
    glfwSetMouseButtonCallback(ctx.window, mouseButtonCallback);
    glfwSetCursorPosCallback(ctx.window, cursorPosCallback);
    glfwSetFramebufferSizeCallback(ctx.window, resizeCallback);

    // Load OpenGL functions
    glewExperimental = true;
    GLenum status = glewInit();
    if (status != GLEW_OK) {
        std::cerr << "Error: " << glewGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    std::cout << "OpenGL version: " << glGetString(GL_VERSION) << std::endl;

    // Initialize GUI
    ImGui::CreateContext();
    ImGui_ImplGlfwGL3_Init(ctx.window, false /*do not install callbacks*/);


    // Initialize rendering
    glGenVertexArrays(1, &ctx.defaultVAO);
    glBindVertexArray(ctx.defaultVAO);
    glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
    init(ctx);

    // Start rendering loop
    while (!glfwWindowShouldClose(ctx.window)) {

        glfwPollEvents();
        ctx.elapsed_time = glfwGetTime();
        ImGui_ImplGlfwGL3_NewFrame();

        // 1. Show a simple window.
        // Tip: if we don't call ImGui::Begin()/ImGui::End() the widgets automatically appears in a window called "Debug".
        {
            //static float f = 0.0f;
            //static int counter = 0;
            ImGui::Text("Texture Control");                           // Display some text (you can use a format string too)
            ImGui::SliderFloat("Roughness Manual", &specpow, 0.0f, 1.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
            ImGui::SliderFloat("Metallicness Manual", &metalv, 0.0f, 1.0f); // Edit 3 floats representing a color

            ImGui::Checkbox("Metallic from Texture", &show_spec);
            ImGui::Checkbox("Roughness from Texture", &show_diff);
            ImGui::Checkbox("Show AO", &show_amb);
            ImGui::Checkbox("Show Normals", &show_normals);
            ImGui::Checkbox("Gamma Correction On", &gamma_on);
            ImGui::Checkbox("Albedo On", &alb_on);

            ImGui::Text("The Sliders only work, if you turn off the Textures!");

            //ImGui::Checkbox("Cubemap On/Off", &cubemap_on);
            //ImGui::SliderInt("Index", &cmpow, 1, 8);
        }



        glUniform1f(glGetUniformLocation(ctx.program, "specular_power"), specpow);
        glUniform1f(glGetUniformLocation(ctx.program, "metalV"), metalv);
        glUniform1i(glGetUniformLocation(ctx.program, "show_specular"), show_spec);
        glUniform1i(glGetUniformLocation(ctx.program, "show_diffuse"), show_diff);
        glUniform1i(glGetUniformLocation(ctx.program, "show_ambient"), show_amb);
        glUniform1i(glGetUniformLocation(ctx.program, "gamma_on"), gamma_on);
        glUniform1i(glGetUniformLocation(ctx.program, "normals_on"), show_normals);
        glUniform1i(glGetUniformLocation(ctx.program, "alb_on"), alb_on);


        display(ctx);
        ImGui::Render();
        glfwSwapBuffers(ctx.window);

    }


    // Shutdown
    ImGui_ImplGlfwGL3_Shutdown();
    //ImGui::DestroyContext(ctx);
    glfwDestroyWindow(ctx.window);
    glfwTerminate();
    std::exit(EXIT_SUCCESS);
}
