#ifndef MESH_H
#define MESH_H

#include <glm/glm.hpp>
#include <string>
#include <vector>

class Mesh {
 public:
  void initFromVectors(const std::vector<glm::vec3> &vertices,
                       const std::vector<glm::i32vec3> &faces);
  void loadFromFile(const std::string &filePath);
  void saveToFile(const std::string &filePath);
  void subdivide(int iterations);
  void simplify(int num_faces_to_remove);
  void validate();
  std::vector<glm::vec3> getVertices() const {
    return _vertices;
  }
  std::vector<glm::i32vec3> getFaces() const {
    return _faces;
  }
  int add_edge_vertexes(int vertex_id1, int vertex_id2, int vertex_id3, 
                              std::vector<std::vector<std::pair<int, glm::i32vec3>>> &edge_vertexes,
                              std::vector<glm::vec3> &vertices_new);
 private:
  std::vector<glm::vec3> _vertices;
  std::vector<glm::i32vec3> _faces;
  std::tuple<int, int, int> normalize_face(int v0, int v1, int v2);
};
#endif  // MESH_H
