#include "mesh.h"

#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <algorithm> 

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

using namespace std;

namespace {
struct Edge {
  int u;
  int v;
  Edge(int _u, int _v) {
    if (_u > _v)
      std::swap(_u, _v);
    u = _u;
    v = _v;
  }
  bool operator<(const Edge &edge) const {
    if (u < edge.u)
      return true;
    else if (u > edge.u)
      return false;
    return v < edge.v;
  }
};

struct Face {
  int u;
  int v;
  int w;
  Face(int _u, int _v, int _w) {
    u = _u;
    v = _v;
    w = _w;
  }
  bool operator<(const Face &face) const {
    auto ua = u;
    auto va = v;
    auto wa = w;
    auto ub = face.u;
    auto vb = face.v;
    auto wb = face.w;
    if (wa > va) {
      std::swap(wa, va);
    }
    if (va > ua) {
      std::swap(va, ua);
    }
    if (wa > va) {
      std::swap(wa, va);
    }
    if (wb > vb) {
      std::swap(wb, vb);
    }
    if (vb > ub) {
      std::swap(vb, ub);
    }
    if (wb > vb) {
      std::swap(wb, vb);
    }
    if (ua < ub)
      return true;
    else if (ua > ub)
      return false;
    if (va < vb)
      return true;
    else if (va > vb)
      return false;
    return wa < wb;
  }
};
struct vec3compare {
  bool operator()(const glm::vec3 &v0, const glm::vec3 &v1) const {
    if (v0.x < v1.x)
      return true;
    if (v0.x > v1.x)
      return false;
    if (v0.y < v1.y)
      return true;
    if (v0.y > v1.y)
      return false;
    return v0.z < v1.z;
  }
};
}  // namespace

void Mesh::initFromVectors(const std::vector<glm::vec3> &vertices,
                           const std::vector<glm::i32vec3> &faces) {
  _vertices = vertices;
  _faces = faces;
}

void Mesh::loadFromFile(const std::string &filePath) {
  tinyobj::attrib_t attrib;
  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> materials;

  // QFileInfo info(QString(filePath.c_str()));
  std::string err;
  bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &err,
                              filePath.c_str(), (filePath + "/").c_str(), true);
  if (!err.empty()) {
    std::cerr << err << std::endl;
  }

  if (!ret) {
    std::cerr << "Failed to load/parse .obj file" << std::endl;
    return;
  }

  for (size_t s = 0; s < shapes.size(); s++) {
    size_t index_offset = 0;
    for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
      unsigned int fv = shapes[s].mesh.num_face_vertices[f];

      glm::i32vec3 face;
      for (size_t v = 0; v < fv; v++) {
        tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];

        face[v] = idx.vertex_index;
      }
      _faces.push_back(face);

      index_offset += fv;
    }
  }
  for (size_t i = 0; i < attrib.vertices.size(); i += 3) {
    _vertices.emplace_back(attrib.vertices[i], attrib.vertices[i + 1],
                           attrib.vertices[i + 2]);
  }
  std::cout << "Loaded " << _faces.size() << " faces and " << _vertices.size()
            << " vertices" << std::endl;

  // TODO: Convert the mesh into your own data structure, if necessary.
}

void Mesh::saveToFile(const std::string &filePath) {
  // TODO: Convert your data structure back to the basic format, if necessary.

  std::ofstream outfile;
  outfile.open(filePath);

  // Write vertices
  for (size_t i = 0; i < _vertices.size(); i++) {
    const glm::vec3 &v = _vertices[i];
    outfile << "v " << v[0] << " " << v[1] << " " << v[2] << std::endl;
  }

  // Write faces
  for (size_t i = 0; i < _faces.size(); i++) {
    const glm::i32vec3 &f = _faces[i];
    outfile << "f " << (f[0] + 1) << " " << (f[1] + 1) << " " << (f[2] + 1)
            << std::endl;
  }

  outfile.close();
}

// TODO: implement the operations
// TODO: mesh validation (tips: use assertions)
// Optional
void Mesh::validate() {
    // Set to track unique faces to detect duplicates
    std::set<Face> faceSet;
    // Check for duplicate faces
    for (const auto &f : _faces) {
        Face face(f[0], f[1], f[2]);
        // Insert the face into the set, if it's already present, it's a duplicate
        if (faceSet.find(face) != faceSet.end()) {
            std::cerr << "Error: Duplicate face found: (" << f[0] << ", " << f[1] << ", " << f[2] << ")" << std::endl;
            assert(false); // Duplicate face detected
        }
        faceSet.insert(face);
    }

    // Check for invalid vertex indices in faces
    for (const auto &f : _faces) {
        if (f[0] >= _vertices.size() || f[1] >= _vertices.size() || f[2] >= _vertices.size()) {
            std::cerr << "Error: Face (" << f[0] << ", " << f[1] << ", " << f[2] 
                      << ") contains invalid vertex indices." << std::endl;
            assert(false); // Invalid vertex index detected
        }
    }

    // Ensure each vertex is used in at least one face
    std::vector<bool> vertexUsed(_vertices.size(), false);
    for (const auto &f : _faces) {
        vertexUsed[f[0]] = true;
        vertexUsed[f[1]] = true;
        vertexUsed[f[2]] = true;
    }
    for (size_t i = 0; i < vertexUsed.size(); i++) {
        if (!vertexUsed[i]) {
            std::cerr << "Warning: Vertex " << i << " is not used in any face." << std::endl;
            // Not a critical error, so we don't assert here
        }
    }

    // Flipping Faces Check
    std::map<Edge, int> edge_count;
    for (const auto &f : _faces) {
        Edge e1(f[0], f[1]);
        Edge e2(f[1], f[2]);
        Edge e3(f[0], f[2]);

        edge_count[e1]++;
        edge_count[e2]++;
        edge_count[e3]++;
    }

    for (const auto &entry : edge_count) {
        auto edge = entry.first;
        auto count = entry.second;
        if (count > 2) {
            std::cerr << "Warning: Flipping face detected for edge (" << edge.u << ", " << edge.v << ")" << std::endl;
            // This is a sign of a flipping face, where the same edge appears more than twice.
        }
    }

    // Normals consistency check (Optional)
    std::vector<glm::vec3> normals(_faces.size());
    for (size_t i = 0; i < _faces.size(); i++) {
        const glm::vec3 &v0 = _vertices[_faces[i][0]];
        const glm::vec3 &v1 = _vertices[_faces[i][1]];
        const glm::vec3 &v2 = _vertices[_faces[i][2]];

        // Calculate face normal
        glm::vec3 normal = glm::normalize(glm::cross(v1 - v0, v2 - v0));
        normals[i] = normal;
    }

    // Connectivity check
    std::vector<bool> visited(_vertices.size(), false);
    std::queue<int> q;
    q.push(0); // Start from vertex 0
    visited[0] = true;

    while (!q.empty()) {
        int current = q.front();
        q.pop();

        for (const auto &f : _faces) {
            if (f[0] == current || f[1] == current || f[2] == current) {
                for (int v : {f[0], f[1], f[2]}) {
                    if (!visited[v]) {
                        visited[v] = true;
                        q.push(v);
                    }
                }
            }
        }
    }

    bool disconnected = false;
    for (size_t i = 0; i < visited.size(); i++) {
        if (!visited[i]) {
            std::cerr << "Warning: Vertex " << i << " is disconnected from the mesh." << std::endl;
            disconnected = true;
        }
    }
    if (!disconnected) {
        std::cout << "Mesh is fully connected." << std::endl;
    }

    // Self-intersection check (Optional, requires more complex geometry algorithms)
    // This is a placeholder for more advanced checks, like checking whether any two faces intersect improperly.

    // Boundary edge check (Optional, for open or closed mesh detection)
    for (const auto &entry : edge_count) {
        auto edge = entry.first;
        auto count = entry.second;
        if (count == 1) {
            std::cerr << "Warning: Boundary edge detected between vertices (" << edge.u << ", " << edge.v << ")" << std::endl;
            // This means the mesh is not closed (it has open edges).
        }
    }

    std::cout << "Mesh validation completed successfully." << std::endl;
}


// TODO: Loop Subdivision
void Mesh::subdivide(int iterations) {
  for (int it = 0; it < iterations; it++) {
    std::map<Edge, int> edge_vertex_map;  // Maps an edge to its midpoint index
    std::map<glm::vec3, int, vec3compare> index_map;  // Maps vertices to their indices
    std::vector<glm::vec3> vertices_new;  // Stores new vertices
    std::vector<glm::i32vec3> faces_new;  // Stores new faces

    // Helper function to get or create vertex index
    auto get_vertex_index = [&](const glm::vec3 &v) {
      if (!index_map.count(v)) {
        index_map[v] = int(vertices_new.size());
        vertices_new.push_back(v);
      }
      return index_map.at(v);
    };

    // Helper function to insert a new face
    auto insert_triangle = [&](const glm::vec3 v0, const glm::vec3 v1, const glm::vec3 v2) {
      faces_new.emplace_back(get_vertex_index(v0), get_vertex_index(v1), get_vertex_index(v2));
    };

    /*
     * This is a framework to help you merge identical vertices.
     * You can insert a triangle into the new mesh like this:
     * insert_triangle(
     *   glm::vec3{1.0f, 0.0f, 0.0f},
     *   glm::vec3{0.0f, 1.0f, 0.0f},
     *   glm::vec3{0.0f, 0.0f, 1.0f}
     * );
     * However, you still need to be careful on the triangle orientation.
     * */
    /*****************************/
    /* Write your code here. */

    auto add_edge_vertex = [&](int v1, int v2) {
        Edge edge(v1, v2); 
        if (!edge_vertex_map.count(edge)) {
            glm::vec3 mid_point = (_vertices[v1] + _vertices[v2]) * 0.5f;
            int edge_idx = get_vertex_index(mid_point);
            edge_vertex_map[edge] = edge_idx;  
            vertices_new[edge_idx] = mid_point;  
        } 
        return edge_vertex_map[edge]; 
    };

    // Loop over each face
    for (const auto &face : _faces) {
      int v0 = face[0];
      int v1 = face[1];
      int v2 = face[2];

      // Get midpoints of edges
      int e0 = add_edge_vertex(v0, v1);  // Midpoint of edge v0-v1
      int e1 = add_edge_vertex(v1, v2) ;  // Midpoint of edge v1-v2
      int e2 = add_edge_vertex(v0, v2);  // Midpoint of edge v2-v0

      // Create four new faces
      insert_triangle(_vertices[v0], vertices_new[e0], vertices_new[e2]);
      insert_triangle(vertices_new[e0], _vertices[v1], vertices_new[e1]);
      insert_triangle(vertices_new[e2], vertices_new[e1], _vertices[v2]);
      insert_triangle(vertices_new[e0], vertices_new[e1], vertices_new[e2]);
    }

    // Update the edge vertex positions based on adjacent vertices
  for (const auto &pair : edge_vertex_map) {
      const Edge &edge = pair.first;
      int edge_idx = pair.second;
      int v1 = edge.u;
      int v2 = edge.v;

      glm::vec3 new_pos;
      std::vector<int> adjacent_vertices;

      // Find adjacent faces to this edge to gather surrounding vertices
      for (const auto &face : _faces) {
        if ((face[0] == v1 && face[1] == v2) || (face[0] == v2 && face[1] == v1)) {
          adjacent_vertices.push_back(face[2]);
        } else if ((face[1] == v1 && face[2] == v2) || (face[1] == v2 && face[2] == v1)) {
          adjacent_vertices.push_back(face[0]);
        } else if ((face[2] == v1 && face[0] == v2) || (face[2] == v2 && face[0] == v1)) {
          adjacent_vertices.push_back(face[1]);
        }
      }

      // If there are exactly 2 adjacent vertices, this is a boundary edge
      if (adjacent_vertices.size() == 2) {
        glm::vec3 posA = _vertices[v1];
        glm::vec3 posB = _vertices[v2];
        glm::vec3 posC = _vertices[adjacent_vertices[0]];
        glm::vec3 posD = _vertices[adjacent_vertices[1]];
        new_pos = (3.0f / 8.0f) * posA + (3.0f / 8.0f) * posB +
                  (1.0f / 8.0f) * posC + (1.0f / 8.0f) * posD;
      } else {
        // This is a boundary edge, average the two vertices' positions
        new_pos = (_vertices[v1] + _vertices[v2]) * 0.5f;
      }

      vertices_new[edge_idx] = new_pos;
    }

    // Update original vertex positions based on adjacent edges
    for (size_t i = 0; i < _vertices.size(); ++i) {
      glm::vec3 new_pos = _vertices[i];
      int k = 0;  // number of adjacent vertices

      glm::vec3 sum_adj_pos(0.0f);
      std::vector<int> adj_vertices;  // Adjacent vertices list

      // Collect all adjacent vertices to vertex i
      for (size_t j = 0; j < _vertices.size(); ++j) {
          if (i != j) {  
              for (const auto &face : _faces) {
                  if ((face[0] == i && (face[1] == j || face[2] == j)) ||
                      (face[1] == i && (face[0] == j || face[2] == j)) ||
                      (face[2] == i && (face[0] == j || face[1] == j))) {
                      adj_vertices.push_back(j);  
                      break; 
                  }
              }
          }
      }

      k = adj_vertices.size();  // Number of adjacent vertices
      // Sum positions of adjacent vertices
      for (int adj : adj_vertices) {
        sum_adj_pos += _vertices[adj];
      }

      if (k == 2) {
        // Boundary vertex case, use the weighted formula
        new_pos = (3.0f / 4.0f) * _vertices[i] + (1.0f / 8.0f) * sum_adj_pos;
      } else {
        // Non-boundary vertex case, use general weighted formula
        float beta = (5.0 / 8.0 - pow(3.0 / 8.0 + cos(2 * M_PI / k) / 4.0, 2)) / k;
        new_pos = (1.0f - k * beta) * _vertices[i] + beta * sum_adj_pos;
      }
      vertices_new[get_vertex_index(_vertices[i])] = new_pos;
    }
    
    /*****************************/
    // Update the mesh with new vertices and faces
    _vertices = vertices_new;
    _faces = faces_new;
  }
}

glm::mat4 QMatrix(const glm::vec3 &p0,
                  const glm::vec3 &p1,
                  const glm::vec3 &p2) {
  /*
   * TODO: Implement this function to compute the Q matrix.
   * The Q matrix is defined with respect to the plane corresponding to the given triangle.
   * You are provided with the three vertices of a triangle.
   * Your task is to derive the plane equation and subsequently compute the Q matrix.
   * If the triangle is degenerate (collapsed), simply return a zero matrix.
   * Note: The resulting matrix should be symmetric.
   */
    // Compute the normal vector of the plane
    glm::dvec3 normal = glm::normalize(glm::cross(glm::dvec3(p1) - glm::dvec3(p0), glm::dvec3(p2) - glm::dvec3(p0)));
     
    // Check if the triangle is degenerate (collapsed)
    if (glm::length(normal) < 1e-4f) {
        return glm::mat4(0.0f);  // Return zero matrix for degenerate triangles
    }

    // Extract the components of the normal vector
    double A = normal.x;
    double B = normal.y;
    double C = normal.z;

    // Compute D using the plane equation: Ax + By + Cz + D = 0
    double D = -glm::dot(normal, glm::dvec3(p0));

    // Construct the Q matrix
    glm::mat4 Q = glm::mat4(
        A * A, A * B, A * C, A * D,
        A * B, B * B, B * C, B * D,
        A * C, B * C, C * C, C * D,
        A * D, B * D, C * D, D * D
    );

    return Q;
}

glm::vec3 GenerateMergedPoint(const glm::vec3 &p0,
                              const glm::vec3 &p1,
                              glm::mat4 Q) {
  /*
   * TODO: Generate the merged point using the Q matrix.
   * Provided your QMatrix function is correctly implemented, 
   * the Q matrix is guaranteed to be a symmetric semi-positive definite.
   * The error value of a particular position is defined by the formula x^TQx,
   * where 'x' represents a Homogeneous linear coordinate, expressed in 4 dimensions but interpreted in 3 dimensions.
   * Your task is to identify and return the position that yields the minimum error. 
   * If the position cannot be determined, return the midpoint between p0 and p1.
   */
    // Construct a 3x3 matrix from the top-left 3x3 submatrix of Q
    glm::mat3 Q_3x3(Q[0][0], Q[0][1], Q[0][2],
                    Q[1][0], Q[1][1], Q[1][2],
                    Q[2][0], Q[2][1], Q[2][2]);

    // Construct the 3D vector b from the top 3 elements of the last column of Q
    glm::vec3 b(-Q[0][3], -Q[1][3], -Q[2][3]);

    // Solve the linear system Q_3x3 * x = b
    glm::vec3 optimal_point;

    float det = glm::determinant(Q_3x3);
    
    if (glm::abs(det) > 1e-3f) {
        optimal_point = glm::inverse(Q_3x3) * b;
    } else {
        glm::mat3 pseudo_inverse = glm::transpose(Q_3x3) / (glm::dot(Q_3x3[0], Q_3x3[0]) + glm::dot(Q_3x3[1], Q_3x3[1]) + glm::dot(Q_3x3[2], Q_3x3[2]));
        optimal_point = pseudo_inverse * b;
    }

    return optimal_point;
}

// QEM Mesh Simplification
// You do not need to change this function under this framework.
void Mesh::simplify(int num_faces_to_remove) {
  std::vector<glm::vec3> vertices = _vertices;
  std::vector<int> leader(vertices.size());
  std::vector<glm::mat4> Qs(vertices.size(), glm::mat4{0.0f});
  std::vector<std::set<int>> adjacent_vertices(vertices.size());
  std::set<Edge> edges;
  std::vector<std::set<Face>> adjacent_faces(vertices.size());
  std::set<Face> faces;
  auto add_vertex = [&](const glm::vec3 &pos, const glm::mat4 &q) {
    auto index = int(leader.size());
    leader.push_back(index);
    Qs.push_back(q);
    vertices.push_back(pos);
    adjacent_vertices.emplace_back();
    adjacent_faces.emplace_back();
    return index;
  };

  for (size_t i = 0; i < vertices.size(); i++) {
    leader[i] = int(i);
  }

  std::function<int(int x)> find_leader;
  find_leader = [&leader, &find_leader](int x) -> int {
    return (x == leader[x]) ? x : (leader[x] = find_leader(leader[x]));
  };

  auto insert_face = [&](int u, int v, int w) {
    if (u == v || v == w || u == w)
      return;
    Face face(u, v, w);
    faces.insert(face);
    adjacent_faces[u].insert(face);
    adjacent_faces[v].insert(face);
    adjacent_faces[w].insert(face);
  };

  auto remove_face = [&](int u, int v, int w) {
    Face face(u, v, w);
    faces.erase(face);
    adjacent_faces[u].erase(face);
    adjacent_faces[v].erase(face);
    adjacent_faces[w].erase(face);
  };
  auto insert_edge = [&](int u, int v) {
    if (u == v)
      return;
    edges.insert(Edge(u, v));
    adjacent_vertices[u].insert(v);
    adjacent_vertices[v].insert(u);
  };
  auto remove_edge = [&](int u, int v) {
    edges.erase(Edge(u, v));
    adjacent_vertices[u].erase(v);
    adjacent_vertices[v].erase(u);
  };

  for (auto &face : _faces) {
    auto q = QMatrix(vertices[face.x], vertices[face.y], vertices[face.z]);
    Qs[face.x] += q;
    Qs[face.y] += q;
    Qs[face.z] += q;
    insert_face(face.x, face.y, face.z);
    insert_edge(face.x, face.y);
    insert_edge(face.y, face.z);
    insert_edge(face.x, face.z);
  }

  struct MergePack {
    float err;
    Edge e;
    glm::vec4 v;
    glm::mat4 q;
    bool operator<(const MergePack &pack) const {
      return err > pack.err;
    }
  };

  std::set<Edge> in_flight;
  std::priority_queue<MergePack> packs;
  auto add_pack = [&](const MergePack &pack) {
    if (in_flight.count(pack.e))
      return;
    in_flight.insert(pack.e);
    packs.push(pack);
  };

  auto new_pack = [&](const Edge &edge) {
    MergePack pack{
        0.0f, edge, {0.0f, 0.0f, 0.0f, 1.0f}, Qs[edge.u] + Qs[edge.v]};
    auto res = GenerateMergedPoint(vertices[edge.u], vertices[edge.v], pack.q);
    pack.v = glm::vec4{res, 1.0f};
    pack.err = glm::dot(pack.v, pack.q * pack.v);
    add_pack(pack);
  };

  for (auto &edge : edges) {
    new_pack(edge);
  }

  int target_face_num = int(_faces.size()) - num_faces_to_remove;
  std::set<Face> removing_faces;
  std::set<Edge> removing_edges;
  std::set<Face> new_faces;
  std::set<Edge> new_edges;
  while (target_face_num < faces.size()) {
    auto merge_pack = packs.top();
    packs.pop();
    auto edge = merge_pack.e;
    in_flight.erase(edge);
    if (leader[edge.u] != edge.u || leader[edge.v] != edge.v) {
      continue;
    }
    int index = add_vertex(glm::vec3{merge_pack.v}, merge_pack.q);
    leader[edge.u] = index;
    leader[edge.v] = index;
    removing_faces.clear();
    removing_edges.clear();
    new_edges.clear();
    new_faces.clear();
    for (auto ni : adjacent_vertices[edge.u]) {
      removing_edges.insert(Edge(edge.u, ni));
      new_edges.insert(Edge(find_leader(edge.u), find_leader(ni)));
    }
    for (auto ni : adjacent_vertices[edge.v]) {
      removing_edges.insert(Edge(edge.v, ni));
      new_edges.insert(Edge(find_leader(edge.v), find_leader(ni)));
    }
    for (auto face : adjacent_faces[edge.u]) {
      removing_faces.insert(face);
      new_faces.insert(
          Face(find_leader(face.u), find_leader(face.v), find_leader(face.w)));
    }
    for (auto face : adjacent_faces[edge.v]) {
      removing_faces.insert(face);
      new_faces.insert(
          Face(find_leader(face.u), find_leader(face.v), find_leader(face.w)));
    }
    for (auto face : removing_faces) {
      remove_face(face.u, face.v, face.w);
    }
    for (auto e : removing_edges) {
      remove_edge(e.u, e.v);
    }
    for (auto face : new_faces) {
      insert_face(face.u, face.v, face.w);
    }
    for (auto e : new_edges) {
      if (e.u == e.v)
        continue;
      insert_edge(e.u, e.v);
      new_pack(e);
    }
  }

  std::map<glm::vec3, int, vec3compare> index_map;
  std::vector<glm::vec3> vertices_new;
  std::vector<glm::i32vec3> faces_new;

  auto get_vertex_index = [&](const glm::vec3 &v) {
    if (!index_map.count(v)) {
      index_map[v] = int(vertices_new.size());
      vertices_new.push_back(v);
    }
    return index_map.at(v);
  };

  auto insert_triangle = [&](const glm::vec3 v0, const glm::vec3 v1,
                             const glm::vec3 v2) {
    faces_new.emplace_back(get_vertex_index(v0), get_vertex_index(v1),
                           get_vertex_index(v2));
  };

  for (auto &face : faces) {
    insert_triangle(vertices[face.u], vertices[face.v], vertices[face.w]);
  }

  _vertices = vertices_new;
  _faces = faces_new;
}
