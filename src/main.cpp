#include <chrono>
#include <glm/glm.hpp>
#include <iostream>
#include <string>

#include "mesh.h"

using namespace std;
using namespace std::chrono;

int main(int argc, char *argv[]) {
  if (argc != 5) {
    cerr << "Usage:" << endl;
    cerr << "  Arguments <input .obj file path> <output .obj file path> "
            "<method (subdivide, simplify)> <parameter>"
         << endl;
    return 1;
  }
  string infile = argv[1];        // Input .obj file path
  string outfile = argv[2];       // Output .obj file path
  string method = argv[3];        // subdivide/simplify
  int parameter = stoi(argv[4]);  // # of iterations for 'subdivide' / # of
                                  // faces to remove for 'simplify'

  Mesh m;
  m.loadFromFile(infile);

  auto t0 = high_resolution_clock::now();

  if (method == "subdivide")
    m.subdivide(parameter);
  else if (method == "simplify")
    m.simplify(parameter);
  else {
    cerr << "Error: Unknown method \"" << method << "\"" << endl;
  }

  auto t1 = high_resolution_clock::now();
  auto duration = duration_cast<milliseconds>(t1 - t0).count();
  cout << "Execution took " << duration << " milliseconds." << endl;

  m.validate();  // Validate the processed mesh
  m.saveToFile(outfile);

  return 0;
}
