[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/2fXFzLJ8)
# Advanced Computer Graphics Lab 1: Mesh Processing
**Released date:** 2024/9/14

**Due date:** 2024/10/8 23:59

## Introduction

In this lab, you will take a step into triangular meshes and implement some basic but useful (for your final project) algorithms of mesh processing in C++.

In this lab, you need to submit (commit and push to your repo) the corresponding codes and results.

The submission time depends on the last ***push*** to the GitHub-classroom repo. Considering the network issue, please don't rush DDL :)

If you have any questions in finishing this homework, please contact TA (wzf22@mails.tsinghua.edu.cn, or the 4th guy in the order of entering WeChat Group) as soon as possible.

To begin, please first clone this repo to local:
```
git clone {your-repo-name}.git --recursive
or
git clone {your-repo-name}.git
git submodule --init
```

Then compile the code by:
```
sh compile.sh (For Linux)
compile.bat (For Windows)
```
This command will automatically create a folder `build` and build the project based on `CMakeList.txt`. Get into this folder, you will see the executable file `mesh_main` (`mesh_main.exe` for Windows), a copy of `meshes` folder containing the test meshes and an empty `results` folder for you to export processed meshes.

(**Notice:** For submission, please place your processed meshes in `results` folder in the ***root*** directory)

## Requirements

You need to implement 2 algorithms for mesh processing, Mesh Subdivision and Mesh Simplification. We've provided a codebase for you and your implementation should build on it.

There are only 3 main source files for this codebase (all in `src` folder): `main.cpp`, `mesh.cpp` and `mesh.h`. The code is based on a class `Mesh`, which storages the mesh data and supports all mesh processing functions. We've implemented `Mesh::loadFromFile` and `Mesh::saveToFile` (using `ting_obj_loader`) for you to read and write `.obj` file easily. There are only vertices and faces storaged, so if necessary, you can use any other data structures for your convenience. In this lab, you should finish the member functions `validate`, `subdivide` and `simplify` according to the provided codebase.

Feel free for class `Mesh` and please **DON'T** modify `main.cpp` !!!

Besides the code submission, you need to run some commands to export meshes (specified below). In the report, you should briefly introduce your implementation (e.g. the data structure you used for more convenient processing) and take a **screenshot** of your output meshes.

p.s. There are a few example meshes in the folder `assets/extra_meshes/`, feel free to play with them!

### Mesh Validation (0pt but ***important***)

Customized meshes usually suffer from some issues. For example, consider the tetrahedron mesh `example.obj` below:
```
v 0 0 0
v 1 0 0
v 0 1 0
v 0 0 1

f 1 2 4
f 1 3 2
f 1 4 3
f 2 3 4
f 2 3 4 # replicate face
```

It has a replicate face `f 2 3 4`. When you open it in some mesh software, it seems the same as the one without replicate faces. However, suppose a surface point sampling algorithm applies on this mesh, this replicate face will be counted twice and double points will be sampled on this triangle.

Therefore, you should implement `Mesh::validate` before implementing any mesh operations so that you can run it to detect whether your mesh operations are mangling your data structures (tips: you can use `assert` statements to sanity-check the mesh). For grading, we have another validation function but we didn't release it. Make your `validate` function as thorough and comprehensive as possible—doing so will save you time and make it easier for the TAs to help you in hours.

Here are some points that may help:
* replicate faces
* flipping faces
* some loopholes occur, which should be compact
* ...

If you come up with other points, welcome to the issue board!

### Mesh Subdivision (50pts)

Mesh subdivision is often used to smooth a triangular mesh, by iteratively adding the number of triangles. In this section, you need to implement a simple subdivision algorithm called Loop Subdivision. To be more specific, in one iteration, every face is split into 4 faces by connecting the intermediate vertices in the edges. Let's call the original vertices even vertices and the created intermediate vertices as odd vertices. To smooth the surface, we will recompute these vertex positions. The following figure shows the specific weights for the algorithm. To learn more, see the [paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/thesis-10.pdf) or [lecture slides](https://www.cs.cmu.edu/afs/cs/academic/class/15462-s14/www/lec_slides/Subdivision.pdf).

![](https://github.com/Yao-class-graphics-studio/assignment-1-mesh/blob/main/assets/figures/subdivide.jpg)

After implementation, please run the following commands and submit the corresponding processed meshes with your codes.
```
./mesh_main meshes/icosahedron_input.obj results/icosahedron_subdivided_1.obj subdivide 1
./mesh_main meshes/icosahedron_input.obj results/icosahedron_subdivided_3.obj subdivide 3
./mesh_main meshes/bunny_200.obj results/bunny_200_subdivided_1.obj subdivide 1
./mesh_main meshes/bunny_200.obj results/bunny_200_subdivided_3.obj subdivide 3
```

Scores:

* Correct subdivision algorithm (25pts)
* Valid mesh output (9pts)
* Algorithm efficiency (16pts)

### Mesh Simplification (50pts)

![](https://github.com/Yao-class-graphics-studio/assignment-1-mesh/blob/main/assets/figures/simplify.jpg)

In this section, you need to implement QEM Mesh Simplification algorithm. A sequence of processing results is shown above (Figure 4 in the [paper](http://www.cs.cmu.edu/~./garland/quadrics/quadrics.html)).

The algorithm mainly focuses on the Edge Collapse operation. There are many implementation details so you need to code it carefully. Additionally, you can ignore the parameter `t` in the paper (which is used to collapse the near but not adjacent vertices) since we won't test such examples.

After implementation, please run the following commands and submit the corresponding processed meshes with your codes.
```
./mesh_main meshes/bunny.obj results/bunny_simplified_10000.obj simplify 10000
./mesh_main meshes/bunny.obj results/bunny_simplified_17000.obj simplify 17000
./mesh_main meshes/cow.obj results/cow_simplified_5000.obj simplify 5000
./mesh_main meshes/cow.obj results/cow_simplified_5600.obj simplify 5600
```

Scores:

* Correct edge collapse algorithm (25pts)
* Valid mesh output (9pts)
* Algorithm efficiency (16pts)

## Resources

To inspect and interact with input/output meshes, it’s worth getting familiar with some sort of 3D model viewing/editing software. One good option is [MeshLab](https://www.meshlab.net/), which is free, lightweight, and provides a ton of useful functionality, including easy conversions between many different 3D file formats. If you’re already familiar with another 3D package such as [Maya](https://www.autodesk.com/products/maya/overview) or [Blender](https://www.blender.org/), those are perfectly fine, too.

Further readings about other mesh processing algorithms:
* [Catmull-Clark Subdivision](https://people.eecs.berkeley.edu/~sequin/CS284/PAPERS/CatmullClark_SDSurf.pdf)
* [A Remeshing Approach to Multiresolution Modeling](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.86.7407&rep=rep1&type=pdf)
* [Bilateral Mesh Denoising](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.14.1033&rep=rep1&type=pdf)
* Watertight Manifold Surface Generation: [Manifold](https://github.com/hjwdzh/Manifold) & [ManifoldPlus](https://github.com/hjwdzh/ManifoldPlus)
* [Fast Tetrahedral Meshing in the Wild](https://arxiv.org/pdf/1908.03581.pdf)

## Acknowledgement

Part of our code is inherited from [brown-cs-224-Mesh](https://github.com/brown-cs-224/Mesh-Stencil). We are grateful to the authors for releasing their code. 
"# ACG-hw1" 
