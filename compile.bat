@echo off
if exist "build" (rd /s/q "build") else (echo good!)

md build
cd build
cmake -G "MinGW Makefiles" -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ ..
make -j

md results