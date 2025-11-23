#!/bin/bash

# Paths
CASADI_INC=../deps/casadi_install/include
CASADI_LIB=../deps/casadi_install/lib
SRC_FILE=../src/cpp/test_casadi.cpp
SRC_FILE2=../src/cpp/example_ds.cpp
OUT_FILE=../src/cpp/test_casadi
OUT_FILE2=../src/cpp/example_ds

#include directory to the list of directories to be saeerched for headerfiles during preprocessing = -I
#inlucde directory to the shared library to link

g++ "$SRC_FILE" -I"$CASADI_INC" -L"$CASADI_LIB" -lcasadi -std=c++17 -Wl,-rpath,"$CASADI_LIB" -o "$OUT_FILE"
g++ "$SRC_FILE2" -I"$CASADI_INC" -L"$CASADI_LIB" -lcasadi -std=c++17 -Wl,-rpath,"$CASADI_LIB" -o "$OUT_FILE"

export LD_LIBRARY_PATH="$CASADI_LIB:$LD_LIBRARY_PATH"

# Run
"$OUT_FILE"
"$OUT_FILE2"
