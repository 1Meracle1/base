#!/bin/fish

time clang++ -std=c++20 -Wall -Wextra -Wshadow -fno-rtti -fno-exceptions -fuse-ld=mold -O3 -march=native -flto -fwhole-program-vtables -ftree-vectorize testbed.cpp -o testbed || exit
./testbed