#!/bin/fish

time clang++ -std=c++20 -Wall -Wextra -Wshadow -fno-rtti -fno-exceptions -fuse-ld=mold -fsanitize=address,leak,undefined -fno-omit-frame-pointer -O0 -g testbed.cpp -o testbed || exit
./testbed