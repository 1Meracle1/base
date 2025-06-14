#!/bin/fish

# -msse2 or /arch:SSE2
# -mavx2 -mbmi -mbmi or /arch:AVX2
time clang++ -std=c++23 -Wall -Wextra -Wshadow -fno-rtti -fno-exceptions -fuse-ld=mold -fsanitize=address,leak,undefined -fno-omit-frame-pointer -O0 -g -msse2 -mavx2 -mbmi -mbmi testbed.cpp -o testbed || exit

time ./testbed