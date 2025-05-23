#include "include/array.h"
#include "include/defines.h"
#include "include/memory.h"
#include "include/defer.h"
#include "include/assert.h"
#include "include/slice.h"
#include "include/string.h"
#include <cstdio>

int main()
{
    VirtualArena arena{};
    Allocator*   allocator = &arena;

    String str = String::from_utf8_lossy(allocator, "hello");
}