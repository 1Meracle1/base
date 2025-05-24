#include "include/array.h"
#include "include/defines.h"
#include "include/memory.h"
#include "include/defer.h"
#include "include/assert.h"
#include "include/slice.h"
#include "include/string.h"
#include <cstdio>
#include <iostream>

int main()
{
    VirtualArena arena{};
    Allocator*   allocator = &arena;

    String str = String::from_utf8_lossy(
        allocator, "In the quiet twilight, dreams unfold, soft whispers of a story untold.\n"
                   "ćeść panśtwu\n"
                   "月明かりが静かに照らし出し、夢を見る心の奥で詩が静かに囁かれる\n"
                   "Stars collide in the early light of hope, echoing the silent call of the night.\n"
                   "夜の静寂、希望と孤独が混ざり合うその中で詩が永遠に続く\n");
    for (u32 codepoint : str)
    {
        std::cout << std::hex << codepoint << (codepoint == cast(u32)0xA ? '\n' : ' ');
    }
    std::cout << '\n' << str << '\n';
}