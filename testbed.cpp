#include "include/array.h"
#include "include/defines.h"
#include "include/memory.h"
#include "include/defer.h"
#include "include/assert.h"
#include "include/slice.h"
#include <cstdio>

int main()
{
    // MemoryBlock* mem_block = MemoryBlock::init(Megabytes(1), Megabytes(8));
    // Assert(mem_block != nullptr);
    // defer(
    //     [mem_block]
    //     {
    //         printf("hello, ");
    //         printf("world!\n");
    //         mem_block->deinit();
    //     });

    // u64 pos          = 0;
    // u64 m_block_size = Megabytes(8);
    // Assert(is_power_of_two(pos));
    // u64 aligned_pos = align_formula(pos, m_block_size);
    // std::printf("aligned_pos: %zu\n", aligned_pos);

    VirtualArena arena{};
    Allocator*   allocator = &arena;

    // u64  size_to_alloc = sizeof(u64) * 10000;
    // u64* ints          = cast(u64*) arena.alloc(size_to_alloc, alignof(i64));
    // Assert(ints != nullptr);
    // Assert(ints[0] == 0);
    // Assert(ints[9999] == 0);

    // arena.reset_to(0);

    // ints = cast(u64*) arena.alloc(size_to_alloc, alignof(i64));
    // Assert(ints != nullptr);
    // Assert(ints[0] == 0);
    // Assert(ints[9999] == 0);

    // {
    //     auto temp_arena = arena.temp_arena_guard();
    //     ints            = cast(u64*) arena.alloc(size_to_alloc, alignof(i64));
    //     Assert(ints != nullptr);
    //     Assert(ints[0] == 0);
    //     Assert(ints[9999] == 0);
    // }

    // Allocator* allocator    = &arena;
    // auto       another_ints = allocator->alloc<u8>(Megabytes(7));
    // Assert(another_ints[9999] == 0);
    // auto another_ints1 = allocator->alloc<u8>(Megabytes(7));
    // Assert(another_ints1[9999] == 0);

    Slice str_slice = "hello";
    for (auto c : str_slice)
    {
        printf("%c\n", c);
    }

    Array<int> array{allocator, 10};

    for(std::size_t i = 0; i < 1000000;)
    {   
        auto new_i = GrowthFormulaDefault{}(i);
        printf("%zu - %zu\n", i, new_i);
        i = new_i;
    }
}