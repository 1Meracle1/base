#include "include/defines.h"
#include "include/memory.h"
#include "include/defer.h"
#include "include/assert.h"
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

    u64  size_to_alloc = sizeof(u64) * 10000;
    u64* ints          = cast(u64*) arena.alloc(size_to_alloc, alignof(i64));
    Assert(ints != nullptr);
    Assert(ints[0] == 0);
    Assert(ints[9999] == 0);

    // arena.reset_to(0);

    ints = cast(u64*) arena.alloc(size_to_alloc, alignof(i64));
    Assert(ints != nullptr);
    Assert(ints[0] == 0);
    Assert(ints[9999] == 0);

    // arena.reset_to(0);

    ints = cast(u64*) arena.alloc(size_to_alloc, alignof(i64));
    Assert(ints != nullptr);
    Assert(ints[0] == 0);
    Assert(ints[9999] == 0);

    Allocator* allocator        = &arena;
    u64        another_ints_len = 1000000;
    // i64*      another_ints     = cast(i64*) allocator.alloc(sizeof(i64) * another_ints_len, alignof(i64));
    auto another_ints = allocator->alloc<i64>(another_ints_len);
    Assert(another_ints != nullptr);
    Assert(another_ints[0] == 0);
    Assert(another_ints[another_ints_len - 1] == 0);
}