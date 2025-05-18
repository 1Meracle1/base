#ifndef MEMORY_H
#define MEMORY_H

#include "defines.h"
#include "assert.h"
#include <cstdlib>
#include <cstring>
#include <functional>
#include <utility>

constexpr u64 DEFAULT_ALIGNMENT = sizeof(rawptr) * 2;

struct Allocator
{
    virtual ~Allocator() {}

    template <typename T> T*   alloc(u64 count = 1) { return cast(T*) alloc(sizeof(T) * count, alignof(T)); }
    template <typename T> void free(T* ptr, u64 count = 1) { free(ptr, sizeof(T) * count); }

    virtual rawptr alloc(u64 size, u64 alignment) = 0;
    virtual void   free(rawptr ptr, u64 size)     = 0;
};

struct HeapAllocator : Allocator
{
    rawptr alloc(u64 size, u64 alignment) override { return std::aligned_alloc(alignment, size); }
    void   free(rawptr ptr, u64 size) override
    {
        (void)size;
        return std::free(ptr);
    }
};

enum class AllocationError
{
    None,
    Out_Of_Memory,
    Invalid_Argument,
    Invalid_Pointer,
};

// -----------------------------------------------------------------------------------
// MARK: Interface with OS
// -----------------------------------------------------------------------------------

#ifdef OS_LINUX
// clang-format off
    #include <sys/mman.h>
    #include <fcntl.h>
    #include <unistd.h>
    #include <errno.h>

    extern int getpagesize(void);
    extern int madvise(void* __addr, size_t __len, int __advice);

    static rawptr os_reserve(rawptr ptr, u64 size, AllocationError* err)
    {
        Assert(size > 0);
        auto mapped = mmap(ptr, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (mapped == MAP_FAILED)
        {
            if (err != nullptr)
            {
                if (errno == ENOMEM)
                {
                    *err = AllocationError::Out_Of_Memory;
                }
                else if (errno == EINVAL)
                {
                    *err = AllocationError::Invalid_Argument;
                }
            }
            return nullptr;
        }
        return mapped;
    }

    static AllocationError os_commit(rawptr data, u64 size)
    {
        Assert(size > 0);
        mprotect(data, size, PROT_READ | PROT_WRITE);
        if (errno == EINVAL)
        {
            return AllocationError::Invalid_Pointer;
        }
        else if (errno == ENOMEM)
        {
            return AllocationError::Out_Of_Memory;
        }
        return AllocationError::None;
    }

    static AllocationError os_decommit(rawptr data, u64 size)
    {
        Assert(size > 0);
        int res = mprotect(data, size, PROT_NONE);
        if (res == -1) 
        {
            if (errno == EINVAL) return AllocationError::Invalid_Pointer;
            else                 return AllocationError::Out_Of_Memory;
        }
    #define _MADV_FREE 8
        res = madvise(data, size, _MADV_FREE);
        if (res == -1) 
        {
            if (errno == EINVAL) return AllocationError::Invalid_Pointer;
            else                 return AllocationError::Out_Of_Memory;
        }
        return AllocationError::None;
    }

    static AllocationError os_release(rawptr data, u64 size) 
    { 
        Assert(size > 0);
        int res = munmap(data, size); 
        if (res == -1) 
        {
            if (errno == EINVAL) return AllocationError::Invalid_Pointer;
            else                 return AllocationError::Out_Of_Memory;
        }
        return AllocationError::None;
    }

    static u64 os_page_size() { return (u64)getpagesize(); }
// clang-format on
#else
#error Virtual memory OS allocation interface is not implemented on this platform
#endif

// -----------------------------------------------------------------------------------
// MARK: Utilities
// -----------------------------------------------------------------------------------

constexpr bool is_power_of_two(u64 x) { return (x & (x - 1)) == 0; }

constexpr u64 align_formula(u64 pos, u64 alignment) { return (pos + alignment - 1) & ~(alignment - 1); }

constexpr u64 closest_power_of_two(u64 x)
{
    u64 n = x;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;
    return n + 1;
}

// -----------------------------------------------------------------------------------
// MARK: Memory block impl
// -----------------------------------------------------------------------------------

struct MemoryBlock
{
    constexpr static u64 header_size() { return sizeof(MemoryBlock); }

    static MemoryBlock* init(u64 commit_size, u64 block_size)
    {
        Assert(commit_size > header_size());
        Assert(commit_size < block_size);

        MemoryBlock* mem_block = nullptr;
        auto         err       = AllocationError::None;
        mem_block              = cast(MemoryBlock*) os_reserve(nullptr, block_size, &err);
        Assert(err == AllocationError::None && mem_block != nullptr);

        err = os_commit(mem_block, commit_size);
        Assert(err == AllocationError::None && mem_block != nullptr);

        mem_block->m_used        = header_size();
        mem_block->m_commit_size = commit_size;
        mem_block->m_reserved    = block_size;
        mem_block->m_committed   = commit_size;
        return mem_block;
    }

    static void deinit(MemoryBlock* mem_block)
    {
        if (mem_block != nullptr)
        {
            auto err = AllocationError::None;
            err      = os_release(mem_block, mem_block->m_reserved);
            Assert(err == AllocationError::None);
        }
    }

    MemoryBlock(const MemoryBlock&)            = delete;
    MemoryBlock& operator=(const MemoryBlock&) = delete;

    MemoryBlock(MemoryBlock&& other) noexcept
        : m_prev(std::exchange(other.m_prev, nullptr))
        , m_next(std::exchange(other.m_next, nullptr))
        , m_used(std::exchange(other.m_used, header_size()))
        , m_committed(std::exchange(other.m_committed, header_size()))
        , m_reserved(std::exchange(other.m_reserved, header_size()))
        , m_commit_size(other.m_commit_size)
    {
    }

    MemoryBlock& operator=(MemoryBlock&& other) noexcept
    {
        m_prev        = std::exchange(other.m_prev, nullptr);
        m_next        = std::exchange(other.m_next, nullptr);
        m_used        = std::exchange(other.m_used, header_size());
        m_committed   = std::exchange(other.m_committed, header_size());
        m_reserved    = std::exchange(other.m_reserved, header_size());
        m_commit_size = other.m_commit_size;
        return *this;
    }

    void reset_by(u64 used)
    {
        Assert(m_used >= header_size());
        if (m_used - header_size() <= used)
        {
            m_used = header_size();
        }
        else
        {
            m_used -= used;
        }
    }

    void reset_to(u64 new_used_pos) { m_used = std::max(new_used_pos, header_size()); }

    bool can_fit(u64 size, u64 alignment)
    {
        u64 new_used = align_formula(m_used, alignment);
        return new_used + size <= m_reserved;
    }

    // Caller should ensure the requested allocation will fit the memory block,
    // by calling `can_fit` method
    rawptr alloc(u64 size, u64 alignment)
    {
        Assert(is_power_of_two(alignment));
        Assert(m_used >= header_size());
        Assert(can_fit(size, alignment));
        u64 aligned_used = align_formula(m_used, alignment);
        u64 new_used     = aligned_used + size;
        if (new_used >= m_committed)
        {
            auto new_committed = align_formula(m_committed + new_used, m_commit_size);
            auto err           = os_commit(this, new_committed);
            Assert(err == AllocationError::None);
            m_committed = new_committed;
        }
        m_used     = new_used;
        rawptr res = cast(u8*) this + aligned_used;
        std::memset(res, 0, size);
        return res;
    }

    MemoryBlock* m_prev        = nullptr;
    MemoryBlock* m_next        = nullptr;
    u64          m_used        = 0;
    u64          m_committed   = 0;
    u64          m_reserved    = 0;
    u64          m_commit_size = 0;
};

// -----------------------------------------------------------------------------------
// MARK: Temp Arena Guard
// -----------------------------------------------------------------------------------

struct TempArenaGuard
{
    explicit TempArenaGuard(struct VirtualArena* arena);
    ~TempArenaGuard();

    TempArenaGuard(TempArenaGuard&& other) noexcept;
    TempArenaGuard& operator=(TempArenaGuard&& other) noexcept;
    TempArenaGuard(const TempArenaGuard& other)            = delete;
    TempArenaGuard& operator=(const TempArenaGuard& other) = delete;

  private:
    struct VirtualArena* m_arena = nullptr;
    u64                  m_pos   = 0;
};

// -----------------------------------------------------------------------------------
// MARK: Virtual Arena
// -----------------------------------------------------------------------------------

struct VirtualArena : Allocator
{
    VirtualArena(u64 block_size = Megabytes(8), u64 commit_size = Megabytes(1), u64 alignment = DEFAULT_ALIGNMENT)
        : m_alignment(alignment)
    {
        Assert(block_size > sizeof(MemoryBlock));
        Assert(commit_size > 0);
        Assert(alignment > 0);
        Assert(is_power_of_two(block_size));
        Assert(is_power_of_two(commit_size));
        Assert(is_power_of_two(alignment));
        Assert(commit_size > alignment);
        Assert(block_size > commit_size);

        u64 page_size = os_page_size();
        Assert(is_power_of_two(page_size));
        m_commit_size = align_formula(commit_size, page_size);
        m_block_size  = align_formula(block_size, m_commit_size);
        m_block       = MemoryBlock::init(m_commit_size, m_block_size);
        m_reserved    = m_block->m_reserved;
        m_used += m_block->m_used;
    }

    VirtualArena(VirtualArena&& other) noexcept
        : m_block(std::exchange(other.m_block, nullptr))
        , m_reserved(std::exchange(other.m_reserved, 0))
        , m_used(std::exchange(other.m_used, 0))
        , m_alignment(other.m_alignment)
        , m_commit_size(other.m_commit_size)
        , m_block_size(other.m_block_size)
    {
    }

    VirtualArena& operator=(VirtualArena&& other) noexcept
    {
        m_block       = std::exchange(other.m_block, nullptr);
        m_reserved    = std::exchange(other.m_reserved, 0);
        m_used        = std::exchange(other.m_used, 0);
        m_alignment   = other.m_alignment;
        m_commit_size = other.m_commit_size;
        m_block_size  = other.m_block_size;
        return *this;
    }

    VirtualArena(const VirtualArena& other)            = delete;
    VirtualArena& operator=(const VirtualArena& other) = delete;

    ~VirtualArena()
    {
        if (m_block != nullptr)
        {
            auto start_block = m_block;
            // deallocate memory blocks forwards
            for (auto mem_block = start_block->m_next; mem_block != nullptr;)
            {
                auto next_block = mem_block->m_next;
                MemoryBlock::deinit(mem_block);
                mem_block = next_block;
            }
            // ... and backwards
            for (auto mem_block = start_block; mem_block != nullptr;)
            {
                auto prev_block = mem_block->m_prev;
                MemoryBlock::deinit(mem_block);
                mem_block = prev_block;
            }
            m_used     = 0;
            m_reserved = 0;
        }
    }

    rawptr allocate(u64 size) { return alloc(size, m_alignment); }

    rawptr allocate(u64 size, u64 alignment)
    {
        Assert(size < m_block_size);
        if (m_block == nullptr || !m_block->can_fit(size, alignment))
        {
            MemoryBlock* mem_block = MemoryBlock::init(m_commit_size, m_block_size);
            Assert(mem_block->can_fit(size, alignment));
            mem_block->m_prev = m_block;
            m_block->m_next   = mem_block;
            m_block           = mem_block;
            m_reserved += m_block->m_reserved;
            m_used += m_block->m_used;
        }
        u64    prev_used_in_block = m_block->m_used;
        rawptr res                = m_block->alloc(size, alignment);
        Assert(res != nullptr);
        m_used += m_block->m_used - prev_used_in_block;
        return res;
    }

    void reset_to(u64 pos)
    {
        if (m_block != nullptr && pos < m_used)
        {
            while (m_used > pos)
            {
                u64 total_diff         = m_used - pos;
                u64 prev_used_in_block = m_block->m_used;
                m_block->reset_by(total_diff);
                m_used -= prev_used_in_block - m_block->m_used;
                if (m_used > pos && m_block->m_prev != nullptr)
                {
                    auto prev_block = m_block;
                    m_block         = m_block->m_prev;
                    if (m_block->m_next != prev_block)
                    {
                        m_block->m_next = prev_block;
                    }
                }
                else
                {
                    break;
                }
            }
        }
    }

    rawptr alloc(u64 size, u64 alignment) override { return allocate(size, alignment); }
    void   free(rawptr ptr, u64 size) override
    {
        (void)ptr;
        (void)size;
    }
    Allocator* allocator() { return this; }

    TempArenaGuard temp_arena_guard() { return TempArenaGuard(this); }

    u64 used() const { return m_used; }

  private:
    MemoryBlock* m_block;
    u64          m_reserved;
    u64          m_used;
    u64          m_alignment;
    u64          m_commit_size;
    u64          m_block_size;
};

inline TempArenaGuard::TempArenaGuard(struct VirtualArena* arena)
    : m_arena(arena)
    , m_pos(arena->used())
{
}

inline TempArenaGuard::~TempArenaGuard()
{
    if (m_arena != nullptr)
    {
        m_arena->reset_to(m_pos);
        m_arena = nullptr;
    }
}

inline TempArenaGuard::TempArenaGuard(TempArenaGuard&& other) noexcept
    : m_arena(std::exchange(other.m_arena, nullptr))
    , m_pos(std::exchange(other.m_pos, 0))
{
}

inline TempArenaGuard& TempArenaGuard::operator=(TempArenaGuard&& other) noexcept
{
    m_arena = std::exchange(other.m_arena, nullptr);
    m_pos   = std::exchange(other.m_pos, 0);
    return *this;
}

#endif