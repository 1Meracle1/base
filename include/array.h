#ifndef ARRAY_H
#define ARRAY_H

#include "memory.h"
#include "slice.h"
#include "types.h"
#include <type_traits>
#include <utility>

template <typename F>
concept GrowthFormulaConcept = requires(F f, std::size_t current_capacity) {
    { f(current_capacity) } -> std::same_as<std::size_t>;
} && std::is_default_constructible_v<F>;

struct GrowthFormulaDefault
{
    std::size_t operator()(std::size_t current_capacity)
    {
        // 0 - 8
        // 8 - 13
        // 13 - 21
        // 21 - 33
        // 33 - 51
        // 51 - 78
        // 78 - 118
        // 118 - 178
        // 178 - 268
        // 268 - 403
        // 403 - 606
        // 606 - 910
        // 910 - 1366
        // 1366 - 2050
        // 2050 - 3076
        // 3076 - 4615
        // 4615 - 6924
        // 6924 - 10387
        // 10387 - 15582
        // 15582 - 23374
        // 23374 - 35062
        // 35062 - 52594
        // 52594 - 78892
        // 78892 - 118339
        // 118339 - 177510
        // 177510 - 266266
        // 266266 - 399400
        // 399400 - 599101
        // 599101 - 898653
        // 898653 - 1347981
        return std::max<std::size_t>((current_capacity + 1) * 3 >> 1, 8);
    }
};

template <typename ValueType, GrowthFormulaConcept GrowthFormula = GrowthFormulaDefault> struct Array
{
    using value_type      = ValueType;
    using pointer         = value_type*;
    using const_pointer   = const value_type*;
    using reference       = value_type&;
    using const_reference = std::conditional_t<TrivialSmall<value_type>, value_type, const value_type&>;

    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;

    using iterator_category = std::contiguous_iterator_tag;
    using iterator          = pointer;
    using const_iterator    = const_pointer;

    Allocator*  m_allocator{nullptr};
    value_type* m_ptr{nullptr};
    size_type   m_len{0};
    size_type   m_capacity{0};

    Array() = default;

    constexpr explicit Array(Slice<value_type> slice) noexcept
        : m_ptr(slice.m_ptr)
        , m_len(slice.m_len)
    {
    }

    explicit Array(Allocator* allocator, Slice<value_type> slice) noexcept
        : m_allocator(allocator)
    {
        append_many(slice);
    }

    constexpr explicit Array(Allocator* allocator) noexcept
        : m_allocator(allocator)
    {
    }

    Array(Allocator* allocator, size_type capacity) noexcept
        : m_allocator(allocator)
        , m_capacity(capacity)
    {
        if (m_capacity > 0)
        {
            m_ptr = m_allocator->alloc<value_type>(m_capacity);
        }
    }

    Array(Allocator* allocator, size_type capacity, size_type length) noexcept
        : m_allocator(allocator)
        , m_capacity(std::max(capacity, length))
    {
        if (m_capacity > 0)
        {
            m_ptr = m_allocator->alloc<value_type>(m_capacity);
            m_len = length;
        }
    }

    ~Array() noexcept { free_allocated_memory(); }

    constexpr Array(Array&& other) noexcept
        : m_allocator(std::exchange(other.m_allocator, nullptr))
        , m_ptr(std::exchange(other.m_ptr, nullptr))
        , m_len(std::exchange(other.m_len, 0))
        , m_capacity(std::exchange(other.m_capacity, 0))
    {
    }

    constexpr Array& operator=(Array&& other) noexcept
    {
        if (this != &other)
        {
            free_allocated_memory();
            m_allocator = std::exchange(other.m_allocator, nullptr);
            m_ptr       = std::exchange(other.m_ptr, nullptr);
            m_len       = std::exchange(other.m_len, 0);
            m_capacity  = std::exchange(other.m_capacity, 0);
        }
        return *this;
    }

    Array(const Array&)                  = delete;
    Array& operator=(const Array& other) = delete;

    // clang-format off
    reference       operator[](size_type i)       { return m_ptr[i]; }
    const_reference operator[](size_type i) const { return m_ptr[i]; }

    constexpr size_type len()       const { return m_len; }
    constexpr bool      empty()     const { return m_len == 0; }
    constexpr bool      not_empty() const { return m_len > 0; }

    reference       front()       { Assert(not_empty()); return m_ptr[0]; }
    const_reference front() const { Assert(not_empty()); return m_ptr[0]; }
    reference       back()        { Assert(not_empty()); return m_ptr[len() - 1]; }
    const_reference back()  const { Assert(not_empty()); return m_ptr[len() - 1]; }

    constexpr pointer       data()       { return m_ptr; }
    constexpr const_pointer data() const { return m_ptr; }

    iterator       begin()        { return m_ptr; }
    iterator       end()          { return m_ptr + m_len; }
    const_iterator begin()  const { return m_ptr; }
    const_iterator end()    const { return m_ptr + m_len; }
    const_iterator cbegin() const { return m_ptr; }
    const_iterator cend()   const { return m_ptr + m_len; }
    // clang-format on

    constexpr Slice<value_type> view() const { return Slice(m_ptr, m_len); }

    void check_reserve(size_type added_elements_length = 1)
    {
        if (m_capacity + added_elements_length >= m_len)
        {
            auto new_capacity = GrowthFormula{}(m_capacity + added_elements_length);
            Assert(new_capacity > m_capacity);
            m_allocator->realloc(m_ptr, m_capacity, new_capacity);
            m_capacity = new_capacity;
        }
    }

    void append(const_reference value)
    {
        check_reserve();
        m_ptr[m_len] = value;
        m_len++;
    }

    void append(value_type&& value)
        requires(!TrivialSmall<value_type>)
    {
        check_reserve();
        m_ptr[m_len] = std::forward<value_type>(value);
        m_len++;
    }

    // meant to be used in scenarios when non-trivial objects are used
    // in order to avoid extra copies
    // ` auto ptr = array.alloc_element();
    // ` ptr->x = 2;
    [[nodiscard]] pointer append()
    {
        check_reserve();
        return m_ptr[m_len];
    }

    // void append_many(Slice<value_type>&& elements)
    //     requires(!std::is_trivially_copyable_v<value_type> && std::is_move_assignable_v<value_type>)
    // {
    //     if (!elements.empty())
    //     {
    //         check_reserve(elements.len());
    //         for (size_type i = 0; i < elements.len(); i++)
    //         {
    //             m_ptr[m_len] = std::move(elements[i]);
    //             m_len++;
    //         }
    //     }
    // }

    void append_many(Slice<value_type> elements)
        requires(!TrivialSmall<value_type> && std::is_copy_assignable_v<value_type>)
    {
        if (!elements.empty())
        {
            check_reserve(elements.len());
            for (size_type i = 0; i < elements.len(); i++)
            {
                m_ptr[m_len] = elements[i];
                m_len++;
            }
        }
    }

    void append_many(Slice<value_type> elements)
        requires(TrivialSmall<value_type>)
    {
        if (!elements.empty())
        {
            check_reserve(elements.len());
            std::memcpy(&m_ptr[m_len], elements.data(), elements.len());
            m_len += elements.len();
        }
    }

    void remove_unordered(size_type i)
        requires(TrivialSmall<value_type>)
    {
        Assert(i < m_len);
        if (m_len > 1 && i < m_len - 1)
        {
            m_ptr[i] = m_ptr[m_len - 1];
        }
        --m_len;
    }

    void remove_unordered(size_type i)
        requires(!TrivialSmall<value_type> && std::is_move_assignable_v<value_type>)
    {
        Assert(i < m_len);
        if (m_len > 1 && i < m_len - 1)
        {
            m_ptr[i] = std::move(m_ptr[m_len - 1]);
        }
        --m_len;
    }

    void remove_unordered(size_type i)
        requires(!TrivialSmall<value_type> && std::is_copy_assignable_v<value_type> &&
                 !std::is_move_assignable_v<value_type>)
    {
        Assert(i < m_len);
        if (m_len > 1 && i < m_len - 1)
        {
            m_ptr[i] = m_ptr[m_len - 1];
        }
        --m_len;
    }

    void reset_length() { m_len = 0; }

    void free_allocated_memory()
    {
        if (m_allocator && m_capacity > 0)
        {
            m_allocator->free(m_ptr, m_capacity);
        }
        m_capacity  = 0;
        m_len       = 0;
        m_ptr       = nullptr;
        m_allocator = nullptr;
    }

    // [[nodiscard]] Array<Slice<value_type>> split(Allocator* allocator, const_reference sep) const
    // {
    //     size_type number_parts = 1;
    //     for (size_type pos = 0; pos < len();)
    //     {
    //         Slice<value_type> rem{m_ptr + pos, len() - pos};
    //         i64               index = rem.linear_search(sep);
    //         if (index == -1)
    //         {
    //             break;
    //         }
    //         pos += index + 1;
    //         number_parts += 1;
    //     }

    //     Array<Slice<value_type>> res{allocator, number_parts};
    //     for (u64 pos = 0; pos < len();)
    //     {
    //         Slice<value_type> rem{m_ptr + pos, len() - pos};
    //         i64               index = rem.linear_search(sep);
    //         if (index == -1)
    //         {
    //             res.append(rem.slice_from(pos));
    //             break;
    //         }
    //         else
    //         {
    //             res.append(rem.slice(pos, index + 1));
    //             pos += index + 1;
    //         }
    //     }
    //     return res;
    // }

    // [[nodiscard]] Array<Slice<value_type>> split(Allocator* allocator, Slice<value_type> sep) const
    // {
    //     Array<Slice<value_type>> res{allocator, 1};
    //     if (len() <= sep.len())
    //     {
    //         res.append(*this);
    //         return res;
    //     }
    //     for (u64 pos = 0; pos < len() - sep.len(); pos++)
    //     {
    //         Slice<value_type> rem{m_ptr + pos, len() - pos};
    //         if (rem.starts_with(sep))
    //         {
    //             res.append(rem.slice_to(sep.len()));
    //             pos += sep.len();
    //         }
    //     }
    //     return res;
    // }
};

#endif