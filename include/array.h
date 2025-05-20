#ifndef ARRAY_H
#define ARRAY_H

#include "slice.h"
#include "memory.h"
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
    using pointer         = ValueType*;
    using const_pointer   = const ValueType*;
    using reference       = ValueType&;
    using const_reference = const ValueType&;

    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;

    Allocator*       m_allocator{nullptr};
    Slice<ValueType> m_data{};
    size_type        m_capacity{0};

    explicit Array(Allocator* allocator) noexcept
        : m_allocator(allocator)
    {
    }

    Array(Allocator* allocator, size_type capacity) noexcept
        : m_allocator(allocator)
        , m_capacity(capacity)
    {
        if (m_capacity > 0)
        {
            m_data.m_ptr = m_allocator->alloc<value_type>(m_capacity);
        }
    }

    Array(Allocator* allocator, size_type capacity, size_type length) noexcept
        : m_allocator(allocator)
        , m_capacity(std::max(capacity, length))
    {
        if (m_capacity > 0)
        {
            m_data.m_ptr = m_allocator->alloc<value_type>(m_capacity);
            m_data.m_len = length;
        }
    }

    ~Array() noexcept
    {
        if (m_allocator)
        {
            if (m_capacity > 0)
            {
                m_allocator->free(m_data.m_ptr, m_capacity);
            }
            m_capacity   = 0;
            m_data.m_len = 0;
            m_data.m_ptr = nullptr;
            m_allocator  = nullptr;
        }
    }

    Array(Array&& other) noexcept
        : m_allocator(std::exchange(other.m_allocator, nullptr))
        , m_data(std::exchange(other.m_data, Slice<ValueType>()))
        , m_capacity(std::exchange(other.m_capacity, 0))
    {
    }

    Array& operator=(Array&& other) noexcept
    {
        m_allocator = std::exchange(other.m_allocator, nullptr);
        m_data      = std::exchange(other.m_data, Slice<ValueType>());
        m_capacity  = std::exchange(other.m_capacity, 0);
        return *this;
    }

    Array(const Array&)                  = delete;
    Array& operator=(const Array& other) = delete;

    Slice<value_type>&       data() { return m_data; }
    const Slice<value_type>& data() const { return m_data; }

    void check_reserve(size_type added_elements_length = 1)
    {
        if (m_capacity + added_elements_length >= m_data.len())
        {
            auto new_capacity = GrowthFormula{}(m_capacity + added_elements_length);
            Assert(new_capacity > m_capacity);
            m_allocator->realloc(m_data.m_ptr, m_capacity, new_capacity);
            m_capacity = new_capacity;
        }
    }

    void append(const_reference value)
    {
        check_reserve();
        m_data[m_data.len()] = value;
        m_data.m_len++;
    }

    void append(value_type&& value)
    {
        check_reserve();
        m_data[m_data.len()] = std::forward<value_type>(value);
        m_data.m_len++;
    }

    void append_many(pointer ptr, size_type length)
    {
        Assert(ptr != nullptr && length > 0);
        check_reserve(length);
        for (size_type i = 0; i < length; i++)
        {
            m_data[m_data.len()] = ptr[i];
            m_data.m_len++;
        }
    }

    void clear() { m_data = Slice<ValueType>(); }
};

#endif