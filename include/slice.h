#ifndef SLICE_H
#define SLICE_H

#include "defines.h"
#include <concepts>
#include <cstddef>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <type_traits>
#include "assert.h"

template <typename ValueType> struct Slice
{
    using iterator_category = std::random_access_iterator_tag;
    using value_type        = std::remove_cv_t<ValueType>;
    using difference_type   = std::ptrdiff_t; // does it make any difference to use u64 instead?
    using pointer           = ValueType*;
    using reference         = ValueType&;

  private:
    pointer m_data = nullptr;
    u64     m_len  = 0;

  public:
    constexpr Slice(ValueType* data, u64 len) noexcept
        : m_data(data)
        , m_len(len)
    {
    }

    // implicit constructor for literal arrays like `int arr[] = {1, 2, 3};`,
    // and for character strings `const char* str = "hello";` - FIXME: should actually be a different constructor?
    template <u64 N>
    constexpr Slice(ValueType (&arr)[N]) noexcept
        : m_data(arr)
        , m_len(N)
    {
    }

    // implicit constructor for intializer lists;
    template <typename Elt, typename = std::enable_if_t<!std::is_array_v<Elt>>> // std::is_same_v<T, const Elt> &&
    // template <typename Elt>
    constexpr Slice(std::initializer_list<Elt> init_list) noexcept
        // requires(!std::is_array_v<Elt>)
        : m_data(init_list.begin())
        , m_len(init_list.size())
    {
    }

    // [start..)
    constexpr Slice slice_from(u64 start) const noexcept { return slice(start, m_len); }

    // [..end)
    constexpr Slice slice_to(u64 end) const noexcept { return slice(0, end); }

    // [start, end)
    constexpr Slice slice(u64 start, u64 end) const noexcept
    {
        Assert(start < m_len && end <= m_len);
        u64        len  = end - start;
        ValueType* data = len == 0 ? nullptr : m_data + start;
        return Slice(data, len);
    }

    constexpr pointer data() noexcept { return m_data; }
    constexpr pointer data() const noexcept { return m_data; }
    constexpr u64     len() const noexcept { return m_len; }
    constexpr bool    is_empty() const noexcept { return m_len == 0; }

    reference operator[](u64 i) const { return m_data[i]; }

    constexpr void swap(u64 i, u64 j) { std::swap(m_data[i], m_data[j]); }

    constexpr void reverse()
    {
        u64 half = m_len / 2;
        for (u64 i = 0; i < half; i++)
            swap(i, m_len - i - 1);
    }

    constexpr i64 linear_search(const ValueType& v) const
    {
        for (u64 i = 0; i < m_len; i++)
            if (m_data[i] == v)
                return i;
        return -1;
    }

    using PredicateType = std::function<bool(const ValueType&, const ValueType&)>;

    constexpr i64 linear_search(const ValueType& v, PredicateType&& predicate) const
    {
        for (u64 i = 0; i < m_len; i++)
            if (predicate(m_data[i], v))
                return i;
        return -1;
    }

    constexpr bool contains(const ValueType& v) const
        requires(std::equality_comparable<ValueType>)
    {
        return linear_search(v);
    }

    constexpr bool contains(const ValueType& v, PredicateType&& predicate) const
        requires(std::equality_comparable<ValueType>)
    {
        return linear_search(v, std::forward<PredicateType>(predicate));
    }

    constexpr void zero()
    {
        if (m_len > 0)
            std::memset(m_data, 0, m_len);
    }
};

#endif