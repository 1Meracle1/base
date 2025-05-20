#ifndef SLICE_H
#define SLICE_H

#include "defines.h"
#include "memory.h"
#include <algorithm>
#include <compare>
#include <concepts>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <type_traits>

template <typename F, typename ValueType>
concept SliceElementEqualityPredicate = requires(F f, const ValueType& lhs, const ValueType& rhs) {
    { f(lhs, rhs) } -> std::same_as<bool>;
};

template <typename F, typename ValueType>
concept SliceElementWeakOrderingComparePredicate = requires(F f, const ValueType& lhs, const ValueType& rhs) {
    { f(lhs, rhs) } -> std::same_as<std::weak_ordering>;
};

template <typename ValueType> struct Slice
{
    using value_type      = ValueType;
    using pointer         = ValueType*;
    using const_pointer   = const ValueType*;
    using reference       = ValueType&;
    using const_reference = const ValueType&;

    using iterator_category = std::contiguous_iterator_tag;
    using iterator          = pointer;
    using const_iterator    = const_pointer;

    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;

    pointer   m_ptr = nullptr;
    size_type m_len = 0;

  public:
    constexpr Slice() noexcept
        : m_ptr(nullptr)
        , m_len(0)
    {
    }

    constexpr Slice(pointer ptr, size_type len) noexcept
        : m_ptr(ptr)
        , m_len(len)
    {
    }

    // implicit constructor for literal arrays like `int arr[] = {1, 2, 3};`,
    template <size_type N>
    constexpr Slice(value_type (&arr)[N]) noexcept
        : m_ptr(arr)
        , m_len(N)
    {
    }

    // implicit constructor for intializer lists;
    // commented out as initializer lists aren't literals? - TBC
    // constexpr Slice(std::initializer_list<value_type> init_list) noexcept
    //     : m_ptr(init_list.begin())
    //     , m_len(init_list.size())
    // {
    // }

    // [start..)
    constexpr Slice slice_from(size_type start) const noexcept { return slice(start, len()); }

    // [..end)
    constexpr Slice slice_to(size_type end) const noexcept { return slice(0, end); }

    // [end-start..)
    constexpr Slice slice_from_back(size_type length) const noexcept { return slice(len() - length, len()); }

    // [start, end)
    constexpr Slice slice(size_type start, size_type end) const noexcept
    {
        Assert(start <= len() && end <= len());
        size_type   len  = end - start;
        value_type* data = len == 0 ? nullptr : m_ptr + start;
        return Slice(data, len);
    }

    constexpr pointer       data() noexcept { return m_ptr; }
    constexpr const_pointer data() const noexcept { return m_ptr; }
    constexpr size_type     len() const noexcept { return m_len; }
    constexpr bool          empty() const noexcept { return len() == 0; }

    reference       operator[](size_type i) { return m_ptr[i]; }
    const_reference operator[](size_type i) const { return m_ptr[i]; }

    iterator       begin() { return m_ptr; }
    iterator       end() { return m_ptr + len(); }
    const_iterator begin() const { return m_ptr; }
    const_iterator end() const { return m_ptr + len(); }
    const_iterator cbegin() const { return m_ptr; }
    const_iterator cend() const { return m_ptr + len(); }

    std::reverse_iterator<iterator>       rbegin() { return std::reverse_iterator<iterator>(end()); }
    std::reverse_iterator<iterator>       rend() { return std::reverse_iterator<iterator>(begin()); }
    std::reverse_iterator<const_iterator> rbegin() const { return std::reverse_iterator<const_iterator>(end()); }
    std::reverse_iterator<const_iterator> rend() const { return std::reverse_iterator<const_iterator>(begin()); }
    std::reverse_iterator<const_iterator> crbegin() const { return std::reverse_iterator<const_iterator>(cend()); }
    std::reverse_iterator<const_iterator> crend() const { return std::reverse_iterator<const_iterator>(cbegin()); }

    reference       front() { return m_ptr[0]; }
    const_reference front() const { return m_ptr[0]; }
    reference       back() { return m_ptr[len() - 1]; }
    const_reference back() const { return m_ptr[len() - 1]; }

    constexpr void swap(size_type i, size_type j) { std::swap(m_ptr[i], m_ptr[j]); }

    constexpr void reverse()
    {
        size_type half = len() / 2;
        for (size_type i = 0; i < half; i++)
            swap(i, len() - i - 1);
    }

    constexpr i64 linear_search(const_reference v) const
    {
        for (size_type i = 0; i < len(); i++)
            if (m_ptr[i] == v)
                return i;
        return -1;
    }

    using PredicateType = std::function<bool(const_reference, const_reference)>;

    constexpr i64 linear_search(const_reference v, PredicateType&& predicate) const
    {
        for (size_type i = 0; i < len(); i++)
            if (predicate(m_ptr[i], v))
                return i;
        return -1;
    }

    constexpr bool contains_value(const_reference v) const
        requires(std::equality_comparable<value_type>)
    {
        return linear_search(v) != -1;
    }

    constexpr bool contains_value(const_reference v, PredicateType&& predicate) const
        requires(std::equality_comparable<value_type>)
    {
        return linear_search(v, std::forward<PredicateType>(predicate)) != -1;
    }

    void zero()
    {
        if (len() > 0)
            std::memset(m_ptr, 0, len());
    }

    constexpr bool bytes_equal(const Slice<value_type>& other) const
    {
        if (len() != other.len())
            return false;
        return std::memcmp(m_ptr, other.m_ptr, len() * sizeof(value_type)) == 0;
    }

    constexpr bool equal(const Slice<value_type>& other) const
    {
        if (len() != other.len())
            return false;
        for (size_type i = 0; i < len(); i++)
            if (m_ptr[i] != other[i])
                return false;
        return true;
    }

    constexpr bool equal(const Slice<value_type>& other, PredicateType&& predicate) const
    {
        if (len() != other.m_len)
            return false;
        for (size_type i = 0; i < len(); i++)
            if (!predicate(m_ptr[i], other[i]))
                return false;
        return true;
    }

    constexpr bool has_prefix(const Slice<value_type>& needle) const
    {
        if (len() < needle.len())
            return false;
        return equal(slice_to(needle.len()));
    }

    constexpr bool has_prefix(const Slice<value_type>& needle, PredicateType&& predicate) const
    {
        if (len() < needle.len())
            return false;
        return equal(slice_to(needle.len()), std::forward<PredicateType>(predicate));
    }

    constexpr bool has_suffix(const Slice<value_type>& needle) const
    {
        if (len() < needle.len())
            return false;
        return equal(slice_from_back(needle.len()));
    }

    constexpr bool has_suffix(const Slice<value_type>& needle, PredicateType&& predicate) const
    {
        if (len() < needle.len())
            return false;
        return equal(slice_from_back(needle.len()), std::forward<PredicateType>(predicate));
    }

    constexpr Slice unique()
    {
        if (len() < 2)
            return *this;
        size_type i = 1;
        for (size_type j = 1; j < len(); j++)
        {
            if (m_ptr[j] != m_ptr[j - 1])
            {
                if (i != j)
                    m_ptr[i] = m_ptr[j];
                i += 1;
            }
        }
        return slice_to(i);
    }

    template <SliceElementEqualityPredicate<value_type> F> constexpr Slice unique(F predicate)
    {
        if (len() < 2)
            return *this;
        size_type i = 1;
        for (size_type j = 1; j < len(); j++)
        {
            if (!predicate(m_ptr[j], m_ptr[j - 1]))
            {
                if (i != j)
                    m_ptr[i] = m_ptr[j];
                i += 1;
            }
        }
        return slice_to(i);
    }

    template <typename F> constexpr ValueType reduce(ValueType initial_value, F&& f) const
    {
        auto res = initial_value;
        for (size_type j = 1; j < len(); j++)
            res += f(res, m_ptr[j]);
        return res;
    }

    template <SliceElementWeakOrderingComparePredicate<value_type> F> constexpr void sort(F&& f)
    {
        std::sort(begin(), end(), std::forward<SliceElementWeakOrderingComparePredicate<F, value_type>>(f));
    }

    Slice clone(Allocator* allocator)
    {
        if (m_len > 0)
        {
            auto ptr = allocator->alloc<value_type>(m_len);
            for (size_type i = 0; i != len(); i++)
                ptr[i] = m_ptr[i];
            return Slice(ptr, m_len);
        }
        return Slice(nullptr, 0);
    }
};

#endif