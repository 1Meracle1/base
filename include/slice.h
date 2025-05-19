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

  private:
    pointer   m_data = nullptr;
    size_type m_len  = 0;

  public:
    constexpr Slice() noexcept
        : m_data(nullptr)
        , m_len(0)
    {
    }

    constexpr Slice(pointer data, size_type len) noexcept
        : m_data(data)
        , m_len(len)
    {
    }

    // implicit constructor for literal arrays like `int arr[] = {1, 2, 3};`,
    // and for character strings `const char* str = "hello";` - FIXME: should actually be a different constructor?
    template <size_type N>
    constexpr Slice(value_type (&arr)[N]) noexcept
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
    constexpr Slice slice_from(size_type start) const noexcept { return slice(start, len()); }

    // [..end)
    constexpr Slice slice_to(size_type end) const noexcept { return slice(0, end); }

    // [end-start..)
    constexpr Slice slice_from_back(size_type length) const noexcept { return slice(len() - length, len()); }

    // [start, end)
    constexpr Slice slice(size_type start, size_type end) const noexcept
    {
        Assert(start < len() && end <= len());
        size_type   len  = end - start;
        value_type* data = len == 0 ? nullptr : m_data + start;
        return Slice(data, len);
    }

    constexpr pointer       data() noexcept { return m_data; }
    constexpr const_pointer data() const noexcept { return m_data; }
    constexpr size_type     len() const noexcept { return m_len; }
    constexpr bool          empty() const noexcept { return len() == 0; }

    reference       operator[](size_type i) { return m_data[i]; }
    const_reference operator[](size_type i) const { return m_data[i]; }

    iterator       begin() { return m_data; }
    iterator       end() { return m_data + len(); }
    const_iterator begin() const { return m_data; }
    const_iterator end() const { return m_data + len(); }
    const_iterator cbegin() const { return m_data; }
    const_iterator cend() const { return m_data + len(); }

    std::reverse_iterator<iterator>       rbegin() { return std::reverse_iterator<iterator>(end()); }
    std::reverse_iterator<iterator>       rend() { return std::reverse_iterator<iterator>(begin()); }
    std::reverse_iterator<const_iterator> rbegin() const { return std::reverse_iterator<const_iterator>(end()); }
    std::reverse_iterator<const_iterator> rend() const { return std::reverse_iterator<const_iterator>(begin()); }
    std::reverse_iterator<const_iterator> crbegin() const { return std::reverse_iterator<const_iterator>(cend()); }
    std::reverse_iterator<const_iterator> crend() const { return std::reverse_iterator<const_iterator>(cbegin()); }

    reference       front() { m_data[0]; }
    const_reference front() const { m_data[0]; }
    reference       back() { m_data[len() - 1]; }
    const_reference back() const { m_data[len() - 1]; }

    constexpr void swap(size_type i, size_type j) { std::swap(m_data[i], m_data[j]); }

    constexpr void reverse()
    {
        size_type half = len() / 2;
        for (size_type i = 0; i < half; i++)
            swap(i, len() - i - 1);
    }

    constexpr i64 linear_search(const_reference v) const
    {
        for (size_type i = 0; i < len(); i++)
            if (m_data[i] == v)
                return i;
        return -1;
    }

    using PredicateType = std::function<bool(const_reference, const_reference)>;

    constexpr i64 linear_search(const_reference v, PredicateType&& predicate) const
    {
        for (size_type i = 0; i < len(); i++)
            if (predicate(m_data[i], v))
                return i;
        return -1;
    }

    constexpr bool contains_value(const_reference v) const
        requires(std::equality_comparable<value_type>)
    {
        return linear_search(v);
    }

    constexpr bool contains_value(const_reference v, PredicateType&& predicate) const
        requires(std::equality_comparable<value_type>)
    {
        return linear_search(v, std::forward<PredicateType>(predicate));
    }

    constexpr void zero()
    {
        if (len() > 0)
            std::memset(m_data, 0, len());
    }

    constexpr bool bytes_equal(const_reference other) const
    {
        if (len() != other.len())
            return false;
        return std::memcmp(m_data, other.m_data, len() * sizeof(value_type)) == 0;
    }

    constexpr bool equal(const_reference other) const
    {
        if (len() != other.len())
            return false;
        for (size_type i = 0; i < len(); i++)
            if (m_data[i] != other[i])
                return false;
        return true;
    }

    constexpr bool equal(const_reference other, PredicateType&& predicate) const
    {
        if (len() != other.m_len)
            return false;
        for (size_type i = 0; i < len(); i++)
            if (!predicate(m_data[i], other[i]))
                return false;
        return true;
    }

    constexpr bool has_prefix(const_reference needle) const
    {
        if (len() < needle.len())
            return false;
        return equal(slice_to(needle.len()));
    }

    constexpr bool has_prefix(const_reference needle, PredicateType&& predicate) const
    {
        if (len() < needle.len())
            return false;
        return equal(slice_to(needle.len()), std::forward<PredicateType>(predicate));
    }

    constexpr bool has_suffix(const_reference needle) const
    {
        if (len() < needle.len())
            return false;
        return equal(slice_from_back(needle.len()));
    }

    constexpr bool has_suffix(const_reference needle, PredicateType&& predicate) const
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
            if (m_data[j] != m_data[j - 1])
            {
                if (i != j)
                    m_data[i] = m_data[j];
                i += 1;
            }
        }
        return slice_to(i);
    }

    constexpr Slice unique(PredicateType&& predicate)
    {
        if (len() < 2)
            return *this;
        size_type i = 1;
        for (size_type j = 1; j < len(); j++)
        {
            if (!predicate(m_data[j], m_data[j - 1]))
            {
                if (i != j)
                    m_data[i] = m_data[j];
                i += 1;
            }
        }
        return slice_to(i);
    }
};

#endif