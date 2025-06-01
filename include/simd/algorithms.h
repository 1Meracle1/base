#ifndef SIMD_ALGORITHMS_H
#define SIMD_ALGORITHMS_H

#include "vector.h"

namespace simd
{
template <typename T> inline i64 first_index_of(const T* ptr, u64 length, T needle)
{
    if (length == 0)
        return -1;
    auto vectorized = []<typename instruction_set_t>(const T* ptr, u64 length, T needle) -> i64
    {
        using Vector           = Vector<u8, instruction_set_t>;
        constexpr u64 v_length = Vector::length;
        auto          v_needle = Vector::splat(needle);
        u64           pos      = 0;
        while (pos + v_length <= length)
        {
            auto v_haystack = Vector::load_unaligned(ptr + pos);
            auto matches    = v_haystack == v_needle;
            auto mask       = matches.movemask();
            if (mask != 0)
            {
                auto match_offset_in_v = bit_scan_first_set_bit(mask);
                return pos + match_offset_in_v;
            }
            pos += v_length;
        }
        for (; pos < length; ++pos)
        {
            if (ptr[pos] == needle)
                return pos;
        }
        return -1;
    };
    auto scalar = [](const T* ptr, u64 length, T needle) -> i64
    {
        for (u64 i = 0; i < length; ++i)
        {
            if (ptr[i] == needle)
                return i;
        }
        return -1;
    };
    return dispatch_instruction_set<T>(length, vectorized, scalar, ptr, length, needle);
}

template <typename T> inline bool equal(const T* haystack_ptr, u64 haystack_len, const T* needle_ptr, u64 needle_len)
{
    if (haystack_len == 0 || needle_len == 0 || haystack_len != needle_len)
        return false;
    auto vectorized = []<typename instruction_set_t>(const T* haystack_ptr, u64 haystack_len, const T* needle_ptr,
                                                     u64 needle_len) -> bool
    {
        using Vector           = Vector<u8, instruction_set_t>;
        constexpr u64 v_length = Vector::length;
        u64           pos      = 0;
        while (pos + v_length <= haystack_len)
        {
            auto v_haystack = Vector::load_unaligned(haystack_ptr + pos);
            auto v_needle   = Vector::load_unaligned(needle_ptr + pos);
            auto matches    = v_haystack == v_needle;
            auto mask       = matches.movemask();
            if (mask != 0xFFFF)
            {
                return false;
            }
            pos += v_length;
        }
        for (; pos < haystack_len; ++pos)
            if (haystack_ptr[pos] != needle_ptr[pos])
                return false;
        return true;
    };
    auto scalar = [](const T* haystack_ptr, u64 haystack_len, const T* needle_ptr, u64 needle_len) -> bool
    { return std::memcmp(haystack_ptr, needle_ptr, haystack_len * sizeof(T)) == 0; };
    return dispatch_instruction_set<T>(haystack_len, vectorized, scalar, haystack_ptr, haystack_len, needle_ptr,
                                       needle_len);
}

// template <typename T, typename InstructionSet>
// inline i64 first_index_any_of(const T* haystack_ptr, u64 haystack_length, const T* needle_ptr, u64 needle_length)
// {
//     using Vector           = Vector<u8, InstructionSet>;
//     constexpr u64 v_length = Vector::length;
//     auto          v_needle = Vector::splat(needle);
//     u64           pos      = 0;
//     while (pos + v_length <= length)
//     {
//         auto v_haystack = Vector::load_unaligned(ptr + pos);
//         auto matches    = v_haystack == v_needle;
//         auto mask       = matches.mask();
//         if (mask != 0)
//         {
//             auto match_offset_in_v = bit_scan_first_set_bit(mask);
//             return pos + match_offset_in_v;
//         }
//         pos += v_length;
//     }
//     for (; pos < length; ++pos)
//     {
//         if (ptr[pos] == needle)
//             return pos;
//     }
//     return -1;
// }

// template <typename T>
// inline i64 first_index_any_of(const T* haystack_ptr, u64 haystack_length, const T* needle_ptr, u64 needle_length)
// {
//     if (haystack_length == 0 || needle_length == 0)
//         return -1;
//     if constexpr (RegisterTraitRequirement<T, AVX2InstructionSet>)
//     {
//         if (supported_features().avx2 && InstructionSetLength<T, AVX2InstructionSet>::fits(length))
//         {
//             return first_index_any_of<T, AVX2InstructionSet>(haystack_ptr, haystack_length, needle_ptr,
//             needle_length);
//         }
//     }
//     if constexpr (RegisterTraitRequirement<T, AVXInstructionSet>)
//     {
//         if (supported_features().avx && InstructionSetLength<T, AVXInstructionSet>::fits(length))
//         {
//             return first_index_of<T, AVXInstructionSet>(ptr, length, needle);
//         }
//     }
//     if constexpr (RegisterTraitRequirement<T, SSE2InstructionSet>)
//     {
//         if (supported_features().sse2 && InstructionSetLength<T, SSE2InstructionSet>::fits(length))
//         {
//             return first_index_of<T, SSE2InstructionSet>(ptr, length, needle);
//         }
//     }
//     for (u64 i = 0; i < length; ++i)
//     {
//         if (ptr[i] == needle)
//             return i;
//     }
//     return -1;
// }
} // namespace simd

#endif