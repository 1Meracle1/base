#ifndef VECTOR_H
#define VECTOR_H

#include "cpuid.h"
#include "x86_sse2.h"
#include "x86_sse41.h"
#include "x86_avx.h"
#include "x86_avx2.h"
#include "../defines.h"
#include "../assert.h"
#include <array>
#include <cmath>
#include <cstring>
#include <ostream>
#include <type_traits>

/*
Inspiration:
https://www.openmymind.net/SIMD-With-Zig/
https://ziglang.org/documentation/0.14.1/#Vectors
https://github.com/xtensor-stack/xsimd

fn firstIndexOf(haystack: []const u8, needle: u8) ?usize {
  const vector_len = 8;


  // {111, 111, 111, 111, 111, 111, 111, 111}
  const vector_needles: @Vector(vector_len, u8) = @splat(@as(u8, needle));

  // Because we're implementing our own std.simd.firstTrue
  // we can move the following two vectors, indexes and null
  // outside the loop and re-use them.

  // {0, 1, 2, 3, 4, 5, 6, 7}
  const indexes = std.simd.iota(u8, vector_len);

  // {255, 255, 255, 255, 255, 255, 255, 255}
  const nulls: @Vector(vector_len, u8) = @splat(@as(u8, 255));

  var pos: usize = 0;
  var left = haystack.len;
  while (left > 0) {
    if (left < vector_len) {
      // fallback to a normal scan when our input (or what's left of
      // it is smaller than our vector_len)
      return std.mem.indexOfScalarPos(u8, haystack, pos, needle);
    }

    const h: @Vector(vector_len, u8) = haystack[pos..][0..vector_len].*;
    const matches = h == vector_needles;

    if (@reduce(.Or, matches)) {
      // we found a match, we just need to find its index
      const result = @select(u8, matches, indexes, nulls);

      // we have to add pos to this value, since this is merely
      // the index within this vector_len chunk (e.g. 0-7).
      return @reduce(.Min, result) + pos;
    }

    pos += vector_len;
    left -= vector_len;
  }
  return null;
}
*/
namespace simd
{

template <typename T, typename Tag>
concept RegisterTraitRequirement = requires { typename RegisterTrait<T, Tag>::element_type; };

template <typename T, u64 N, typename Tag>
concept RegisterTraitLengthRequirement = (RegisterTraitRequirement<T, Tag> && RegisterTrait<T, Tag>::length == N);

inline u8 bit_scan_first_set_bit(i32 mask)
{
    u8 res;
#if defined(__GNUC__) || defined(__clang__)
    res = cast(u8) __builtin_ctz(mask);
    return res;
#elif defined(_MSC_VER)
    _BitScanForward(&res, cast(u64) mask);
    return res;
#else
    u8 count = 0;
    while ((mask & 1) == 0)
    {
        mask >>= 1;
        count++;
    }
    return count;
#endif
}

// -----------------------------------------------------------------------------------------------
// MARK: Vector
// -----------------------------------------------------------------------------------------------

template <typename T, typename InstructionSet>
    requires(RegisterTraitRequirement<T, InstructionSet> && RegisterTrait<T, InstructionSet>::length > 0)
struct alignas(RegisterTrait<T, InstructionSet>::alignment) Vector
{
    using register_trait = RegisterTrait<T, InstructionSet>;
    using register_type  = typename register_trait::register_type;
    using element_type   = T;

    // static_assert(RegisterTraitRequirement<T, InstructionSet>);
    // static_assert(register_trait::length > 0);

    static constexpr u64 length    = register_trait::length;
    static constexpr u64 alignment = register_trait::alignment;

  private:
    register_type m_data;

  public:
    Vector() noexcept
        : m_data(register_trait::setzero())
    {
    }

    explicit Vector(register_type reg) noexcept
        : m_data(reg)
    {
    }

    explicit Vector(T v) noexcept
        : m_data(register_trait::set1(v))
    {
    }

    template <u64 N>
    explicit Vector(const T (&arr)[N]) noexcept
        requires(N == length)
        : m_data(register_trait::load_unaligned(arr))
    {
    }

    template <u64 N>
    explicit Vector(const std::array<T, N>& arr) noexcept
        requires(N == length)
        : m_data(register_trait::load_unaligned(arr.data()))
    {
    }

    static Vector splat(T scalyr) { return Vector(register_trait::set1(scalyr)); }
    static Vector zero() { return Vector(register_trait::setzero()); }
    static Vector load_unaligned(const T* ptr) { return Vector(register_trait::load_unaligned(ptr)); }
    void          store_unaligned(T* ptr) const { register_trait::store_unaligned(ptr, m_data); }
    static Vector load_aligned(const T* ptr) { return Vector(register_trait::load_aligned(ptr)); }
    void          store_aligned(T* ptr) const { register_trait::store_aligned(ptr, m_data); }

    T operator[](u64 i) const
    {
        Assert(i < length);
        alignas(alignment) std::array<T, length> temp{};
        store_aligned(temp.data());
        return temp[i];
    }

    Vector operator-() const { return Vector(register_trait::sub(register_trait::setzero(), m_data)); }

    Vector operator+(const Vector& other) const { return Vector(register_trait::add(m_data, other.m_data)); }
    Vector operator-(const Vector& other) const { return Vector(register_trait::sub(m_data, other.m_data)); }
    Vector operator*(const Vector& other) const { return Vector(register_trait::mul(m_data, other.m_data)); }
    Vector operator/(const Vector& other) const { return Vector(register_trait::div(m_data, other.m_data)); }
    Vector operator&(const Vector& other) const { return Vector(register_trait::bitwise_and(m_data, other.m_data)); }
    Vector operator|(const Vector& other) const { return Vector(register_trait::bitwise_or(m_data, other.m_data)); }
    Vector operator^(const Vector& other) const { return Vector(register_trait::bitwise_xor(m_data, other.m_data)); }

    // clang-format off
    Vector operator+=(const Vector& other) const { *this = *this + other; return *this; }
    Vector operator-=(const Vector& other) const { *this = *this - other; return *this; }
    Vector operator*=(const Vector& other) const { *this = *this * other; return *this; }
    Vector operator/=(const Vector& other) const { *this = *this / other; return *this; }
    Vector operator&=(const Vector& other) const { *this = *this & other; return *this; }
    Vector operator|=(const Vector& other) const { *this = *this | other; return *this; }
    Vector operator^=(const Vector& other) const { *this = *this ^ other; return *this; }
    // clang-format on

    Vector operator==(const Vector& other) const { return Vector(register_trait::cmpeq(m_data, other.m_data)); }
    Vector operator!=(const Vector& other) const { return Vector(register_trait::cmpneq(m_data, other.m_data)); }
    Vector operator<(const Vector& other) const { return Vector(register_trait::cmplt(m_data, other.m_data)); }
    Vector operator<=(const Vector& other) const { return Vector(register_trait::cmple(m_data, other.m_data)); }
    Vector operator>(const Vector& other) const { return Vector(register_trait::cmpgt(m_data, other.m_data)); }
    Vector operator>=(const Vector& other) const { return Vector(register_trait::cmpge(m_data, other.m_data)); }

    Vector operator~() const
        requires(
            std::is_invocable_r_v<register_type, decltype(&register_trait::bitwise_xor), register_type, register_type>)
    {
        T all_ones;
        if constexpr (std::is_integral_v<T>)
        {
            all_ones = static_cast<T>(-1);
        }
        else
        {
            std::memset(&all_ones, 0xFF, sizeof(T));
        }
        return Vector(register_trait::bitwise_xor(m_data, register_trait::set1(all_ones)));
    }

    friend Vector bitwise_andnot(const Vector& keep, const Vector& invert_mask)
        requires(std::is_invocable_r_v<register_type,
                                       decltype(&register_trait::bitwise_andnot),
                                       register_type,
                                       register_type>)
    {
        return Vector(register_trait::bitwise_andnot(keep.m_data, invert_mask.m_data));
    }

    friend Vector sqrt(const Vector& v)
        requires(std::is_invocable_r_v<register_type, decltype(register_trait::sqrt), register_type>)
    {
        return Vector(register_trait::sqrt(v.m_data));
    }
    friend Vector abs(const Vector& v)
        requires(std::is_invocable_r_v<register_type, decltype(register_trait::abs), register_type>)
    {
        return Vector(register_trait::abs(v.m_data));
    }
    friend Vector min(const Vector& a, const Vector& b)
        requires(std::is_invocable_r_v<register_type, decltype(register_trait::min), register_type, register_type>)
    {
        return Vector(register_trait::min(a.m_data, b.m_data));
    }
    friend Vector max(const Vector& a, const Vector& b)
        requires(std::is_invocable_r_v<register_type, decltype(register_trait::max), register_type, register_type>)
    {
        return Vector(register_trait::max(a.m_data, b.m_data));
    }

#define FRIEND_FLOATING_SINGLE_ARG(name)                                                                               \
    friend Vector name(const Vector& v)                                                                                \
        requires(std::is_floating_point_v<element_type> &&                                                             \
                 std::is_invocable_r_v<register_type, decltype(register_trait::name), register_type>)                  \
    {                                                                                                                  \
        return Vector(register_trait::name(v.m_data));                                                                 \
    }
    FRIEND_FLOATING_SINGLE_ARG(floor);
    FRIEND_FLOATING_SINGLE_ARG(ceil);
    FRIEND_FLOATING_SINGLE_ARG(round_nearest);
    FRIEND_FLOATING_SINGLE_ARG(truncate);
    FRIEND_FLOATING_SINGLE_ARG(rsqrt);
    FRIEND_FLOATING_SINGLE_ARG(rcp);
#undef FRIEND_FLOATING_SINGLE_ARG

    friend Vector blendv(const Vector& v_false, const Vector& v_true, const Vector& mask)
        requires(std::is_invocable_r_v<register_type,
                                       decltype(register_trait::blendv),
                                       register_type,
                                       register_type,
                                       register_type>)
    {
        return Vector(register_trait::blendv(v_false.m_data, v_true.m_data, mask.m_data));
    }

    i32 movemask() const { return register_trait::movemask(m_data); }

    friend Vector sll(const Vector& v, i32 count)
        requires(std::is_integral_v<element_type> &&
                 std::is_invocable_r_v<register_type, decltype(&register_trait::slli), register_type, i32>)
    {
        return Vector(register_trait::slli(v.m_data, count));
    }
    friend Vector srl(const Vector& v, i32 count)
        requires(std::is_integral_v<element_type> &&
                 std::is_invocable_r_v<register_type, decltype(&register_trait::srli), register_type, i32>)
    {
        return Vector(register_trait::srli(v.m_data, count));
    }
    Vector operator<<(i32 count) const { return sll(*this, count); }
    Vector operator<<=(i32 count) const
    {
        *this = *this << count;
        return *this;
    }
    Vector operator>>(i32 count) const { return srl(*this, count); }
    Vector operator>>=(i32 count) const
    {
        *this = *this >> count;
        return *this;
    }

    friend std::ostream& operator<<(std::ostream& os, const Vector& v)
    {
        alignas(alignment) std::array<T, length> temp;
        register_trait::store_aligned(temp.data(), v.m_data());
        os << "Vector(" << length << ", [";
        for (u64 i = 0; i < length; ++i)
        {
            if constexpr (std::is_same_v<T, i8> || std::is_same_v<T, u8>)
            {
                os << cast(i32) temp[i];
            }
            else
            {
                os << temp[i];
            }
            if (i < length - 1)
                os << ", ";
        }
        os << "])";
        return os;
    }
};

} // namespace simd

#endif