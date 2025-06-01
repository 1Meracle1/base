#ifndef INSTRUCTION_SETS_H
#define INSTRUCTION_SETS_H

#include "../types.h"
#include "cpuid.h"
#include <concepts>
#include <utility>

namespace simd
{
template <typename T>
concept Scalar = std::is_integral_v<T> || std::is_floating_point_v<T>;

template <typename T>
concept InstructionSetConcept = requires {
    T::name;
    T::is_data_parallel;
};

struct ScalarInstructionSet
{
    static constexpr const char* name             = "Scalar";
    static constexpr bool        is_data_parallel = false;
};
static_assert(InstructionSetConcept<ScalarInstructionSet>);

struct SSE2InstructionSet
{
    static constexpr const char* name             = "SSE2";
    static constexpr bool        is_data_parallel = true;
};
static_assert(InstructionSetConcept<SSE2InstructionSet>);

struct SSE3InstructionSet
{
    static constexpr const char* name             = "SSE3";
    static constexpr bool        is_data_parallel = true;
};
static_assert(InstructionSetConcept<SSE3InstructionSet>);

struct SSE4InstructionSet
{
    static constexpr const char* name             = "SSE4";
    static constexpr bool        is_data_parallel = true;
};
static_assert(InstructionSetConcept<SSE4InstructionSet>);

struct SSE41InstructionSet
{
    static constexpr const char* name             = "SSE4.1";
    static constexpr bool        is_data_parallel = true;
};
static_assert(InstructionSetConcept<SSE41InstructionSet>);

struct AVXInstructionSet
{
    static constexpr const char* name             = "AVX";
    static constexpr bool        is_data_parallel = true;
};
static_assert(InstructionSetConcept<AVXInstructionSet>);

struct AVX2InstructionSet
{
    static constexpr const char* name             = "AVX2";
    static constexpr bool        is_data_parallel = true;
};
static_assert(InstructionSetConcept<AVX2InstructionSet>);

template <typename T, typename InstructionSet> struct RegisterTrait;

template <typename T> struct RegisterTrait<T, ScalarInstructionSet>;

template <typename T, typename VectorizedF, typename ScalarF>
inline auto dispatch_instruction_set(u64 length, VectorizedF vectorized, ScalarF scalar)
// requires(std::is_same_v<std::invoke_result_t<VectorizedF>, std::invoke_result_t<ScalarF>> &&
//          !std::is_void_v<std::invoke_result_t<VectorizedF>>)
{
#define _instruction_set_dispatch_check(instruction_set, feature, func)                                                \
    if constexpr (requires { RegisterTrait<T, instruction_set>::length != 0; })                                        \
    {                                                                                                                  \
        if (length >= RegisterTrait<T, instruction_set>::length)                                                       \
        {                                                                                                              \
            if (supported_features().feature)                                                                          \
            {                                                                                                          \
                return std::forward<VectorizedF>(vectorized).template operator()<instruction_set>();                   \
            }                                                                                                          \
        }                                                                                                              \
    }
    _instruction_set_dispatch_check(AVX2InstructionSet, avx2, vectorized);
    _instruction_set_dispatch_check(AVXInstructionSet, avx, vectorized);
    _instruction_set_dispatch_check(SSE41InstructionSet, sse41, vectorized);
    _instruction_set_dispatch_check(SSE3InstructionSet, sse3, vectorized);
    _instruction_set_dispatch_check(SSE2InstructionSet, sse2, vectorized);
    return std::forward<ScalarF>(scalar)();
#undef _instruction_set_dispatch_check
}

template <typename T, typename VectorizedF, typename ScalarF, typename... Args>
inline auto dispatch_instruction_set(u64 length, VectorizedF&& vectorized, ScalarF&& scalar, Args&&... args)
// requires(std::is_same_v<std::invoke_result_t<VectorizedF>, std::invoke_result_t<ScalarF>> &&
//          !std::is_void_v<std::invoke_result_t<VectorizedF>>)
{

#define _instruction_set_dispatch_check(instruction_set, feature, func)                                                \
    if constexpr (requires { RegisterTrait<T, instruction_set>::length != 0; })                                        \
    {                                                                                                                  \
        if (length >= RegisterTrait<T, instruction_set>::length)                                                       \
        {                                                                                                              \
            if (supported_features().feature)                                                                          \
            {                                                                                                          \
                return std::forward<VectorizedF>(vectorized)                                                           \
                    .template operator()<instruction_set>(std::forward<Args>(args)...);                                \
            }                                                                                                          \
        }                                                                                                              \
    }
    _instruction_set_dispatch_check(AVX2InstructionSet, avx2, vectorized);
    _instruction_set_dispatch_check(AVXInstructionSet, avx, vectorized);
    _instruction_set_dispatch_check(SSE41InstructionSet, sse41, vectorized);
    _instruction_set_dispatch_check(SSE3InstructionSet, sse3, vectorized);
    _instruction_set_dispatch_check(SSE2InstructionSet, sse2, vectorized);
    return std::forward<ScalarF>(scalar)(std::forward<Args>(args)...);
#undef _instruction_set_dispatch_check
}

} // namespace simd

#endif