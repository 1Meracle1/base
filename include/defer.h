#ifndef DEFER_H
#define DEFER_H

#include <utility>
#include "defines.h"

template <typename F> struct ScopeExit
{
    ScopeExit(F&& callable) : f(std::forward<F>(callable)) {}
    ~ScopeExit() { f(); }

    ScopeExit(ScopeExit&&)                 = delete;
    ScopeExit(const ScopeExit&)            = delete;
    ScopeExit& operator=(ScopeExit&&)      = delete;
    ScopeExit& operator=(const ScopeExit&) = delete;

    F f;
};

#define defer(lambda) auto CONCATENATE(_defer_, __COUNTER__) = ScopeExit(lambda)

#endif