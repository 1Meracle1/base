#pragma once

#include "parse.h"
#include "string.h"
#include <chrono>
#include <cstdio>
#include <ctime>
#include <ratio>
#include <string>
#include <iomanip>
#include <sstream>

using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;

static inline TimePoint time_now() { return std::chrono::high_resolution_clock::now(); }

static inline u64 time_diff_sec(const TimePoint& start, const TimePoint& end)
{
    return cast(u64) std::chrono::duration<double>(end - start).count();
}

static inline u64 time_diff_milli(const TimePoint& start, const TimePoint& end)
{
    return cast(u64) std::chrono::duration<double, std::milli>(end - start).count();
}

static inline u64 time_diff_micro(const TimePoint& start, const TimePoint& end)
{
    return cast(u64) std::chrono::duration<double, std::micro>(end - start).count();
}

static inline u64 time_diff_nano(const TimePoint& start, const TimePoint& end)
{
    return cast(u64) std::chrono::duration<double, std::nano>(end - start).count();
}

#define MeasureTime(lambda) auto CONCATENATE(_measure_time_, __COUNTER__) = MeasureTimeScope(lambda)

#define MeasureTimeMicro(label)                                                                                        \
    auto CONCATENATE(_measure_time_, __COUNTER__) =                                                                    \
        MeasureTimeScope([](const TimePoint& start, const TimePoint& end)                                              \
                         { std::cout << (label) << " took " << time_diff_micro(start, end) << " micros.\n"; })

#define MeasureTimeMilli(label)                                                                                        \
    auto CONCATENATE(_measure_time_, __COUNTER__) =                                                                    \
        MeasureTimeScope([](const TimePoint& start, const TimePoint& end)                                              \
                         { std::cout << (label) << " took " << time_diff_milli(start, end) << " millis.\n"; })

struct MeasureTimeScope
{
    using callback_type = std::function<void(const TimePoint& start, const TimePoint& end)>;

    MeasureTimeScope(callback_type&& func)
        : start(time_now())
        , f(std::forward<callback_type>(func))
    {
    }

    ~MeasureTimeScope() { f(start, time_now()); }

    TimePoint     start;
    callback_type f;
};

static inline String
timestamp_to_rfc3339(Allocator* allocator, const TimePoint& tp, bool output_fractional_seconds = true)
{
    auto sctp = std::chrono::system_clock::now() + std::chrono::duration_cast<std::chrono::system_clock::duration>(
                                                       tp - std::chrono::high_resolution_clock::now());
    std::time_t tt = std::chrono::system_clock::to_time_t(sctp);
    auto        ms =
        std::chrono::duration_cast<std::chrono::microseconds>(tp.time_since_epoch() % std::chrono::seconds(1)).count();
    if (ms < 0)
    {
        ms += 1000000;
    }
    // YYYY-MM-DDTHH:MM:SSZ or YYYY-MM-DDTHH:MM:SS.123456Z
    u64         reserved_size = cast(u64)(output_fractional_seconds ? 32 : 21);
    String      result{allocator, reserved_size};
    char        buffer[64];
    std::tm     tm_utc  = *std::gmtime(&tt);
    std::size_t written = std::strftime(buffer, sizeof(buffer), "%FT%T", &tm_utc);
    if (written > 0)
    {
        Slice<const char> bytes(buffer, written);
        result.push_str(bytes);
    }
    else
    {
        return String(); // error handling?
    }

    if (output_fractional_seconds)
    {
        result.push(cast(u32) '.');

        written = std::snprintf(buffer, sizeof(buffer), "%06ld", ms);
        if (written > 0)
        {
            Slice<const char> bytes(buffer, written);
            result.push_str(bytes);
        }
        else
        {
            result.push_str("000000"); // error handling?
        }
    }
    result.push(cast(u32) 'Z');

    return result;
}

// expected format: YYYY-MM-DDTHH:MM:SS[.ffffff]Z
inline bool timestamp_from_rfc3339(Slice<u8> str, TimePoint& tp)
{
    if (str.len() < 20)
        return false;

    std::tm tm_val{};

    auto      parse_err  = ParseIntFromStringError::None;
    Slice<u8> curr_slice = str;

#define expect_and_advance(c)                                                                                          \
    if (curr_slice.first() != '-')                                                                                     \
        return false;                                                                                                  \
    curr_slice = curr_slice.drop(1)

#define parse_and_advance(n, to)                                                                                       \
    parse_err = parse_int(curr_slice.take(n), to);                                                                     \
    if (parse_err != ParseIntFromStringError::None)                                                                    \
        return false;                                                                                                  \
    curr_slice = curr_slice.drop(n)

    parse_and_advance(4, tm_val.tm_year);
    tm_val.tm_year -= 1900;
    expect_and_advance('-');
    parse_and_advance(2, tm_val.tm_mon);
    tm_val.tm_mon -= 1;
    expect_and_advance('-');
    parse_and_advance(2, tm_val.tm_mday);
    expect_and_advance('T');
    parse_and_advance(2, tm_val.tm_hour);
    expect_and_advance(':');
    parse_and_advance(2, tm_val.tm_min);
    expect_and_advance(':');
    parse_and_advance(2, tm_val.tm_sec);

    u64 fractional_microseconds = 0;
    if (curr_slice.not_empty() && curr_slice.first() == '.')
    {
        curr_slice = curr_slice.drop(1);
        if (curr_slice.last() == 'Z')
        {
            curr_slice = curr_slice.drop_back(1);
        }
        if (curr_slice.len() > 6)
        {
            return false;
        }
        curr_slice = curr_slice.take_max(6);
        parse_err  = parse_int(curr_slice, fractional_microseconds);
        if (parse_err != ParseIntFromStringError::None)
        {
            return false;
        }
    }

    std::time_t seconds_since_epoch;
#if defined(_WIN32) || defined(_WIN64)
    seconds_since_epoch = _mkgmtime(&tm_val);
#else
    // timegm is not in standard C++, but common on POSIX systems
    seconds_since_epoch = timegm(&tm_val);
#endif

    // timegm/_mkgmtime return -1 on error.
    // -1 can be a valid time_t for dates before 1970-01-01, but RFC3339 usually implies positive timestamps.
    if (seconds_since_epoch == -1 && (tm_val.tm_year * 100 + tm_val.tm_mon * 10 + tm_val.tm_mday > 700101))
    {              // Heuristic: error if date is >= 1970-01-01
        return {}; // Error in time conversion
    }

    auto system_tp_result = std::chrono::system_clock::from_time_t(seconds_since_epoch);
    system_tp_result += std::chrono::microseconds(fractional_microseconds);

    tp = std::chrono::time_point_cast<std::chrono::high_resolution_clock::duration>(
        std::chrono::high_resolution_clock::now() +
        std::chrono::duration_cast<std::chrono::high_resolution_clock::duration>(system_tp_result -
                                                                                 std::chrono::system_clock::now()));

    return true;
}
