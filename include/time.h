#pragma once

#include "list.h"
#include "parse_num.h"
#include "string.h"
#include "memory.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <iostream>
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

/*
    Credits go to https://github.com/edanor/umesimd/blob/master/microbenchmarks/utilities/TimingStatistics.h
    Example:

    Slice<u8> data{};
    bool      ok = read_entire_file(allocator, ByteSliceFromCstr("include/string.h"), data);
    std::cout << "file read successfully? - " << std::boolalpha << ok << ", bytes: " << data.len() << '\n';

    MeasureTimeStats stats_utf8_lossy{allocator};
    for (u64 i = 0; i < 10000; ++i)
    {
        auto start = time_now();
        auto str   = String::from_utf8_lossy(allocator, data);
        auto diff  = time_diff_nano(start, time_now());
        stats_utf8_lossy.append(diff);
    }

    MeasureTimeStats stats_raw{allocator};
    for (u64 i = 0; i < 10000; ++i)
    {
        auto start = time_now();
        auto str   = String::from_raw(allocator, data);
        auto diff  = time_diff_nano(start, time_now());
        stats_raw.append(diff);
    }

    stats_raw.print_summary_with_reference(
        ByteSliceFromCstrZeroTerm("Construction of string from raw bytes vs utf8 lossy: "),
        stats_utf8_lossy
    );
*/
class MeasureTimeStats
{
  private:
    SinglyLinkedList<u64> m_measurements;
    f32                   m_average  = 0.0f;
    f32                   m_variance = 0.0f;
    u64                   m_count    = 0;

  public:
    explicit MeasureTimeStats(Allocator* allocator)
        : m_measurements{allocator}
    {
    }

    void append(u64 elapsed)
    {
        m_measurements.push_back(elapsed);
        f32 diff = cast(f32) elapsed - m_average;
        m_average += diff / (1.0f + cast(f32) m_count);
        m_variance += diff * (f32(elapsed) - m_average);
        ++m_count;
    }

    f32 average() const { return m_average; }
    f32 std_dev() const { return m_count > 0 ? std::sqrtf(m_variance) / cast(f32) m_count : 0.0f; }

    f32 speedup(f32 reference) const { return reference / m_average; }
    f32 speedup(const MeasureTimeStats& reference) const
    {
        return m_average > 0.0f ? reference.m_average / m_average : 0.0f;
    }

    f32 confidence90() { return 1.645f * std_dev() / std::sqrtf(cast(f32) m_count); }
    f32 confidence95() { return 1.96f * std_dev() / std::sqrtf(cast(f32) m_count); }

    void print_summary(Slice<u8> summary_description) const
    {
        std::cout << summary_description << average() << "ns, dev: " << std_dev() << '\n';
    }

    void print_summary_with_reference(Slice<u8> summary_description, const MeasureTimeStats& reference) const
    {
        std::cout << summary_description << average() << "ns, dev: " << std_dev()
                  << "ns (speedup: " << speedup(reference) << ")\n";
    }
};

class MeasureTimeStats2
{
  private:
    u64       m_total    = 0;
    f32       m_average  = 0.0f;
    f32       m_variance = 0.0f;
    u64       m_count    = 0;
    TimePoint m_start    = {};

  public:
    void start() { m_start = time_now(); }

    void end()
    {
        u64 elapsed = time_diff_nano(m_start, time_now());
        m_total += elapsed;
        f32 diff = cast(f32) elapsed - m_average;
        m_average += diff / (1.0f + cast(f32) m_count);
        m_variance += diff * (f32(elapsed) - m_average);
        ++m_count;
    }

    f32 average() const { return m_average; }
    f32 std_dev() const { return m_count > 0 ? std::sqrtf(m_variance) / cast(f32) m_count : 0.0f; }

    f32 speedup(f32 reference) const { return reference / m_average; }
    f32 speedup(const MeasureTimeStats2& reference) const
    {
        return m_average > 0.0f ? reference.m_average / m_average : 0.0f;
    }

    f32 confidence90() { return 1.645f * std_dev() / std::sqrtf(cast(f32) m_count); }
    f32 confidence95() { return 1.96f * std_dev() / std::sqrtf(cast(f32) m_count); }

    void print_summary(Slice<u8> summary_description) const
    {
        std::cout << summary_description << average() << "ns, dev: " << std_dev() << '\n';
    }

    void print_summary_ms(Slice<u8> summary_description) const
    {
        std::cout << summary_description << ", total: " << m_total / 1000 / 1000
                  << " millis, average: " << average() / 1000 / 1000 << " millis\n";
    }

    void print_summary_with_reference_ms(Slice<u8> summary_description, const MeasureTimeStats2& reference) const
    {
        std::cout << summary_description << ", total: " << m_total / 1000 / 1000
                  << " millis, average: " << average() / 1000 / 1000 << " millis (speedup:" << speedup(reference)
                  << ")\n";
    }
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

    #if defined (_MSC_VER)
        written = std::snprintf(buffer, sizeof(buffer), "%06lld", ms);
    #else
        written = std::snprintf(buffer, sizeof(buffer), "%06ld", ms);
    #endif
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
