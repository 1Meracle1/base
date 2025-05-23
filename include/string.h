#ifndef STRING_H
#define STRING_H

#include "array.h"
#include "memory.h"
#include <utility>

namespace string_impl
{

// https://tools.ietf.org/html/rfc3629
// clang-format off
static Slice<const u8> UTF8_CHAR_WIDTH = {
    // 1  2  3  4  5  6  7  8  9  A  B  C  D  E  F
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 0
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 1
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 2
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 3
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 4
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 5
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 6
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 7
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 8
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 9
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // A
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // B
    0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, // C
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, // D
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, // E
    4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // F
};

static constexpr std::size_t utf8_char_width(u8 b) { return cast(std::size_t)(UTF8_CHAR_WIDTH[b]); }

static constexpr u8 CONT_MASK = 0b00111111;
// clang-format on

// returns `true` if next chunk was found, `false` otherwise
static constexpr bool next_utf8_chunk(const Slice<u8>& bytes, Slice<u8>& valid_bytes, Slice<u8>& invalid_bytes)
{
    if (bytes.empty())
    {
        valid_bytes   = Slice<u8>();
        invalid_bytes = Slice<u8>();
        return false;
    }

    constexpr u8 TAG_CONT_U8 = 128;
    std::size_t  i           = 0;
    std::size_t  valid_up_to = 0;
    while (i < bytes.len())
    {
        u8 byte = bytes[i];
        i++;
        if (byte < 128)
        {
            // ASCII byte
        }
        else
        {
            auto width = utf8_char_width(byte);
            if (width == 2)
            {
                if ((bytes[i] & 192) != TAG_CONT_U8)
                {
                    break;
                }
                i++;
            }
            else if (width == 3)
            {
                u8 next = bytes[i];
                if (byte == 0xE0 && next >= 0xA0 && next <= 0xBF)
                {
                }
                else if (byte >= 0xE1 && byte <= 0xEC && next >= 0x80 && next <= 0xBF)
                {
                }
                else if (byte == 0xED && next >= 0x80 && next <= 0x9F)
                {
                }
                else if (byte >= 0xEE && byte <= 0xEF && next >= 0x80 && next <= 0xBF)
                {
                }
                else
                {
                    break;
                }
                i++;
                if ((bytes[i] & 192) != TAG_CONT_U8)
                {
                    break;
                }
                i++;
            }
            else if (width == 4)
            {
                u8 next = bytes[i];
                if (byte == 0xF0 && next >= 0x90 && next <= 0xBF)
                {
                }
                else if (byte >= 0xF1 && byte <= 0xF3 && next >= 0x80 && next <= 0xBF)
                {
                }
                else if (byte == 0xF4 && next >= 0x80 && next <= 0x8F)
                {
                }
                else
                {
                    break;
                }
                i++;
                if ((bytes[i] & 192) != TAG_CONT_U8)
                {
                    break;
                }
                i++;
                if ((bytes[i] & 192) != TAG_CONT_U8)
                {
                    break;
                }
                i++;
            }
            else
            {
                break;
            }
        }

        valid_up_to = i;
    }

    Slice<u8> inspected = bytes.slice_to(i);
    inspected.split_at_unchecked(valid_up_to, valid_bytes, invalid_bytes);
    return true;
}
} // namespace string_impl

struct String
{
    using size_type  = std::size_t;
    using value_type = u8;

  private:
    Array<value_type> m_data{};

    String() = default;

    String(Slice<value_type> bytes)
        : m_data(bytes)
    {
    }

  public:
    String(Allocator* allocator)
        : m_data(allocator)
    {
    }

    String(Allocator* allocator, size_type capacity)
        : m_data(allocator, capacity)
    {
    }

    String(Allocator* allocator, size_type capacity, size_type length)
        : m_data(allocator, capacity, length)
    {
    }

    String(String&& other)
        : m_data(std::exchange(other.m_data, {}))
    {
    }

    String& operator=(String&& other)
    {
        m_data.free_allocated_memory();
        m_data = std::exchange(other.m_data, {});
        return *this;
    }

    static String from_utf8_lossy(Allocator* allocator, Slice<const char> bytes)
    {
        return from_utf8_lossy(allocator, bytes.reinterpret_elements_as<value_type>());
    }

    static String from_utf8_lossy(Allocator* allocator, Slice<value_type> bytes)
    {
        Slice<value_type> valid_bytes;
        Slice<value_type> invalid_bytes;
        bool              found_next_chunk = string_impl::next_utf8_chunk(bytes, valid_bytes, invalid_bytes);
        if (!found_next_chunk)
        {
            return String();
        }
        if (invalid_bytes.empty())
        {
            return String(std::move(valid_bytes));
        }

        auto REPLACEMENT = Slice("uFFFD").reinterpret_elements_as<value_type>();

        String res{allocator, bytes.len()};
        while (found_next_chunk)
        {
            res.push_str(valid_bytes);
            if (invalid_bytes.not_empty())
            {
                res.push_str(REPLACEMENT);
            }
            size_type remaining_start = valid_bytes.len() + invalid_bytes.len();
            auto      remaining_bytes = bytes.slice_from(remaining_start);
            found_next_chunk          = string_impl::next_utf8_chunk(remaining_bytes, valid_bytes, invalid_bytes);
        }
        return res;
    }

    void push_str(Slice<value_type> bytes) { m_data.append_many(bytes); }

    // template <size_type N>
    // constexpr explicit String(value_type (&arr)[N]) noexcept
    //     : m_data(arr, N)
    // {
    // }
};

#endif