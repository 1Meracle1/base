#ifndef STRING_H
#define STRING_H

#include "array.h"
#include "list.h"
#include "memory.h"
#include <cstddef>
#include <iterator>
#include <utility>
#include <iostream>

namespace string_impl
{

// https://tools.ietf.org/html/rfc3629
// clang-format off
static const u8 UTF8_CHAR_WIDTH[256] = {
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
// clang-format on

constexpr u8  CONT_MASK             = 0x3F;
constexpr u8  TAG_CONT              = 0x80; // 0b10000000
constexpr u8  TAG_TWO_B             = 0xC0; // 0b11000000
constexpr u8  TAG_THREE_B           = 0xE0; // 0b11100000
constexpr u8  TAG_FOUR_B            = 0xF0; // 0b11110000
constexpr u32 MAX_ONE_B             = 0x80;
constexpr u32 MAX_TWO_B             = 0x800;
constexpr u32 MAX_THREE_B           = 0x10000;
constexpr u32 REPLACEMENT_CODEPOINT = 0xFFFD;

// returns `true` if next chunk was found, `false` otherwise
static constexpr bool next_utf8_chunk(const Slice<u8>& bytes, Slice<u8>& valid_bytes, Slice<u8>& invalid_bytes)
{
    if (bytes.empty())
    {
        valid_bytes   = Slice<u8>();
        invalid_bytes = Slice<u8>();
        return false;
    }

    std::size_t i           = 0;
    std::size_t valid_up_to = 0;
    while (i < bytes.len())
    {
        u8 byte = bytes[i];
        i++;
        if (byte < TAG_CONT)
        {
            // ASCII byte
        }
        else
        {
            auto width = utf8_char_width(byte);
            if (width == 2)
            {
                if ((bytes[i] & 192) != TAG_CONT)
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
                if ((bytes[i] & 192) != TAG_CONT)
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
                if ((bytes[i] & 192) != TAG_CONT)
                {
                    break;
                }
                i++;
                if ((bytes[i] & 192) != TAG_CONT)
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

static constexpr std::size_t utf8_len(u32 codepoint)
{
    if (codepoint < MAX_ONE_B)
        return 1;
    if (codepoint < MAX_TWO_B)
        return 2;
    if (codepoint < MAX_THREE_B)
        return 3;
    return 4;
}

// SAFETY: `encoded` buffer must be large enough to fit 4 bytes
static constexpr void utf8_encode_raw(u32 codepoint, Slice<u8>& encoded)
{
    u64 len = utf8_len(codepoint);
    Assert(encoded.len() >= len);
    if (len == 1)
    {
        encoded.m_ptr[0] = cast(u8) codepoint;
    }
    else if (len == 2)
    {
        encoded.m_ptr[0] = cast(u8)(codepoint >> 6 & 0x1F) | TAG_TWO_B;
        encoded.m_ptr[1] = cast(u8)(codepoint & 0x3F) | TAG_CONT;
    }
    else if (len == 3)
    {
        encoded.m_ptr[0] = cast(u8)(codepoint >> 12 & 0x0F) | TAG_THREE_B;
        encoded.m_ptr[1] = cast(u8)(codepoint >> 6 & 0x3F) | TAG_CONT;
        encoded.m_ptr[2] = cast(u8)(codepoint & 0x3F) | TAG_CONT;
    }
    else if (len == 4)
    {
        encoded.m_ptr[0] = cast(u8)(codepoint >> 18 & 0x07) | TAG_FOUR_B;
        encoded.m_ptr[1] = cast(u8)(codepoint >> 12 & 0x3F) | TAG_CONT;
        encoded.m_ptr[2] = cast(u8)(codepoint >> 6 & 0x3F) | TAG_CONT;
        encoded.m_ptr[3] = cast(u8)(codepoint & 0x3F) | TAG_CONT;
    }
    else
    {
        Assertr(len <= 4, "Given generalized UTF-8 encoding only supports codepoints of length <= 4");
    }
}

static constexpr u32 utf8_decode_first_codepoint(Slice<const u8> bytes, std::size_t& bytes_consumed)
{
    Assert(bytes.not_empty());

    u8 b0          = bytes.front();
    bytes_consumed = 1;
    if (b0 < MAX_ONE_B)
        return cast(u32) b0;

    if ((b0 & 0xC0) == MAX_ONE_B)
        return REPLACEMENT_CODEPOINT;
    if (b0 == 0xC0 || b0 == 0xC1 || b0 >= 0xF5)
        return REPLACEMENT_CODEPOINT;

    u32 codepoint = 0;
    if ((b0 & 0xE0) == TAG_TWO_B)
    {
        if (bytes.len() < 2) // truncated
            return REPLACEMENT_CODEPOINT;
        u8 b1 = bytes[1];
        if ((b1 & 0xC0) != TAG_CONT)
            return REPLACEMENT_CODEPOINT;

        codepoint = (cast(u32)(b0 & 0x1F) << 6) | cast(u32)(b1 & CONT_MASK);
        if (codepoint < MAX_ONE_B)
            return REPLACEMENT_CODEPOINT;
        bytes_consumed = 2;
    }
    else if ((b0 & 0xF0) == TAG_THREE_B)
    {
        if (bytes.len() < 3) // truncated
            return REPLACEMENT_CODEPOINT;

        u8 b1 = bytes[1];
        u8 b2 = bytes[2];
        if ((b1 & TAG_TWO_B) != TAG_CONT || (b2 & TAG_TWO_B) != TAG_CONT)
            return REPLACEMENT_CODEPOINT;

        codepoint = (cast(u32)(b0 & 0x0F) << 12) | (cast(u32)(b1 & CONT_MASK) << 6) | cast(u32)(b2 & CONT_MASK);
        if (codepoint < 0x800 || (b0 == 0xE0 && b1 < 0xA0) || (codepoint >= 0xD800 && codepoint <= 0xDFFF))
            return REPLACEMENT_CODEPOINT;
        bytes_consumed = 3;
    }
    else if ((b0 & 0xF8) == TAG_FOUR_B)
    {
        if (bytes.len() < 4) // truncated
            return REPLACEMENT_CODEPOINT;

        u8 b1 = bytes[1];
        u8 b2 = bytes[2];
        u8 b3 = bytes[3];
        if ((b1 & 0xC0) != TAG_CONT || (b2 & 0xC0) != TAG_CONT || (b3 & 0xC0) != TAG_CONT)
            return REPLACEMENT_CODEPOINT;

        codepoint = (cast(u32)(b0 & 0x0F) << 18) | (cast(u32)(b1 & CONT_MASK) << 12) |
                    (cast(u32)(b2 & CONT_MASK) << 6) | cast(u32)(b3 & CONT_MASK);
        if (codepoint < 0x100000 || (b0 == 0xE0 && b1 < 0x90) || codepoint > 0x10FFFF)
            return REPLACEMENT_CODEPOINT;
        bytes_consumed = 4;
    }
    else
    {
        return REPLACEMENT_CODEPOINT;
    }
    return codepoint;
}
} // namespace string_impl

/*
    // String str = String::from_utf8_lossy(
    //     allocator, "In the quiet twilight, dreams unfold, soft whispers of a story untold.\n"
    //                "ćeść panśtwu\n"
    //                "月明かりが静かに照らし出し、夢を見る心の奥で詩が静かに囁かれる\n"
    //                "Stars collide in the early light of hope, echoing the silent call of the night.\n"
    //                "夜の静寂、希望と孤独が混ざり合うその中で詩が永遠に続く\n");
    // for (u32 codepoint : str)
    // {
    //     std::cout << std::hex << codepoint << (codepoint == cast(u32)0xA ? '\n' : ' ');
    // }
*/

class CodepointIterator
{
  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type        = u32;
    using difference_type   = std::ptrdiff_t;
    using size_type         = std::size_t;
    using pointer           = value_type*;
    using const_pointer     = const value_type*;
    using reference         = value_type&;
    using const_reference   = const value_type&;

  private:
    const u8* m_current_byte_ptr;
    const u8* m_end_byte_ptr;
    u32       m_current_codepoint;
    size_type m_current_current_codepoint_width;

    void decode_current()
    {
        if (m_current_byte_ptr >= m_end_byte_ptr)
        {
            m_current_codepoint               = 0;
            m_current_current_codepoint_width = 0;
            return;
        }
        Slice<const u8> remaining_bytes{m_current_byte_ptr, cast(size_type)(m_end_byte_ptr - m_current_byte_ptr)};
        m_current_codepoint =
            string_impl::utf8_decode_first_codepoint(remaining_bytes, m_current_current_codepoint_width);
        if (m_current_current_codepoint_width == 0 && m_current_byte_ptr < m_end_byte_ptr)
        {
            m_current_codepoint               = string_impl::REPLACEMENT_CODEPOINT;
            m_current_current_codepoint_width = 1;
        }
    }

  public:
    CodepointIterator()
        : m_current_byte_ptr(nullptr)
        , m_end_byte_ptr(nullptr)
        , m_current_codepoint(0)
        , m_current_current_codepoint_width(0)
    {
    }

    CodepointIterator(const u8* ptr, const u8* end_ptr)
        : m_current_byte_ptr(cast(u8*) ptr)
        , m_end_byte_ptr(cast(u8*) end_ptr)
    {
        decode_current();
    }

    value_type operator*() const
    {
        Assert(m_current_byte_ptr < m_end_byte_ptr);
        return m_current_codepoint;
    }

    const_pointer operator->() const
    {
        Assert(m_current_byte_ptr < m_end_byte_ptr);
        return &m_current_codepoint;
    }

    CodepointIterator& operator++()
    {
        Assert(m_current_byte_ptr < m_end_byte_ptr);
        m_current_byte_ptr += m_current_current_codepoint_width;
        if (m_current_byte_ptr > m_end_byte_ptr)
            m_current_byte_ptr = m_end_byte_ptr;
        decode_current();
        return *this;
    }

    CodepointIterator operator++(int)
    {
        CodepointIterator it = *this;
        ++(*this);
        return it;
    }

    bool operator==(const CodepointIterator& other) const { return m_current_byte_ptr == other.m_current_byte_ptr; }
    bool operator!=(const CodepointIterator& other) const { return m_current_byte_ptr != other.m_current_byte_ptr; }
};

struct String
{
    using size_type  = std::size_t;
    using value_type = u8;

  private:
    Array<value_type> m_data{};

  public:
    // String(Slice<value_type> bytes)
    //     : m_data(bytes)
    // {
    // }
    // String(Slice<const char> bytes)
    //     : m_data(bytes.reinterpret_elements_as<u8>())
    // {
    // }

    String() = default;

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
        auto res = bytes.reinterpret_elements_as<value_type>();
        if (res.back() == 0)
        {
            res = res.slice_to(res.len() - 1);
        }
        return from_utf8_lossy(allocator, res);
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
            String str{allocator, bytes.len()};
            str.push_str(valid_bytes);
            return str;
        }

        String res{allocator, bytes.len()};
        while (found_next_chunk)
        {
            res.push_str(valid_bytes);
            if (invalid_bytes.not_empty())
            {
                res.push(cast(u8) string_impl::REPLACEMENT_CODEPOINT);
            }
            size_type remaining_start = valid_bytes.len() + invalid_bytes.len();
            auto      remaining_bytes = bytes.slice_from(remaining_start);
            found_next_chunk          = string_impl::next_utf8_chunk(remaining_bytes, valid_bytes, invalid_bytes);
        }
        return res;
    }

    static String from_raw(Allocator* allocator, Slice<const char> bytes)
    {
        auto res = bytes.reinterpret_elements_as<value_type>();
        if (res.back() == 0)
        {
            res = res.slice_to(res.len() - 1);
        }
        return from_raw(allocator, res);
    }

    static String from_raw(Allocator* allocator, Slice<value_type> bytes)
    {
        String str{allocator, bytes.len()};
        str.push_str(bytes);
        return str;
    }

    void check_reserve(size_type added_elements_length = 1) { m_data.check_reserve(added_elements_length); }

    void push(u32 codepoint)
    {
        auto len = string_impl::utf8_len(codepoint);
        if (len == 1)
        {
            m_data.append(cast(u8) codepoint);
        }
        else
        {
            m_data.check_reserve(4);
            Slice<u8> encoded{m_data.end(), 4};
            string_impl::utf8_encode_raw(codepoint, encoded);
        }
    }

    void push(value_type byte) { m_data.append(byte); }

    void push_str(Slice<const char> cstr) { push_str(cstr.reinterpret_elements_as<value_type>()); }

    void push_str(Slice<value_type> bytes) { m_data.append_many(bytes); }

    void reset_length() { m_data.reset_length(); }

    // methods handling string as an array of bytes
    constexpr size_type      len_bytes() const { return m_data.len(); }
    bool                     empty() const { return len_bytes() == 0; }
    bool                     not_empty() const { return !empty(); }
    Array<value_type>&       data() { return m_data; }
    const Array<value_type>& data() const { return m_data; }
    const Slice<value_type>  view() const { return m_data.view(); }

    Slice<value_type> substring_bytes(size_type from, size_type to) const { return m_data.view().slice(from, to); }
    Slice<value_type> substring_from_bytes(size_type from) const { return m_data.view().slice_from(from); }
    Slice<value_type> substring_to_bytes(size_type to) const { return m_data.view().slice_to(to); }
    // clang-format off
    String substring_bytes(Allocator* allocator, size_type from, size_type to) const { return String::from_raw(allocator, m_data.view().slice(from, to)); }
    String substring_from_bytes(Allocator* allocator, size_type from) const { return String::from_raw(allocator, m_data.view().slice_from(from)); }
    String substring_to_bytes(Allocator* allocator, size_type to) const { return String::from_raw(allocator, m_data.view().slice_to(to)); }
    // clang-format on

    bool contains_bytes(const String& needle) const { return m_data.view().contains(needle.m_data.view()); }
    bool starts_with_bytes(const String& needle) const { return m_data.view().starts_with(needle.m_data.view()); }
    bool ends_with_bytes(const String& needle) const { return m_data.view().ends_with(needle.m_data.view()); }

    // clang-format off
    [[nodiscard]] Slice<value_type> trim_spaces() const { auto space_chars = Slice<const char>(" \t\n\r").reinterpret_elements_as<u8>(); return m_data.view().trim_left(space_chars).trim_right(space_chars); }
    [[nodiscard]] String            trim_spaces(Allocator* allocator) const { auto space_chars = Slice<const char>(" \t\n\r").reinterpret_elements_as<u8>(); return String::from_raw(allocator, m_data.view().trim_left(space_chars).trim_right(space_chars)); }

    [[nodiscard]] Slice<value_type> trim_left_bytes(String trimmed_chars) const { return m_data.view().trim_left(trimmed_chars.m_data.view()); }
    [[nodiscard]] String            trim_left_bytes(Allocator* allocator, String trimmed_chars) const { return String::from_raw(allocator, m_data.view().trim_left(trimmed_chars.m_data.view())); }

    [[nodiscard]] Slice<value_type> trim_right_bytes(String trimmed_chars) const { return m_data.view().trim_right(trimmed_chars.m_data.view()); }
    [[nodiscard]] String            trim_right_bytes(Allocator* allocator, String trimmed_chars) const { return String::from_raw(allocator, m_data.view().trim_right(trimmed_chars.m_data.view())); }

    [[nodiscard]] Slice<value_type> trim_bytes(String trimmed_chars) const { return m_data.view().trim(trimmed_chars.m_data.view()); }
    [[nodiscard]] String            trim_bytes(Allocator* allocator, String trimmed_chars) const { return String::from_raw(allocator, m_data.view().trim(trimmed_chars.m_data.view())); }
    // clang-format on

    [[nodiscard]] SinglyLinkedList<Slice<value_type>> split_view(Allocator* allocator, value_type c) const
    {
        SinglyLinkedList<Slice<value_type>> res{allocator};
        for (u64 pos = 0, len = len_bytes(); pos < len;)
        {
            Slice<value_type> rem{cast(value_type*) m_data.data() + pos, len - pos};
            i64               index = rem.linear_search(c);
            if (index == -1)
            {
                res.push_back(rem);
                break;
            }
            else
            {
                auto s = rem.take(index);
                res.push_back(s);
                pos += index + 1;
            }
        }
        return res;
    }

    [[nodiscard]] SinglyLinkedList<String> split_owning(Allocator* allocator, value_type c) const
    {
        SinglyLinkedList<String> res{allocator};
        for (u64 pos = 0, len = len_bytes(); pos < len;)
        {
            Slice<value_type> rem{cast(value_type*) m_data.data() + pos, len - pos};
            i64               index = rem.linear_search(c);
            if (index == -1)
            {
                auto s = String::from_raw(allocator, rem);
                res.push_back(std::move(s));
                break;
            }
            else
            {
                auto s = String::from_raw(allocator, rem.take(index));
                res.push_back(std::move(s));
                pos += index + 1;
            }
        }
        return res;
    }

    // iterators
    using iterator       = CodepointIterator;
    using const_iterator = CodepointIterator;

    iterator begin() { return iterator(m_data.begin(), m_data.end()); }
    iterator end() { return iterator(m_data.end(), m_data.end()); }

    const_iterator begin() const { return const_iterator(m_data.begin(), m_data.end()); }
    const_iterator end() const { return const_iterator(m_data.end(), m_data.end()); }

    const_iterator cbegin() const { return begin(); }
    const_iterator cend() const { return end(); }

    friend std::ostream& operator<<(std::ostream& os, const String& str);
};

inline std::ostream& operator<<(std::ostream& os, const String& str)
{
    if (str.len_bytes() > 0)
    {
        os.write(cast(const char*)(str.data().begin()), str.len_bytes());
    }
    return os;
}

#endif