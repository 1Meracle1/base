#pragma once
#include <atomic>
#include <vector>
#include <cstddef>
#include <optional>

template<typename T>
class SPMCQueue {
public:
    explicit SPMCQueue(size_t capacity)
        : buffer_(capacity), capacity_(capacity), head_(0), tail_(0) {}

    // Single producer: thread-safe
    bool enqueue(const T& item) {
        size_t tail = tail_.load(std::memory_order_relaxed);
        size_t next_tail = increment(tail);
        if (next_tail == head_.load(std::memory_order_acquire)) {
            return false; // queue full
        }
        buffer_[tail] = item;
        tail_.store(next_tail, std::memory_order_release);
        return true;
    }

    // Multiple consumers: thread-safe
    std::optional<T> dequeue() {
        size_t head;
        do {
            head = head_.load(std::memory_order_acquire);
            if (head == tail_.load(std::memory_order_acquire)) {
                return std::nullopt; // queue empty
            }
        } while (!head_.compare_exchange_weak(
            head, increment(head),
            std::memory_order_acquire, std::memory_order_relaxed));
        return buffer_[head];
    }

    bool empty() const {
        return head_.load(std::memory_order_acquire) == tail_.load(std::memory_order_acquire);
    }

    bool full() const {
        return increment(tail_.load(std::memory_order_acquire)) == head_.load(std::memory_order_acquire);
    }

private:
    size_t increment(size_t idx) const {
        return (idx + 1) % capacity_;
    }

    std::vector<T> buffer_;
    const size_t capacity_;
    std::atomic<size_t> head_;
    std::atomic<size_t> tail_;
};