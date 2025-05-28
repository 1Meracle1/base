#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <atomic>
#include <thread>
#include <vector>
#include <functional>
#include <memory>
#include <deque>
#include <future>

class ThreadPool {
private:
    struct Task {
        std::function<void()> func;
        Task* next;
        Task(std::function<void()>&& f) : func(std::move(f)), next(nullptr) {}
    };

    struct TaggedPtr {
        Task* ptr;
        uintptr_t tag;
    };

    struct WorkStealingQueue {
        std::atomic<Task*> head;
        std::atomic<Task*> tail;
        
        WorkStealingQueue() : head(nullptr), tail(nullptr) {}

        void push(std::function<void()>&& task) {
            Task* new_task = new Task(std::move(task));
            
            Task* old_tail = tail.load(std::memory_order_relaxed);
            while (true) {
                if (tail.compare_exchange_weak(old_tail, new_task,
                                            std::memory_order_release,
                                            std::memory_order_relaxed)) {
                    if (old_tail == nullptr) {
                        head.store(new_task, std::memory_order_release);
                    } else {
                        old_tail->next = new_task;
                    }
                    break;
                }
            }
        }
        
        bool try_pop(std::function<void()>& task) {
            Task* old_head = head.load(std::memory_order_acquire);
            while (old_head != nullptr) {
                Task* new_head = old_head->next;
                if (head.compare_exchange_weak(old_head, new_head,
                                            std::memory_order_release,
                                            std::memory_order_relaxed)) {
                    task = std::move(old_head->func);
                    delete old_head;
                    return true;
                }
            }
            return false;
        }
        
        bool try_steal(std::function<void()>& task) {
            Task* old_tail = tail.load(std::memory_order_acquire);
            if (old_tail == nullptr) return false;
            
            Task* new_tail = nullptr;
            Task* current = head.load(std::memory_order_acquire);
            
            while (current != nullptr && current->next != old_tail) {
                new_tail = current;
                current = current->next;
            }
            
            if (current != nullptr && 
                tail.compare_exchange_strong(old_tail, new_tail,
                                          std::memory_order_release,
                                          std::memory_order_relaxed)) {
                task = std::move(old_tail->func);
                delete old_tail;
                return true;
            }
            
            return false;
        }

        ~WorkStealingQueue() {
            Task* current = head.load(std::memory_order_relaxed);
            while (current != nullptr) {
                Task* next = current->next;
                delete current;
                current = next;
            }
        }
    };

    std::vector<std::unique_ptr<WorkStealingQueue>> queues;
    std::vector<std::thread> threads;
    std::atomic<bool> done{false};
    std::atomic<size_t> index{0};

    void worker(size_t id) {
        while (!done) {
            std::function<void()> task;
            
            // Try to get task from own queue
            if (queues[id]->try_pop(task)) {
                task();
                continue;
            }
            
            // Try to steal from other queues
            bool found_task = false;
            const size_t queue_count = queues.size();
            for (size_t i = 0; i < queue_count; ++i) {
                const size_t idx = (id + i + 1) % queue_count;
                if (queues[idx]->try_steal(task)) {
                    task();
                    found_task = true;
                    break;
                }
            }
            
            if (!found_task) {
                std::this_thread::yield();
            }
        }
    }

public:
    explicit ThreadPool(size_t thread_count = std::thread::hardware_concurrency())
        : queues(thread_count), threads(thread_count) {
        
        for (size_t i = 0; i < thread_count; ++i) {
            queues[i] = std::make_unique<WorkStealingQueue>();
        }
        
        for (size_t i = 0; i < thread_count; ++i) {
            threads[i] = std::thread(&ThreadPool::worker, this, i);
        }
    }

    ~ThreadPool() {
        done = true;
        for (auto& thread : threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }

    template<typename F>
    auto submit(F&& f) -> std::future<typename std::result_of<F()>::type> {
        using result_type = typename std::result_of<F()>::type;
        auto task = std::make_shared<std::packaged_task<result_type()>>(std::forward<F>(f));
        std::future<result_type> res = task->get_future();
        
        const size_t idx = index++ % queues.size();
        queues[idx]->push([task](){ (*task)(); });
        
        return res;
    }

    // Delete copy and move operations
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ThreadPool(ThreadPool&&) = delete;
    ThreadPool& operator=(ThreadPool&&) = delete;
};

#endif // THREAD_POOL_H