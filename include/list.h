#ifndef LIST_H
#define LIST_H

#include "memory.h"
#include "types.h"
#include <type_traits>
#include <utility>

template <typename T> class SinglyLinkedList
{
  public:
    using value_type        = T;
    using iterator_category = std::forward_iterator_tag;
    using difference_type   = std::ptrdiff_t;
    using size_type         = std::size_t;
    using pointer           = value_type*;
    using const_pointer     = const value_type*;
    using reference         = value_type&;
    using const_reference   = std::conditional_t<TrivialSmall<value_type>, value_type, const value_type&>;

    struct Node
    {
        Node* next;
        T     data;
    };

  private:
    Allocator* allocator;
    Node*      head;
    Node*      tail;

  public:
    explicit SinglyLinkedList(Allocator* allocator)
        : allocator{allocator}
        , head{nullptr}
        , tail{nullptr}
    {
    }

    class ConstIterator;

    class Iterator
    {
      public:
        using iterator_category = std::forward_iterator_tag;
        using value_type        = T;
        using difference_type   = std::ptrdiff_t;
        using pointer           = T*;
        using reference         = T&;

        Iterator(Node* p = nullptr)
            : current_node(p)
        {
        }

        reference operator*() const { return current_node->data; }
        pointer   operator->() const { return &(current_node->data); }

        Iterator& operator++()
        {
            if (current_node)
            {
                current_node = current_node->next;
            }
            return *this;
        }

        Iterator operator++(int)
        {
            Iterator temp = *this;
            ++(*this);
            return temp;
        }

        bool operator==(const Iterator& other) const { return current_node == other.current_node; }
        bool operator!=(const Iterator& other) const { return !(*this == other); }

      private:
        Node* current_node;
        friend class SinglyLinkedList<T>;
        friend class SinglyLinkedList<T>::ConstIterator; // For conversion
    };

    class ConstIterator
    {
      public:
        using iterator_category = std::forward_iterator_tag;
        using value_type        = T;
        using difference_type   = std::ptrdiff_t;
        using pointer           = const T*;
        using reference         = const T&;

        ConstIterator(const Node* p = nullptr)
            : current_node(p)
        {
        }
        ConstIterator(const Iterator& other)
            : current_node(other.current_node)
        {
        }

        reference operator*() const { return current_node->data; }
        pointer   operator->() const { return &(current_node->data); }

        ConstIterator& operator++()
        {
            if (current_node)
            {
                current_node = current_node->next;
            }
            return *this;
        }

        ConstIterator operator++(int)
        {
            ConstIterator temp = *this;
            ++(*this);
            return temp;
        }

        bool operator==(const ConstIterator& other) const { return current_node == other.current_node; }
        bool operator!=(const ConstIterator& other) const { return !(*this == other); }

      private:
        const Node* current_node;
        friend class SinglyLinkedList<T>;
    };

    SinglyLinkedList clone()
    {
        SinglyLinkedList res{allocator};
        if (empty())
        {
            return;
        }
        Node* current_other = head;
        while (current_other != nullptr)
        {
            res.push_back(current_other->data);
            current_other = current_other->next;
        }
    }

    SinglyLinkedList()
        : head(nullptr)
        , tail(nullptr)
    {
    }

    SinglyLinkedList(const SinglyLinkedList& other)               = delete;
    SinglyLinkedList<T>& operator=(const SinglyLinkedList& other) = delete;

    SinglyLinkedList(SinglyLinkedList&& other) noexcept
        : allocator(std::exchange(other.allocator, nullptr))
        , head(std::exchange(other.head, nullptr))
        , tail(std::exchange(other.tail, nullptr))
    {
    }

    ~SinglyLinkedList()
    {
        if (allocator != nullptr)
            clear();
    }

    SinglyLinkedList<T>& operator=(SinglyLinkedList&& other) noexcept
    {
        if (this != &other)
        {
            clear();
            allocator = std::exchange(other.allocator, nullptr);
            head      = std::exchange(other.head, nullptr);
            tail      = std::exchange(other.tail, nullptr);
        }
        return *this;
    }

    reference front()
    {
        Assert(!empty());
        return head->data;
    }

    const_reference front() const
    {
        Assert(!empty());
        return head->data;
    }

    reference back()
    {
        Assert(!empty());
        return tail->data;
    }

    const_reference back() const
    {
        Assert(!empty());
        return tail->data;
    }

    Iterator begin() { return Iterator(head); }
    Iterator end() { return Iterator(nullptr); }

    ConstIterator begin() const { return ConstIterator(head); }
    ConstIterator end() const { return ConstIterator(nullptr); }

    ConstIterator cbegin() const { return ConstIterator(head); }
    ConstIterator cend() const { return ConstIterator(nullptr); }

    bool empty() const noexcept { return head == nullptr; }
    bool not_empty() const noexcept { return head != nullptr; }

    size_type length() const noexcept
    {
        size_type res = 0;
        for (auto it = begin(), itEnd = end(); it != itEnd; ++it)
            ++res;
        return res;
    }

    void clear() noexcept
    {
        Node* current = head;
        while (current != nullptr)
        {
            Node* next_node = current->next;
            allocator->free(current);
            current = next_node;
        }
        head = nullptr;
        tail = nullptr;
    }

    [[nodiscard]] reference push_front()
    {
        Node* new_node = allocator->alloc<Node>();
        new_node->next = head;
        head           = new_node;
        if (tail == nullptr)
        {
            tail = head;
        }
        return head->data;
    }

    void push_front(const_reference value)
    {
        Node* new_node = allocator->alloc<Node>();
        new_node->data = value;
        new_node->next = head;
        head           = new_node;
        if (tail == nullptr)
        {
            tail = head;
        }
    }

    void push_front(value_type&& value)
    {
        Node* new_node = allocator->alloc<Node>();
        new_node->data = std::move(value);
        new_node->next = head;
        head           = new_node;
        if (tail == nullptr)
        {
            tail = head;
        }
    }

    void pop_front()
    {
        Assert(not_empty());
        Node* old_head = head;
        head           = head->next;
        allocator->free(old_head);
        if (head == nullptr)
        {
            tail = nullptr;
        }
    }

    [[nodiscard]] reference push_back()
    {
        Node* new_node = allocator->alloc<Node>();
        if (empty())
        {
            head = new_node;
            tail = new_node;
        }
        else
        {
            tail->next = new_node;
            tail       = new_node;
        }
        return head->data;
    }

    void push_back(const_reference value)
        requires(TrivialSmall<value_type> || std::is_copy_assignable_v<value_type>)
    {
        Node* new_node = allocator->alloc<Node>();
        new_node->data = value;
        if (empty())
        {
            head = new_node;
            tail = new_node;
        }
        else
        {
            tail->next = new_node;
            tail       = new_node;
        }
    }

    void push_back(value_type&& value)
        requires(!TrivialSmall<value_type> && std::is_move_assignable_v<value_type>)
    {
        Node* new_node = allocator->alloc<Node>();
        new_node->data = std::move(value);
        if (empty())
        {
            head = new_node;
            tail = new_node;
        }
        else
        {
            tail->next = new_node;
            tail       = new_node;
        }
    }

    void pop_back()
    {
        Assert(not_empty());
        if (head == tail)
        {
            allocator->free(head);
            head = nullptr;
            tail = nullptr;
        }
        else
        {
            Node* current = head;
            while (current->next != tail)
            {
                current = current->next;
            }
            allocator->free(tail);
            tail       = current;
            tail->next = nullptr;
        }
    }

    Iterator insert_after(ConstIterator pos, const_reference value)
    {
        Node* prev_node = const_cast<Node*>(pos.current_node);
        Assert(prev_node != nullptr);
        Node* new_node  = allocator->alloc<Node>();
        new_node->data  = value;
        new_node->next  = prev_node->next;
        prev_node->next = new_node;
        if (prev_node == tail)
        {
            tail = new_node;
        }
        return Iterator(new_node);
    }

    Iterator insert_after(ConstIterator pos, value_type&& value)
    {
        Node* prev_node = const_cast<Node*>(pos.current_node);
        Assert(prev_node != nullptr);
        Node* new_node  = allocator->alloc<Node>();
        new_node->data  = std::move(value);
        new_node->next  = prev_node->next;
        prev_node->next = new_node;
        if (prev_node == tail)
        {
            tail = new_node;
        }
        return Iterator(new_node);
    }

    Iterator erase_after(ConstIterator pos)
    {
        Node* prev_node = const_cast<Node*>(pos.current_node);
        Assert(prev_node != nullptr && prev_node->next != nullptr);
        Node* node_to_erase = prev_node->next;
        prev_node->next     = node_to_erase->next;
        if (node_to_erase == tail)
        {
            tail = prev_node;
        }
        allocator->free(node_to_erase);
        return Iterator(prev_node->next);
    }

    Iterator insert(ConstIterator pos, const_reference value)
    {
        if (pos.current_node == head)
        {
            push_front(value);
            return begin();
        }
        if (pos.current_node == nullptr)
        {
            push_back(value);
            return Iterator(tail);
        }

        Node* prev = head;
        while (prev != nullptr && prev->next != pos.current_node)
        {
            prev = prev->next;
        }
        Assert(prev != nullptr);

        Node* new_node = allocator->alloc<Node>();
        new_node->data = value;
        new_node->next = prev->next;
        prev->next     = new_node;
        return Iterator(new_node);
    }

    Iterator insert(ConstIterator pos, value_type&& value)
    {
        if (pos.current_node == head)
        {
            push_front(std::move(value));
            return begin();
        }
        if (pos.current_node == nullptr)
        {
            push_back(std::move(value));
            return Iterator(tail);
        }

        Node* prev = head;
        while (prev != nullptr && prev->next != pos.current_node)
        {
            prev = prev->next;
        }
        Assert(prev != nullptr);

        Node* new_node = allocator->alloc<Node>();
        new_node->data = std::move(value);
        new_node->next = prev->next;
        prev->next     = new_node;
        return Iterator(new_node);
    }

    Iterator erase(ConstIterator pos)
    {
        Assert(pos != cend() && not_empty());
        Node* node_to_erase = const_cast<Node*>(pos.current_node);
        if (node_to_erase == head)
        {
            pop_front();
            return begin();
        }

        Node* prev = head;
        while (prev != nullptr && prev->next != node_to_erase)
        {
            prev = prev->next;
        }
        Assert(prev != nullptr);

        prev->next = node_to_erase->next;
        if (node_to_erase == tail)
        {
            tail = prev;
        }
        allocator->free(node_to_erase);
        return Iterator(prev->next);
    }
};

#endif