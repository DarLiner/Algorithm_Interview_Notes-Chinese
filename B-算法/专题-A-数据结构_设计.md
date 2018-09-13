专题-数据结构设计
===
- [LeetCode 设计专题](https://leetcode-cn.com/tag/design/)

Index
---
<!-- TOC -->

- [LRU（最近最少使用）缓存机制](#lru最近最少使用缓存机制)

<!-- /TOC -->


## LRU（最近最少使用）缓存机制
> LeetCode-[146. LRU缓存机制](https://leetcode-cn.com/problems/lru-cache/description/)

**思路**
- **双向链表** + haspmap
  - 数据除了被保存在链表中，同时也保存在 map 中，以实现 `O(1)` 的访问
- **更新过程**：
  - 新数据插入到链表头部
  - 每当缓存命中（即缓存数据被访问），则将数据移到链表头部
  - 当链表满的时候，将链表尾部的数据丢弃
- **操作**：
  - `put(key, value)`：如果 key 在 hash_map 中存在，则先**重置对应的 value 值**，然后获取对应的节点，将节点从链表移除，并移动到链表的头部；若果 key 在 hash_map 不存在，则新建一个节点，并将节点放到链表的头部。当 Cache 存满的时候，将链表最后一个节点删除。
  - `get(key)`：如果 key 在 hash_map 中存在，则把对应的节点放到链表头部，并返回对应的value值；如果不存在，则返回-1。

**C++**（AC）
```C++
// 缓存节点（双端队列）
struct CacheNode {
    int key;
    int value;
    CacheNode *pre, *next;
    CacheNode(int k, int v) : key(k), value(v), pre(nullptr), next(nullptr) {}
};

class LRUCache {
    int size = 0;
    CacheNode* head = nullptr;
    CacheNode* tail = nullptr;
    unordered_map<int, CacheNode*> dp;  // hash_map

    void remove(CacheNode *node) {
        if (node != head) {  // 修改后序节点是需判断是否头结点
            node->pre->next = node->next;
        }
        else {
            head = node->next;
        }

        if (node != tail) {  // 修改前序节点是需判断是否尾结点
            node->next->pre = node->pre;
        }
        else {
            tail = node->pre;
        }
        
        // remove 时不销毁该节点
        //delete node;
        //node = nullptr;
    }

    void setHead(CacheNode *node) {
        node->next = head;
        node->pre = nullptr;

        if (head != nullptr) {
            head->pre = node;
        }
        head = node;

        if (tail == nullptr) {
            tail = head;
        }
    }
public:
    LRUCache(int capacity) : size(capacity) { }

    int get(int key) {
        auto it = dp.find(key);
        if (it != dp.end()) {
            auto node = dp[key];

            // 如果命中了，把该节点移动到头部
            remove(node);
            setHead(node);

            return node->value;
        }

        return -1;
    }

    void put(int key, int value) {
        auto it = dp.find(key);
        if (it != dp.end()) {
            auto node = dp[key];

            node->value = value;    // 更新
            remove(node);
            setHead(node);
        }
        else {
            auto node = new CacheNode(key, value);
            setHead(node);
            dp[key] = node;
            
            // 关键：判断容量
            //if (dp.size() >= size) {  // 若先删除节点，则为 >=
            if (dp.size() > size) {     // 若先存入 dp，则为 >
                auto it = dp.find(tail->key);
                remove(tail);

                // 这里才销毁内存（即使不销毁也能过 LeetCode）
                delete it->second;
                it->second = nullptr;

                dp.erase(it);  // 先销毁，在移除
            }
        }
    }
};
```