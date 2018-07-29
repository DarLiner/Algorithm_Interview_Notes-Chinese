**RoadMap**
---

<h3>说明：</h3>

- 主要编程语言为 C/C++
- 涉及**字符串**的问题可能会使用 Python
- 题目编号以原书为准，如“**面试题 3：数组中重复的数字**”
  - 因为题目不多，所以就不做分类了
- 所有代码均通过 OJ 测试
    > 在线 **OJ 地址**：[剑指Offer_编程题](https://www.nowcoder.com/ta/coding-interviews) - 牛客网 

<h3>Reference</h3>

- 《剑指 Offer（第二版）》 - 何海涛
- Interview-Notebook/[剑指 offer 题解.md](https://github.com/CyC2018/Interview-Notebook/blob/master/notes/%E5%89%91%E6%8C%87%20offer%20%E9%A2%98%E8%A7%A3.md) · CyC2018/Interview-Notebook
- 牛客网相关问题讨论区


**Index**
---
<!-- TOC -->

- [3.1 数组中重复的数字](#31-数组中重复的数字)
- [3.2 不修改数组找出重复的数字](#32-不修改数组找出重复的数字)
- [4. 二维数组中的查找](#4-二维数组中的查找)
- [5. 替换空格](#5-替换空格)
- [6. 从尾到头打印链表](#6-从尾到头打印链表)
- [7. 重建二叉树](#7-重建二叉树)
- [8. 二叉树的下一个结点](#8-二叉树的下一个结点)
- [9. 用两个栈实现队列](#9-用两个栈实现队列)

<!-- /TOC -->


## 3.1 数组中重复的数字
> [数组中重复的数字](https://www.nowcoder.com/practice/623a5ac0ea5b4e5f95552655361ae0a8?tpId=13&tqId=11203&tPage=3&rp=3&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking) - NowCoder

**题目描述**
```// 面试题3（一）：找出数组中重复的数字
// 题目：在一个长度为n的数组里的所有数字都在0到n-1的范围内。数组中某些数字是重复的，但不知道有几个数字重复了，
// 也不知道每个数字重复了几次。请找出数组中任意一个重复的数字。例如，如果输入长度为7的数组{2, 3, 1, 0, 2, 5, 3}，
// 那么对应的输出是重复的数字2或者3。
```
- 要求：时间复杂度`O(N)`，空间复杂度`O(1)`
- 示例
  ```
  Input:
  {2, 3, 1, 0, 2, 5}

  Output:
  2
  ```

**思路**
- 复杂度要求表明**不能使用排序**，也不能使用 **map**/**set**
- 注意到 n 个数字的范围为 `0` 到 `n-1`，考虑类似**选择排序**的思路，通过一次遍历将每个数交换到排序后的位置，如果该位置已经存在相同的数字，那么该数就是重复的
- 示例
  ```
  position-0 : (2,3,1,0,2,5) // 2 <-> 1
               (1,3,2,0,2,5) // 1 <-> 3
               (3,1,2,0,2,5) // 3 <-> 0
               (0,1,2,3,2,5) // already in position
  position-1 : (0,1,2,3,2,5) // already in position
  position-2 : (0,1,2,3,2,5) // already in position
  position-3 : (0,1,2,3,2,5) // already in position
  position-4 : (0,1,2,3,2,5) // nums[i] == nums[nums[i]], exit
  ```
**Code**
```C++
class Solution {
public:
    bool duplicate(int numbers[], int length, int* duplication) {
        if(numbers == nullptr || length <= 0)
            return false;

        for(int i = 0; i < length; ++i) {
            while(numbers[i] != i) {
                if(numbers[i] == numbers[numbers[i]]) {
                    *duplication = numbers[i];
                    return true;
                }
                // 交换numbers[i]和numbers[numbers[i]]
                swap(numbers[i], numbers[numbers[i]]);
            }
        }
        return false;
    }
};
```


## 3.2 不修改数组找出重复的数字
> 未加入牛客网 OJ

**题目描述**
```// 面试题3（二）：不修改数组找出重复的数字
// 题目：在一个长度为n+1的数组里的所有数字都在1到n的范围内，所以数组中至
// 少有一个数字是重复的。请找出数组中任意一个重复的数字，但不能修改输入的
// 数组。例如，如果输入长度为8的数组{2, 3, 5, 4, 3, 2, 6, 7}，那么对应的
// 输出是重复的数字2或者3。
```
- 要求：时间复杂度`O(NlogN)`，空间复杂度`O(1)`

**思路**
- 二分查找
- 以长度为 8 的数组 `{2, 3, 5, 4, 3, 2, 6, 7}` 为例，那么所有数字都在 `1~7` 的范围内。中间的数字 `4` 将 `1~7` 分为 `1~4` 和 `5~7`。统计 `1~4` 内数字的出现次数，它们一共出现了 5 次，说明 `1~4` 内必要重复的数字；反之，若小于等于 4 次，则说明 `5~7` 内必有重复的数字。
- 因为不能使用额外的空间，所以每次统计次数都要重新遍历整个数组一次

**Code**
```C++
int countRange(const int* numbers, int length, int start, int end);

int getDuplication(const int* numbers, int length)
{
    if(numbers == nullptr || length <= 0)
        return -1;

    int start = 1;
    int end = length - 1;
    while(end >= start) {
        int middle = ((end - start) >> 1) + start;
        int count = countRange(numbers, length, start, middle);
        if(end == start) {
            if(count > 1)
                return start;
            else
                break;
        }

        if(count > (middle - start + 1))
            end = middle;
        else
            start = middle + 1;
    }
    return -1;
}

// 因为不能使用额外的空间，所以每次统计次数都要重新遍历整个数组一次
int countRange(const int* numbers, int length, int start, int end) {
    if(numbers == nullptr)
        return 0;

    int count = 0;
    for(int i = 0; i < length; i++)
        if(numbers[i] >= start && numbers[i] <= end)
            ++count;
    return count;
}
```


## 4. 二维数组中的查找
> [二维数组中的查找](https://www.nowcoder.com/practice/abc3fe2ce8e146608e868a70efebf62e?tpId=13&tqId=11154&tPage=1&rp=1&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking) - NowCoder

**题目描述**
```// 面试题4：二维数组中的查找
// 题目：在一个二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按
// 照从上到下递增的顺序排序。请完成一个函数，输入这样的一个二维数组和一个
// 整数，判断数组中是否含有该整数。
```
- 示例
  ```html
  Consider the following matrix:
  [
    [1,   4,  7, 11, 15],
    [2,   5,  8, 12, 19],
    [3,   6,  9, 16, 22],
    [10, 13, 14, 17, 24],
    [18, 21, 23, 26, 30]
  ]

  Given target = 5, return true.
  Given target = 20, return false.
  ```

**思路**
- 从**左下角**开始查找，它左边的数都比它小，下边的数都比它大；因此可以根据 target 和当前元素的大小关系来**缩小查找区间**
- 同理，也可以从**右上角**开始查找
- 时间复杂度：`O(M + N)`

**Code**
```C++
class Solution {
public:
    bool Find(int target, vector<vector<int> > array) {
        int N = array.size();          // 行数
        int M = array[0].size();       // 列数
            
        int i = N - 1;
        int j = 0;
        while (i >= 0 && j < M) {
            if (array[i][j] > target)
                i--;
            else if (array[i][j] < target)
                j++;
            else
                return true;
        }
        return false;
    }
};
```


## 5. 替换空格
> [替换空格](https://www.nowcoder.com/practice/4060ac7e3e404ad1a894ef3e17650423?tpId=13&tqId=11155&tPage=1&rp=1&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking) - NowCoder 

**题目描述**
```// 面试题5：替换空格
// 题目：请实现一个函数，把字符串中的每个空格替换成"%20"。例如输入“We are happy.”，
// 则输出“We%20are%20happy.”。
```

**思路**
- 先遍历一次，找出空格的数量，得到替换后的长度；然后从后往前替换

**Code**
```C++
class Solution {
public:
    void replaceSpace(char *str, int length) {
        if (str == nullptr || length < 0)
            return;

        int l_old = strlen(str);  // == length
        int n_space = count(str, str + l_old, ' ');  // <algorithm>
        int l_new = l_old + n_space * 2;
        str[l_new] = '\0';

        int p_old = l_old-1;
        int p_new = l_new-1;
        while (p_old >= 0) {
            if (str[p_old] != ' ') {
                str[p_new--] = str[p_old--];
            }
            else {
                p_old--;
                str[p_new--] = '0';
                str[p_new--] = '2';
                str[p_new--] = '%';
            }
        }
    }
};
```


## 6. 从尾到头打印链表
> [从尾到头打印链表](https://www.nowcoder.com/practice/d0267f7f55b3412ba93bd35cfa8e8035?tpId=13&tqId=11156&tPage=1&rp=1&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking) - NowCoder

**题目描述**
```// 面试题6：从尾到头打印链表
// 题目：输入一个链表的头结点，从尾到头反过来打印出每个结点的值。
```

**思路**
- 栈
- 头插法

**Code**
```C++
class Solution {
public:
    vector<int> printListFromTailToHead(ListNode* head) {
        vector<int> ret;

        ListNode *p = head;
        while (p != NULL) {
            ret.insert(ret.begin(), p->val);   // 头插
            p = p->next;
        }

        return ret;
    }
};
```


## 7. 重建二叉树
> [重建二叉树](https://www.nowcoder.com/practice/8a19cbe657394eeaac2f6ea9b0f6fcf6?tpId=13&tqId=11157&tPage=1&rp=1&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking) - NowCoder

**题目描述**
```// 面试题7：重建二叉树
// 题目：输入某二叉树的前序遍历和中序遍历的结果，请重建出该二叉树。
// 假设输入的前序遍历和中序遍历的结果中都不含重复的数字。输出它的头结点。
```

**思路**
- 涉及二叉树的问题，应该条件反射般的使用**递归**（无优化要求时）
- 前序遍历的第一个值为根节点的值，使用这个值将中序遍历结果分成两部分，左部分为左子树的中序遍历结果，右部分为右子树的中序遍历的结果。

**Code 1 - 直观无优化**
```C++
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

class Solution {
public:
    TreeNode * reConstructBinaryTree(vector<int> pre, vector<int> vin) {
        if (pre.size() <= 0)
            return NULL;

        TreeNode* root = new TreeNode{ pre[0] };
        for (auto i = 0; i < vin.size(); i++) {
            if (vin[i] == pre[0]) {
                root->left = reConstructBinaryTree(vector<int>(pre.begin() + 1, pre.begin() + 1 + i), vector<int>(vin.begin(), vin.begin() + i));
                root->right = reConstructBinaryTree(vector<int>(pre.begin() + 1 + i, pre.end()), vector<int>(vin.begin() + 1 + i, vin.end()));
            }
        }
        return root;
    }
};
```

**Code 2 - 优化版**
```C++
class Solution {
public:
    TreeNode * reConstructBinaryTree(vector<int> pre, vector<int> vin) {
        return reConstructCore(pre, 0, pre.size(), vin, 0, vin.size());
    }
    
    TreeNode * reConstructCore(vector<int> &pre, int pre_beg, int pre_end, vector<int> &vin, int vin_beg, int vin_end) {
        if (pre_end - pre_beg <= 0)
            return NULL;

        TreeNode* root = new TreeNode{ pre[pre_beg] };
        for (auto i = 0; i < vin_end-vin_beg; i++) {
            if (vin[i+vin_beg] == pre[pre_beg]) {
                root->left = reConstructCore(pre, pre_beg+1, pre_beg+1+i, vin, vin_beg, vin_beg+i);
                root->right = reConstructCore(pre, pre_beg+1+i, pre_end, vin, vin_beg+1+i, vin_end);
            }
        }
        return root;
    }
};
```


## 8. 二叉树的下一个结点
> [二叉树的下一个结点](https://www.nowcoder.com/practice/9023a0c988684a53960365b889ceaf5e?tpId=13&tqId=11210&tPage=3&rp=1&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking) - NowCoder

**题目描述**
```// 面试题8：二叉树的下一个结点
// 题目：给定一棵二叉树和其中的一个结点，如何找出中序遍历顺序的下一个结点？
// 树中的结点除了有两个分别指向左右子结点的指针以外，还有一个指向父结点的指针。
```

**思路**
- 回顾中序遍历的顺序
- 如果一个节点的右子树不为空，那么下一个节点是该节点右子树的最左叶子；
- 否则（右子树为空），沿父节点向上直到找到某个节点是其父节点的左孩子，那么该父节点就是下一个节点

**Code**
```C++
struct TreeLinkNode {
    int val;
    struct TreeLinkNode *left;
    struct TreeLinkNode *right;
    struct TreeLinkNode *next;
    TreeLinkNode(int x) :val(x), left(NULL), right(NULL), next(NULL) {
    }
};

class Solution {
public:
    TreeLinkNode * GetNext(TreeLinkNode* pNode) {
        if (pNode == nullptr)
            return nullptr;
        
        if(pNode->right != nullptr) {
            auto p = pNode->right;
            while(p->left != nullptr)
                p = p->left;
            return p;
        }
        else {
            auto p = pNode;               // 当前节点
            while(p->next != nullptr) {   // 当前节点的父节点不为空
                if (p->next->left == p)   // 当前节点是其父节点的左海子
                    return p->next;       // 那么下一个节点就是当前节点的父节点
                p = p->next;
            }
        }
        return nullptr;  // 当前节点是根节点且没有右孩子，即没有下一个节点
    }
};
```

## 9. 用两个栈实现队列
> [用两个栈实现队列](https://www.nowcoder.com/practice/54275ddae22f475981afa2244dd448c6?tpId=13&tqId=11158&tPage=1&rp=3&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking) - NowCoder

**题目描述**
```// 面试题9：用两个栈实现队列
// 题目：用两个栈实现一个队列。队列的声明如下，请实现它的两个函数appendTail
// 和deleteHead，分别完成在队列尾部插入结点和在队列头部删除结点的功能。
```

**思路**
- 假设 `stack_in` 用于处理入栈操作，`stack_out`用于处理出栈操作
- `stack_in` 按栈的方式正常处理入栈数据；
- 关键在于出栈操作
  - 当`stack_out`为空时，需要先将每个`stack_in`中的数据出栈后压入`stack_out`
  - 反之，每次弹出`stack_out`栈顶元素即可

**Code**
```C++
class Solution
{
public:
    void push(int node) {
        stack_in.push(node);
    }

    int pop() {
        if(stack_out.size() <= 0) {
            while (stack_in.size() > 0) {
                auto tmp = stack_in.top();
                stack_in.pop();
                stack_out.push(tmp);
            }
        }
        
        auto ret = stack_out.top();
        stack_out.pop();
        return ret;
    }

private:
    stack<int> stack_in;
    stack<int> stack_out;
};
```