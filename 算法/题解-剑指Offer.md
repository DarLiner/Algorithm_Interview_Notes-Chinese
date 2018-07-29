**RoadMap**
---

<h3>说明：</h3>

- 主要编程语言为 C/C++
- 涉及**字符串**的问题可能会使用 Python
- 题目编号以原书为准，如“**面试题 3：数组中重复的数字**”
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

<!-- /TOC -->


## 3.1 数组中重复的数字
> [数组中重复的数字](https://www.nowcoder.com/practice/623a5ac0ea5b4e5f95552655361ae0a8?tpId=13&tqId=11203&tPage=3&rp=3&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking) - NowCoder

**题目描述**
```
// 面试题3（一）：找出数组中重复的数字
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
**题目描述**
```
// 面试题3（二）：不修改数组找出重复的数字
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
- 在一个二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。

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
