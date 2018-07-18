**目录**
---
- [算法常识](./算法常识.md)
- [Algorithm_for_Interview](https://github.com/imhuay/Algorithm_for_Interview-Chinese) - 我的 OJ 代码库
  - [必背算法](https://github.com/imhuay/Algorithm_for_Interview-Chinese/tree/master/Algorithm_for_Interview/_必背算法)
<!-- - [常用子函数](./常用子函数.md) -->

**经典问题**
---
<!-- TOC -->

- [Trick](#trick)
  - [敏感空间](#敏感空间)
- [动态规划](#动态规划)
  - [鹰蛋问题 - Egg Dropping](#鹰蛋问题---egg-dropping)
  - [编辑距离 - Edit Distance](#编辑距离---edit-distance)
  - [最长回文子序列/子串 - Longest Palindromic Subsequence/Substring](#最长回文子序列子串---longest-palindromic-subsequencesubstring)
- [贪心](#贪心)
  - [根据身高重建队列 - Queue Reconstruction by Height](#根据身高重建队列---queue-reconstruction-by-height)
- [哈希](#哈希)
  - [局部敏感哈希](#局部敏感哈希)
- [位运算](#位运算)
  - [只出现一次的数字 - Single Number I, II, III](#只出现一次的数字---single-number-i-ii-iii)
- [数据结构](#数据结构)
  - [二叉树](#二叉树)
    - [二叉树的直径 - Diameter of Binary Tree](#二叉树的直径---diameter-of-binary-tree)

<!-- /TOC -->

# Trick
> [百度：ACM trick](https://www.baidu.com/s?wd=ACM%20trick)
1. 如果输入规模 `< 1e4`，那么可以尝试考虑 `O(n^2)` 的算法（两次循环暴力枚举）；如果 `≥ 1e5`，那么可能就需要考虑 `O(nlogn)` 的算法了——这不是绝对的，也可以通过剪枝等操作加速。
1. [C++] string 拼接的速度依次为：1.使用 `stringstream`，2.使用 `append()`，3.使用 `s += c`，4.使用 `s = s + c`——如果有大量拼接操作，要避免使用最后一种方法。

## 敏感空间
> [ACM Trick点&&常用操作记录(持续更新)（敏感空间）](https://blog.csdn.net/feynman1999/article/details/79588347) - CSDN博客 
- `long long` 上界：`2^63-1 = 9,223,372,036,854,775,807 ≈ 9.223372e+18`
  - 阶乘：`20! = 2432902008176640000 ≈ 2.432902e+18` OK，`21!`超
  - Fibnacci 数列：`fib[92] = 7540113804746346429 ≈ 7.540114e+18` OK，`fib[93]` 超
  - Catalan 数：
  - 整数划分：
- **数组大小**：如果内存限制为 2MB，那么最大可开辟的 int 数组为 `2*1024*1024/4 = 524288 ≈ 50,0000`，char 数组为 `2*1024*1024 = 2,097,152 ≈ 200,0000`
  - 实际单个数组是远开不了那么大，比如 windows 默认的栈好像只有 1MB
  - 在无法修改栈大小的情况下，可以使用 `new` 或 `malloc` 在**堆**上开辟内存
    ```C
    int *pa = malloc(sizeof(int)*1024*1024);
    int *pb = new int[1024*1024];
    ```

# 动态规划

## 鹰蛋问题 - Egg Dropping

## 编辑距离 - Edit Distance
- 代码
  - **imhuay**/**Algorithm_for_Interview**/动态规划/[编辑距离.hpp](https://github.com/imhuay/Algorithm_for_Interview-Chinese/blob/master/Algorithm_for_Interview/动态规划/编辑距离.hpp)
- 在线 OJ
  - [编辑距离](https://leetcode-cn.com/problems/edit-distance/description/) - LeetCode (中国) 
- 解析
  - [20ms Detailed Explained C++ Solutions (O(n) Space)](https://leetcode.com/problems/edit-distance/discuss/25846/20ms-Detailed-Explained-C++-Solutions-(O(n)-Space)) - LeetCode 讨论区
  - [两个字符串的编辑距离-动态规划方法](https://blog.csdn.net/ac540101928/article/details/52786435) - CSDN博客

## 最长回文子序列/子串 - Longest Palindromic Subsequence/Substring
- 代码
  - **imhuay**/**Algorithm_for_Interview**/动态规划/[最长回文子序列.hpp](https://github.com/imhuay/Algorithm_for_Interview-Chinese/blob/master/Algorithm_for_Interview/动态规划/最长回文子序列.hpp)
  - **imhuay**/**Algorithm_for_Interview**/动态规划/[最长回文子串.hpp](https://github.com/imhuay/Algorithm_for_Interview-Chinese/blob/master/Algorithm_for_Interview/动态规划/最长回文子串.hpp)
- 在线 OJ
  - [Longest Palindromic Subsequence](https://leetcode.com/problems/undefined/description/) - LeetCode 
  - [Longest Palindromic Substring](https://leetcode.com/problems/undefined/description/) - LeetCode 
- 解析
  - 最长回文子序列/[Straight forward Java DP solution](https://leetcode.com/problems/longest-palindromic-subsequence/discuss/99101/Straight-forward-Java-DP-solution) - LeetCode 讨论区


# 贪心

## 根据身高重建队列 - Queue Reconstruction by Height
- 代码
  - **imhuay**/**Algorithm_for_Interview**/贪心/[根据身高重建队列.hpp](https://github.com/imhuay/Algorithm_for_Interview-Chinese/blob/master/Algorithm_for_Interview/贪心/根据身高重建队列.hpp)
- 在线 OJ
  - [Queue Reconstruction by Height](https://leetcode.com/problems/queue-reconstruction-by-height/) - LeetCode 
- 解析
  - [Easy concept with Python/C++/Java Solution](https://leetcode.com/problems/queue-reconstruction-by-height/discuss/89345/Easy-concept-with-PythonC++Java-Solution) - LeetCode 讨论区
  - [6 lines  Concise C++](https://leetcode.com/problems/queue-reconstruction-by-height/discuss/89348/6-lines-Concise-C++) - LeetCode 讨论区

# 哈希

## 局部敏感哈希


# 位运算

## 只出现一次的数字 - Single Number I, II, III
- 代码
- 在线 OJ
  - [Single Number II](https://leetcode.com/problems/undefined/description/) - LeetCode 
- 解析

# 数据结构

## 二叉树

### 二叉树的直径 - Diameter of Binary Tree
- 代码
  - **imhuay**/**Algorithm_for_Interview**/二叉树/[二叉树的直径.hpp](https://github.com/imhuay/Algorithm_for_Interview-Chinese/blob/master/Algorithm_for_Interview/二叉树/二叉树的直径.hpp)
- 在线 OJ
  - [Diameter of Binary Tree](https://leetcode.com/problems/diameter-of-binary-tree/) - LeetCode
- 解析
  - [C++_Recursive_with brief explanation](https://leetcode.com/problems/diameter-of-binary-tree/discuss/101115/543.-Diameter-of-Binary-Tree-C++_Recursive_with-brief-explanation) - LeetCode 讨论区