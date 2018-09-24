专题-双指针
===
- 双指针问题出现的频率非常高
- 九章算法称之为“为面试而生的双指针算法”

小结
---
- 首尾双指针模板
    - 一般用于寻找数组中满足条件的**两个数**；如果是寻找多个数，则先固定前 n-2 个数
    - 为了不遗漏所有可能情况，一般要求数组**有序**；
    - 遍历时，大于目标时 `hi--`，小于目标时 `lo++`。 
- 同向双指针模板
    - 一般用于寻找满足某个条件的**连续区间**
<!-- - 分离双指针
    - 输入是两个数组/链表，两个指针分别在两个容器中移动 -->

Index
---
<!-- TOC -->

- [首尾双指针](#首尾双指针)
    - [两数之和](#两数之和)
    - [三数之和](#三数之和)
    - [N 数之和](#n-数之和)
    - [最接近的三数之和](#最接近的三数之和)
    - [两数之和 - 小于等于目标值的个数](#两数之和---小于等于目标值的个数)
    - [三数之和 - 小于等于目标值的个数](#三数之和---小于等于目标值的个数)
    - [三角形计数](#三角形计数)
- [同向双指针](#同向双指针)
    - [最小覆盖子串（Minimum Window Substring）](#最小覆盖子串minimum-window-substring)
    - [长度最小的子数组（Minimum Size Subarray Sum）](#长度最小的子数组minimum-size-subarray-sum)
- [其他](#其他)
    - [合并两个有序数组（Merge Sorted Array）](#合并两个有序数组merge-sorted-array)
    - [接雨水（Trapping Rain Water）（一维）](#接雨水trapping-rain-water一维)
    - [颜色分类（Sort Colors）](#颜色分类sort-colors)
    - [两个数组的交集（Intersection of Two Arrays）](#两个数组的交集intersection-of-two-arrays)

<!-- /TOC -->

# 首尾双指针

## 两数之和
> LeetCode/[167. 两数之和 II - 输入有序数组](https://leetcode-cn.com/problems/two-sum-ii-input-array-is-sorted/description/)

**问题描述**（167. 两数之和 II - 输入有序数组）
```python
给定一个已按照升序排列 的有序数组，找到两个数使得它们相加之和等于目标数。

函数应该返回这两个下标值 index1 和 index2，其中 index1 必须小于 index2。

说明:
    返回的下标值（index1 和 index2）不是从零开始的。
    你可以假设每个输入只对应唯一的答案，而且你不可以重复使用相同的元素。

示例:
    输入: numbers = [2, 7, 11, 15], target = 9
    输出: [1,2]
    解释: 2 与 7 之和等于目标数 9 。因此 index1 = 1, index2 = 2 。
```

**拓展**
- 如果存在多个答案，并要求输出所有不重复的可能
    > [三数之和](#三数之和)

**思路 1**
- 首尾双指针
- 因为是有序的，可以尝试使用首尾双指针解决该问题，时间复杂度为 `O(N)`
- **Python**（双指针）
    ```python
    class Solution:
        def twoSum(self, A, t):
            """
            :type A: List[int]
            :type t: int
            :rtype: List[int]
            """
            n = len(A)
            lo, hi = 0, n - 1
            
            ret = []
            while lo < hi:
                s = A[lo] + A[hi]
                if s > t:
                    hi -= 1
                elif s < t:
                    lo += 1
                else:
                    ret.append(lo + 1)
                    ret.append(hi + 1)
                    break
            
            return ret
    ```

**思路 2**
- 本题还可以利用 Hash 表解决，时间复杂度 `O(N)`，空间复杂度 `O(N)`
- 使用 Hash 表不要求数组有序
    > LeetCode/[1. 两数之和](https://leetcode-cn.com/problems/two-sum/description/)
- **Python**（Hash）
    ```python
    class Solution:
        def twoSum(self, A, t):
            """
            :type A: List[int]
            :type t: int
            :rtype: List[int]
            """
            
            d = dict()
            
            for i in range(len(A)):
                if A[i] not in d:
                    d[t - A[i]] = i
                else:
                    return [i, d[A[i]]]
    ```


## 三数之和
> LeetCode/[15. 三数之和](https://leetcode-cn.com/problems/3sum/description/)

**问题描述**
```
给定一个包含 n 个整数的数组 nums，
判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？
找出所有满足条件且不重复的三元组。

注意：答案中不可以包含重复的三元组。

例如, 给定数组 nums = [-1, 0, 1, 2, -1, -4]，

满足要求的三元组集合为：
[
  [-1, 0, 1],
  [-1, -1, 2]
]
```

**思路**
- 排序 + 首尾双指针
- 将第三个数当作前两个数的目标和，在两数之和的基础上套一层循环
- 难点在于如何**去重**（不借用 set）

**python**
```python
class Solution:
    def threeSum(self, A):
        """
        :type A: List[int]
        :rtype: List[List[int]]
        """
        
        A.sort()
        n = len(A)
        
        ret = []
        for i in range(n - 2):
            # 去重时注意判断条件
            if i > 0 and A[i] == A[i - 1]:  # 对第一个数去重
                continue
            
            t = -A[i]
            lo, hi = i + 1, n - 1
            
            while lo < hi:
                s = A[lo] + A[hi]
                
                if s < t:
                    lo += 1
                elif s > t:
                    hi -= 1
                else:
                    ret.append([A[i], A[lo], A[hi]])
                    
                    # 先移动指针再去重
                    lo += 1
                    # hi -= 1  # 不必要
                    
                    # 去重时注意判断条件
                    while lo < hi and A[lo] == A[lo - 1]:  # 对第二个数去重
                        lo += 1
                    #  while lo < hi and A[hi] == A[hi + 1]:  # 对第三个数去重（不必要）
                    #      hi -= 1
        
        return ret
```


## N 数之和
> LeetCode/[18. 四数之和](https://leetcode-cn.com/problems/4sum/description/)

**题目描述**（四数之和）
```
给定一个包含 n 个整数的数组 nums 和一个目标值 target，
判断 nums 中是否存在四个元素 a，b，c 和 d ，使得 a + b + c + d 的值与 target 相等？
找出所有满足条件且不重复的四元组。

注意：

答案中不可以包含重复的四元组。

示例：

给定数组 nums = [1, 0, -1, 0, -2, 2]，和 target = 0。

满足要求的四元组集合为：
[
  [-1,  0, 0, 1],
  [-2, -1, 1, 2],
  [-2,  0, 0, 2]
]
```

**Python**（N 数之和）
```python
def nSum(A, N, t, tmp, ret):
    if len(A) < N or N < 2 or t < A[0] * N or t > A[-1] * N:  # 结束条件
        return

    if N == 2:
        lo, hi = 0, len(A) - 1
        while lo < hi:
            s = A[lo] + A[hi]

            if s < t:
                lo += 1
            elif s > t:
                hi -= 1
            else:
                ret.append(tmp + [A[lo], A[hi]])
                lo += 1
                while lo < hi and A[lo] == A[lo - 1]:  # 去重
                    lo += 1
    else:
        for i in range(len(A) - N + 1):
            if i > 0 and A[i] == A[i - 1]:  # 去重
                continue

            nSum(A[i+1:], N-1, t-A[i], tmp + [A[i]], ret)
            
            
class Solution:
            
    def fourSum(self, A, t):
        """
        :type A: List[int]
        :type t: int
        :rtype: List[List[int]]
        """
        
        A.sort()
        ret = []
        
        nSum(A, 4, t, [], ret)
        
        return ret
```


## 最接近的三数之和
> LeetCode/[16. 最接近的三数之和](https://leetcode-cn.com/problems/3sum-closest/description/)

**问题描述**
```
给定一个包括 n 个整数的数组 nums 和 一个目标值 target。
找出 nums 中的三个整数，使得它们的和与 target 最接近。
返回这三个数的和。假定每组输入只存在唯一答案。

例如，给定数组 nums = [-1，2，1，-4], 和 target = 1.

与 target 最接近的三个数的和为 2. (-1 + 2 + 1 = 2).
```

**思路**
- 排序 + 双指针

**Python**
```python
class Solution:
    def threeSumClosest(self, A, t):
        """
        :type A: List[int]
        :type t: int
        :rtype: int
        """
        A.sort()  # 先排序
        n = len(A)
        
        ans = A[0] + A[1] + A[2]  # 用一个特殊的值初始化
        for i in range(n-2):
            
            lo, hi = i + 1, n - 1  # 首尾指针
            while lo < hi:
                s = A[i] + A[lo] + A[hi]
                if abs(s - t) < abs(ans - t):
                    ans = s
                    if ans == t:
                        return ans
                
                if s < t:
                    lo += 1
                else:
                    hi -= 1
        
        return ans
```


## 两数之和 - 小于等于目标值的个数
> LintCode/[609. 两数和-小于或等于目标值](https://www.lintcode.com/problem/two-sum-less-than-or-equal-to-target)
>> 此为收费问题：[LintCode 练习-609. 两数和-小于或等于目标值](https://blog.csdn.net/qq_36387683/article/details/81460276) - CSDN博客 

**问题描述**
```
给定一个整数数组，找出这个数组中有多少对的和是小于或等于目标值。返回对数。

样例

    给定数组为 [2,7,11,15]，目标值为 24
    返回 5。
    2+7<24
    2+11<24
    2+15<24
    7+11<24
    7+15<24
```

**思路**：
- 排序 + 首尾双指针

**python**
```python
class Solution:

    def twoSum5(self, A, t):
        """
        :type A: List[int]
        :type t: int
        :rtype: List[int]
        """
        n = len(A)
        lo, hi = 0, n - 1
        A.sort()  # 如果是首尾双指针，一般要求有序

        cnt = 0
        while lo < hi:
            s = A[lo] + A[hi]
            if s <= t:
                cnt += hi-lo
                lo += 1
            else:
                hi -= 1

        return cnt
```

**代码说明**
- 以 `[2,7,11,15]` 为例
```
第一轮：lo = 0, hi = 3
    s = A[lo] + A[hi] = 17 <= 24
    因此 [2, 15], [2, 11], [2, 7] 均满足，hi - lo = 3
    lo += 1
第二轮：lo = 1, hi = 3
    s = A[lo] + A[hi] = 22 <= 24
    因此 [7, 15], [7, 11] 均满足，hi - lo = 2
    lo += 1
第三轮：lo = 2, hi = 3
    s = A[lo] + A[hi] = 26 > 24
    hi -= 1
不满足 lo < hi，退出，因此总数量为 3 + 2 = 5
```

**时间复杂度**
- `O(NlogN) + O(N) = O(NlogN)`


## 三数之和 - 小于等于目标值的个数
> LintCode/[918. 三数之和](https://www.lintcode.com/problem/3sum-smaller/description)

**问题描述**
```
给定一个n个整数的数组和一个目标整数target，
找到下标为i、j、k的数组元素0 <= i < j < k < n，满足条件nums[i] + nums[j] + nums[k] < target.

样例
给定 nums = [-2,0,1,3], target = 2, 返回 2.

解释:
    因为有两种三个元素之和，它们的和小于2:
    [-2, 0, 1]
    [-2, 0, 3]
```

**思路**
- 排序 + 双指针

**Python**
```python
class Solution:

    def threeSumSmaller(self, A, t):
        """
        :type A: List[int]
        :type t: int
        :rtype: List[int]
        """
        A.sort()
        n = len(A)
        
        cnt = 0
        for i in range(n - 2):
            lo, hi = i + 1, n - 1
            
            while lo < hi:
                s = A[i] + A[lo] + A[hi]
                
                if s < t:
                    cnt += hi - lo
                    lo += 1
                else:
                    hi -= 1
            
        return cnt
```


## 三角形计数
> LeetCode/[611. 有效三角形的个数](https://leetcode-cn.com/problems/valid-triangle-number/description/)

**问题描述**
```
给定一个包含非负整数的数组，你的任务是统计其中可以组成三角形三条边的三元组个数。

示例 1:
    输入: [2,2,3,4]
    输出: 3
解释:
    有效的组合是: 
    2,3,4 (使用第一个 2)
    2,3,4 (使用第二个 2)
    2,2,3
注意:
    数组长度不超过1000。
    数组里整数的范围为 [0, 1000]。
```

**思路**
- 排序 + 首尾双指针
- 相当于两数之和大于目标值的个数

**Python**
```python
class Solution:
    def triangleNumber(self, A):
        """
        :type A: List[int]
        :rtype: int
        """
        A.sort()
        n = len(A)
        
        cnt = 0
        for i in range(2, n):  # 注意：循环区间
            
            lo, hi = 0, i - 1
            while lo < hi:
                s = A[lo] + A[hi]
                
                if s > A[i]:
                    cnt += hi - lo
                    hi -= 1
                else:
                    lo += 1
                    
        return cnt
```


# 同向双指针

## 最小覆盖子串（Minimum Window Substring）
> LeetCode/76. 最小覆盖子串

**问题描述**
```
给定一个字符串 S 和一个字符串 T，请在 S 中找出包含 T 所有字母的最小子串。

示例：
    输入: S = "ADOBECODEBANC", T = "ABC"
    输出: "BANC"
说明：
    如果 S 中不存这样的子串，则返回空字符串 ""。
    如果 S 中存在这样的子串，我们保证它是唯一的答案。
```

**思路**
- 同向双指针 + HaspMap

**Python**
```python
from collections import Counter

class Solution:
    def minWindow(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        map_t = Counter(t)  # 记录每个字符出现的次数
        
        found = dict()  # 保存滑动窗口 [i,j] 中出现的符号及其次数
        cnt_found = 0  # 记录已经匹配的数量
        min_len = len(s) + 1  # 记录最短结果
        
        res = ""
        l = 0
        for r in range(len(s)):
            if s[r] in map_t:
                found[s[r]] = found.get(s[r], 0) + 1
                if found[s[r]] <= map_t[s[r]]:
                    cnt_found += 1
            
            if cnt_found >= len(t):  # 如果存在这样的字串
                while (s[l] not in map_t) or (s[l] in found and found[s[l]] > map_t[s[l]]):
                    if s[l] in found and found[s[l]] > map_t[s[l]]:
                        found[s[l]] -= 1
                    l += 1
                    
                if r - l + 1 < min_len:  # 更新最小结果
                    min_len = r - l + 1
                    res = s[l: l + min_len]  # 等价于 s[l: r+1]
                    
        return res
```


## 长度最小的子数组（Minimum Size Subarray Sum）
> LeetCode/[209. 长度最小的子数组](https://leetcode-cn.com/problems/minimum-size-subarray-sum/description/)

**问题描述**
```
给定一个含有 n 个正整数的数组和一个正整数 s ，找出该数组中满足其和 ≥ s 的长度最小的连续子数组。如果不存在符合条件的连续子数组，返回 0。

示例: 
    输入: s = 7, nums = [2,3,1,2,4,3]
    输出: 2

解释: 子数组 [4,3] 是该条件下的长度最小的连续子数组。

进阶: 如果你已经完成了O(n) 时间复杂度的解法, 请尝试 O(n log n) 时间复杂度的解法。
```

**思路**
- 同向双指针

**Python**
```python
class Solution:
    def minSubArrayLen(self, s, A):
        """
        :type s: int
        :type A: List[int]
        :rtype: int
        """
        n = len(A)
        
        res = n + 1
        l = -1
        cur_sum = 0
        for r in range(n):
            if A[r] > s:
                return 1
            
            cur_sum += A[r]
            
            if cur_sum >= s:
                while cur_sum >= s:  # 注意要使用 >=
                    l += 1  # 因为 l 初始化为 -1，所以要先 ++
                    cur_sum -= A[l]
                
                res = min(res, r - l + 1)
                    
        return res if res != n + 1 else 0  # 结果不存在时返回 0
```


# 其他

## 合并两个有序数组（Merge Sorted Array）
> LeetCode/[88. 合并两个有序数组](https://leetcode-cn.com/problems/merge-sorted-array/description/)

**问题描述**
```
给定两个有序整数数组 nums1 和 nums2，将 nums2 合并到 nums1 中，使得 num1 成为一个有序数组。

说明:
    初始化 nums1 和 nums2 的元素数量分别为 m 和 n。
    你可以假设 nums1 有足够的空间（空间大小大于或等于 m + n）来保存 nums2 中的元素。

示例:
    输入:
        nums1 = [1,2,3,0,0,0], m = 3
        nums2 = [2,5,6],       n = 3
    输出: [1,2,2,3,5,6]
```

**思路**
- 从后往前遍历

**Python**
```python
class Solution:
    def merge(self, A, m, B, n):
        """
        :type A: List[int]
        :type m: int
        :type B: List[int]
        :type n: int
        :rtype: void Do not return anything, modify A in-place instead.
        """
        
        i = m - 1  # 指向 A 的末尾
        j = n - 1  # 指向 B 的末尾
        p = m + n - 1  # 指向总的末尾
        
        while i >= 0 and j >= 0:  # 注意结束条件
            if A[i] > B[j]:
                A[p] = A[i]
                p -= 1
                i -= 1
            else:
                A[p] = B[j]
                p -= 1
                j -= 1
                
        while j >= 0:  # 如果原 A 数组先排完，需要继续把 B 中的元素放进 A；如果 B 先排完，则不需要
            A[p] = B[j]
            p -= 1
            j -= 1
```


## 接雨水（Trapping Rain Water）（一维）
> LeetCode/[42. 接雨水](https://leetcode-cn.com/problems/trapping-rain-water/description/)

**问题描述**
```
给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。

示例:
    输入: [0,1,0,2,1,0,1,3,2,1,2,1]
    输出: 6
```
<div align="center"><img src="../_assets/TIM截图20180921215706.png" height="" /></div>

**思路 1**
- 一个简单的方法是**遍历两次**数组，分别记录每个位置左侧的最高点和右侧的最低点
- **C++**
    ```C++
    class Solution {
    public:
        int trap(vector<int>& H) {
            int n = H.size();
            
            vector<int> l_max(H);
            vector<int> r_max(H);
            
            for(int i=1; i<n; i++)
                l_max[i] = max(l_max[i-1], l_max[i]);
            
            for(int i=n-2; i>=0; i--)
                r_max[i] = max(r_max[i+1], r_max[i]);
            
            int ret = 0;
            for (int i=1; i<n-1; i++)
                ret += min(l_max[i], r_max[i]) - H[i];
            
            return ret;
        }
    };
    ``` 

**思路 2**
- 双指针，遍历一次数组
- **Python**
    ```python
    class Solution:
        def trap(self, A):
            """
            :type A: List[int]
            :rtype: int
            """
            n = len(A)
            l, r = 0, n - 1
            
            ans = 0
            max_l = max_r = 0
            
            while l <= r:
                if A[l] <= A[r]:
                    if A[l] > max_l:
                        max_l = A[l]
                    else:
                        ans += max_l - A[l]
                    
                    l += 1
                else:
                    if A[r] > max_r:
                        max_r = A[r]
                    else:
                        ans += max_r - A[r]
                    
                    r -= 1
                    
            return ans
    ```

## 颜色分类（Sort Colors）
> LeetCode/[75. 颜色分类](https://leetcode-cn.com/problems/sort-colors/description/)

**问题描述**
```
给定一个包含红色、白色和蓝色，一共 n 个元素的数组，原地对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列。

此题中，我们使用整数 0、 1 和 2 分别表示红色、白色和蓝色。

注意:
    不能使用代码库中的排序函数来解决这道题。

示例:
    输入: [2,0,2,1,1,0]
    输出: [0,0,1,1,2,2]

进阶：
    一个直观的解决方案是使用计数排序的两趟扫描算法。
    首先，迭代计算出0、1 和 2 元素的个数，然后按照0、1、2的排序，重写当前数组。
    你能想出一个仅使用常数空间的一趟扫描算法吗？
```

**思路**
- 首尾双指针
    - `l` 记录最后一个 0 的位置
    - `r` 记录第一个 2 的位置
    - `i` 表示当前遍历的元素

**Python**
```python
class Solution:
    def sortColors(self, A):
        """
        :type A: List[int]
        :rtype: void Do not return anything, modify A in-place instead.
        """
        n = len(A)
        
        l = 0  # l 指示最后一个 0
        r = n - 1  # r 指示第一个 2
        
        i = 0
        while i <= r:
            if A[i] == 0:
                A[i] = A[l]
                A[l] = 0  # 确定最后一个 0
                l += 1
            if A[i] == 2:
                A[i] = A[r]
                A[r] = 2  # 确定第一个 2
                r -= 1
                i -= 1  # 注意回退，因为不确定原 A[r] 处是什么
            i += 1
```


## 两个数组的交集（Intersection of Two Arrays）
> LeetCode/[349. 两个数组的交集](https://leetcode-cn.com/problems/intersection-of-two-arrays/description/)

**问题描述**
```
给定两个数组，编写一个函数来计算它们的交集。

示例 1:
    输入: nums1 = [1,2,2,1], nums2 = [2,2]
    输出: [2]

示例 2:
    输入: nums1 = [4,9,5], nums2 = [9,4,9,8,4]
    输出: [9,4]

说明:
    输出结果中的每个元素一定是唯一的。
    我们可以不考虑输出结果的顺序。
```

**思路**
- 1）排序 + 二分
- 2）Hash
- 3）排序 + 双指针

**Python**（双指针）
```python
class Solution:
    def intersection(self, A, B):
        """
        :type A: List[int]
        :type B: List[int]
        :rtype: List[int]
        """
        A.sort()
        B.sort()
        
        m = len(A)
        n = len(B)
        
        res = []
        i, j = 0, 0
        while i < m and j < n:
            if A[i] < B[j]:
                i += 1
            elif A[i] > B[j]:
                j += 1
            else:
                if (i > 0 and A[i - 1] == A[i]) or (j > 0 and B[j - 1] == B[j]):  # 去重
                    pass
                else:
                    res.append(A[i])
                    
                i += 1
                j += 1
                
                # i += 1
                # while i < m and A[i - 1] == A[i]:
                #     i += 1
                # j += 1
                # while j < n and B[j - 1] == B[j]:
                #     j += 1
        
        return res
```