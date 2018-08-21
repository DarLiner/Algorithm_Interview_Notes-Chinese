题解-LeetCode
===

Index
---
<!-- TOC -->

- [数组](#数组)
  - [15. 三数之和（双指针）](#15-三数之和双指针)
  - [16. 最接近的三数之和（双指针）](#16-最接近的三数之和双指针)
  - [26. 删除排序数组中的重复项](#26-删除排序数组中的重复项)

<!-- /TOC -->

## 数组

### 15. 三数之和（双指针）
> https://leetcode-cn.com/problems/3sum/description/

**问题描述**
```
给定一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？
找出所有满足条件且不重复的三元组。
```

**思路**
- 因为需要输出所有结果，所以不推荐使用 map 来做
- 判断 `a + b + c = 0`，实际上等价于判断 `-a = b + c`
- 基本思路：对数组排序后，对每个 `a`，用首尾双指针进行遍历，具体过程看代码更清晰
- 去重的方法：排序后，跳过相同的数即可
- 注意边界条件

**C++**
```C++
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        if (nums.size() < 3) return vector<vector<int>>();  // 输入数量小于 3 直接退出

        sort(nums.begin(), nums.end());                     // 排序

        vector<vector<int>> ret;
        for (int i = 0; i <= nums.size() - 3; i++) {
            if (i > 0 && nums[i] == nums[i - 1]) continue;  // 跳过第一个数相同的情况

            int target = -nums[i];
            int lo = i + 1;
            int hi = nums.size() - 1;
            while (lo < hi) {
                if (nums[lo] + nums[hi] < target)
                    lo++;
                else if (nums[lo] + nums[hi] > target)
                    hi--;
                else {
                    ret.push_back({ nums[i], nums[lo], nums[hi] });
                    lo++, hi--;   // 不要忘了这双指针都要移动

                    while (lo < hi && nums[lo] == nums[lo - 1]) lo++;   // 跳过第二个数相同的情况
                    while (lo < hi && nums[hi] == nums[hi + 1]) hi--;   // 跳过第三个数相同的情况
                }
            }
        }
        return ret;
    }
};
```

### 16. 最接近的三数之和（双指针）
> https://leetcode-cn.com/problems/3sum-closest/description/

**题目描述**
```
给定一个包括 n 个整数的数组 nums 和 一个目标值 target。找出 nums 中的三个整数，使得它们的和与 target 最接近。返回这三个数的和。假定每组输入只存在唯一答案。

例如，给定数组 nums = [-1，2，1，-4], 和 target = 1.

与 target 最接近的三个数的和为 2. (-1 + 2 + 1 = 2).
```

**思路**
- 同[三数之和（双指针）](#三数之和双指针)
- 区别仅在于循环内的操作不同

**C++**
```C++
class Solution {
public:
    int threeSumClosest(vector<int>& nums, int target) {
        if (nums.size() < 3) return 0;

        sort(nums.begin(), nums.end());  // 别忘了排序

        auto ret = nums[0] + nums[1] + nums[2];  // 保存答案
        for (int i = 0; i <= nums.size()-3; i++) {
            int lo = i + 1;
            int hi = nums.size() - 1;

            while (lo < hi) {
                auto sum = nums[i] + nums[lo] + nums[hi];
                if (abs(target - sum) < abs(target - ret)) {
                    ret = sum;
                    if (ret == target)
                        return ret;
                }
                sum < target ? lo++ : hi--;
            }
        }
        return ret;
    }
};
```

### 26. 删除排序数组中的重复项
> https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array/description/

**题目描述**
```
给定一个排序数组，你需要在原地删除重复出现的元素，使得每个元素只出现一次，返回移除后数组的新长度。

不要使用额外的数组空间，你必须在原地修改输入数组并在使用 O(1) 额外空间的条件下完成。
```

**C++**
```C++
class Solution {
public:
    int removeDuplicates(vector<int>& nums) {
        if (nums.size() <= 1) return nums.size();
            
        int lo = 0;
        int hi = lo + 1;
        
        int n = nums.size();
        while (hi < n) {
            while (hi < n && nums[hi] == nums[lo]) hi++;
            nums[++lo] = nums[hi];
        }
        
        return lo;
    }
};
```