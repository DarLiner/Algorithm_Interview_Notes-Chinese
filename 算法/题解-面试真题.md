题解-面试真题
===
- 记录一些暂时没找到原型的面试真题

Index
---
<!-- TOC -->

- [给定 `n` 个`[0,n)`区间内的数，统计每个数出现的次数，不使用额外空间](#给定-n-个0n区间内的数统计每个数出现的次数不使用额外空间)

<!-- /TOC -->

## 给定 `n` 个`[0,n)`区间内的数，统计每个数出现的次数，不使用额外空间
> 头条

**思路**：
- 基于两个基本运算：
  ```tex
  若 i ∈ [0, n)，则有
    (t*n + i) % n = i
    (t*n + i) / n = t 
  ```
- 顺序遍历每个数 i，i 每出现一次，则 nums[i] += n
- 遍历结束后，i 出现的次数，即 `nums[i] / n`，同时利用 `nums[i] % n` 可以还原之前 `nums[i]` 上的数。

**C++**（未测试）
```C++
vector<int> nums;

void init(vector<int>& nums) {
    for (int i = 0; i < nums.size(); i++) {
        nums[nums[i]] += n;
    }
}

int cnt(int k) {
    return nums[k] / n;
}
```