专题-数学
===

Index
---
<!-- TOC -->

- [直线上最多的点数](#直线上最多的点数)

<!-- /TOC -->

### 直线上最多的点数
> LeetCode - [149. 直线上最多的点数](https://leetcode-cn.com/problems/max-points-on-a-line/description/)

**问题描述**

**思路**
- ~~根据 `y=kx+b`，计算每两个点的 `(k, b)` 对，配合 map 存储~~
- 使用 `(k,b)` 可能存在精度问题，更好的方法是使用 `ax+by+c=0`
- 两者本质上没有区别，实际上就是把 `k` 分为 `a/b` 存储
- 注意：将 `{a, b}` 作为 key 时应该先利用**最大公约数**缩小 `a` 和 `b`

**C++**
```C++
class Solution {
    int gcd(int a, int b) {
        return b == 0 ? a : gcd(b, a%b);
    }
public:
    int maxPoints(vector<Point>& P) {
        int n = P.size();
        if (n <= 2) return n;

        int ret = 0;
        for (int i = 0; i < n; i++) {
            map<pair<int, int>, int> line;
            int tmp = 0;                  // 保存与 P[i] 共线的点
            int dup = 0;                  // 记录重复点
            int col = 0;                  // 跟 P[i] 垂直的点

            for (int j = i + 1; j < n; j++) {
                if (P[i].x == P[j].x && P[i].y == P[j].y) {
                    dup += 1;
                }
                else if (P[i].x == P[j].x) {
                    col += 1;
                    tmp = max(tmp, col);
                }
                else {
                    int a = P[i].y - P[j].y;
                    int b = P[i].x - P[j].x;

                    int t = gcd(a, b);      // 利用最大公约数缩小
                    a /= t;
                    b /= t;
                    line[{a, b}]++;
                    tmp = max(tmp, line[{a, b}]);
                 }
            }
            ret = max(ret, tmp + dup + 1);
        }

        return ret;
    }
};
```