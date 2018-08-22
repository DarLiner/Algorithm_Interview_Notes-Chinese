题解-LeetCode
===

Index
---
<!-- TOC -->

- [数组](#数组)
  - [15. 三数之和（双指针）](#15-三数之和双指针)
  - [16. 最接近的三数之和（双指针）](#16-最接近的三数之和双指针)
  - [26. 删除排序数组中的重复项（迭代）](#26-删除排序数组中的重复项迭代)
  - [729. 我的日程安排表 I（多级排序）](#729-我的日程安排表-i多级排序)
- [暴力搜索](#暴力搜索)
  - [200. 岛屿的个数（DFS | BFS）](#200-岛屿的个数dfs--bfs)

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

### 26. 删除排序数组中的重复项（迭代）
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

### 729. 我的日程安排表 I（多级排序）
> https://leetcode-cn.com/problems/my-calendar-i/description/

**题目描述**
```
实现一个 MyCalendar 类来存放你的日程安排。如果要添加的时间内没有其他安排，则可以存储这个新的日程安排。

MyCalendar 有一个 book(int start, int end)方法。它意味着在 start 到 end 时间内增加一个日程安排，注意，这里的时间是半开区间，即 [start, end), 实数 x 的范围为，  start <= x < end。

当两个日程安排有一些时间上的交叉时（例如两个日程安排都在同一时间内），就会产生重复预订。

每次调用 MyCalendar.book方法时，如果可以将日程安排成功添加到日历中而不会导致重复预订，返回 true。否则，返回 false 并且不要将该日程安排添加到日历中。

请按照以下步骤调用 MyCalendar 类: MyCalendar cal = new MyCalendar(); MyCalendar.book(start, end)

示例 1:
  MyCalendar();
  MyCalendar.book(10, 20); // returns true
  MyCalendar.book(15, 25); // returns false
  MyCalendar.book(20, 30); // returns true
  
  解释: 
    第一个日程安排可以添加到日历中.  第二个日程安排不能添加到日历中，因为时间 15 已经被第一个日程安排预定了。
    第三个日程安排可以添加到日历中，因为第一个日程安排并不包含时间 20 。
```

**思路**
- 按 start 排序，利用二分查找（lower_bound），找到待插入位置
- 关键是判断能否插入，需要同时判断前后位置，此时需要注意边界条件
  ```
  假设 (s, e) 的待插入位置如下，利用 lower_bound 将返回指向 (s2, e2) 的迭代器
    (s1, e1) _ (s2, e2)
  此时，有 s1 < s < s2
  此时如果 e > s2 || s < e1 则无法插入；反之则可以插入
  ```
- **C++**
  ```C++
  class MyCalendar {
      vector<pair<int,int> > m_book;
      
  public:
      MyCalendar() {}
      
      bool book(int s, int e) {
          pair<int, int> tmp{s, e};
          auto it = lower_bound(m_book.begin(), m_book.end(), tmp);  
          // 默认按 pair.first 排序，
          // 所以虽然是多级排序，但实际上并没有额外的操作

          // 注意下面两步的边界判断
          if (it != m_book.end() && e > it->first)
              return false;
          if (it != m_book.begin() && s < (it-1)->second) 
              return false;
          m_book.insert(it, tmp);
          return true;
      }
  };
  ```
- **C++**-利用 STL 中的 set/map 结构自动排序
  ```C++
  // 使用 Set
  class MyCalendar {
      set<pair<int,int> > m_book;
      
  public:
      MyCalendar() {}
      
      bool book(int s, int e) {
          pair<int,int> tmp{s, e};
          
          auto it = m_book.lower_bound(tmp);
          if (it != m_book.end() && it->first < e)
              return false;
          if (it != m_book.begin() && (--it)->second > s)  // 注意 set 只支持 -- 操作符，而不是支持 - 操作符，即无法使用 (it-1)->second
              return false;
          m_book.insert(tmp);
          return true;
      }
  };

  // 使用 Map
  class MyCalendar {
      map<int, int> books;
  public:
      bool book(int s, int e) {
          auto next = books.lower_bound(s);
          if (next != books.end() && next->first < e) 
              return false;
          if (next != books.begin() && s < (--next)->second) 
              return false;
          books[s] = e;
          return true;
      }
  };
  ```

## 暴力搜索

### 200. 岛屿的个数（DFS | BFS）
> https://leetcode-cn.com/problems/number-of-islands/description/

**问题描述**
```
给定一个由 '1'（陆地）和 '0'（水）组成的的二维网格，计算岛屿的数量。一个岛被水包围，并且它是通过水平方向或垂直方向上相邻的陆地连接而成的。你可以假设网格的四个边均被水包围。

示例 1:
  输入:
  11110
  11010
  11000
  00000
  输出: 1
示例 2:
  输入:
  11000
  11000
  00100
  00011
  输出: 3
```

**思路**
- 经典的 DFS | BFS 问题，搜索连通域的个数

**Code**: DFS
```
class Solution {
    int n, m;
public:
    int numIslands(vector<vector<char>>& grid) {  // 注意：是 char 不是 int
        if (!grid.empty()) n = grid.size();
        else return 0;
        if (!grid[0].empty()) m = grid[0].size();
        else return 0;
        
        int ret = 0;
        for (int i=0; i<n; i++)
            for (int j=0; j<m; j++) {
                if (grid[i][j] != '0') {
                    ret += 1;
                    dfs(grid, i, j);
                }
            }
        
        return ret;
    }
    
    void dfs(vector<vector<char>>& grid, int i, int j) {
        if (i < 0 || i >= n || j < 0 || j >= m )  // 边界判断（递归基）
            return;
            
        if (grid[i][j] == '0')
            return;
        else {
            grid[i][j] = '0';  // 如果不想修改原数据，可以复制一个
            // 4 个方向 dfs；一些问题会扩展成 8 个方向，本质上没有区别
            dfs(grid, i+1, j);
            dfs(grid, i-1, j);
            dfs(grid, i, j+1);
            dfs(grid, i, j-1);
        }
    }
};
```

**Code**: BFS
```C++
class Solution {
    int n, m;
public:
    int numIslands(vector<vector<char>>& grid) {
        if (!grid.empty()) n = grid.size();
        else return 0;
        if (!grid[0].empty()) m = grid[0].size();
        else return 0;
        
        int ret = 0;
        for (int i=0; i<n; i++)
            for (int j=0; j<m; j++) {
                if (grid[i][j] != '0') {
                    ret += 1;
                    bfs(grid, i, j);
                }
            }
        
        return ret;
    }
    
    void bfs(vector<vector<char>>& grid, int i, int j) {
        queue<vector<int> > q;
        
        q.push({i,j});
        grid[i][j] = '0';
        while (!q.empty()) {
            i = q.front()[0], j = q.front()[1];
            q.pop();  // 当前节点出队
                      // 当前节点的四周节点依次入队
            if (i > 0 && grid[i-1][j] == '1') {
                q.push({i-1,j});
                grid[i-1][j] = '0';
            }
            if (i < n-1 && grid[i+1][j] == '1') {
                q.push({i+1,j});
                grid[i+1][j] = '0';
            }
            if (j > 0 && grid[i][j-1] == '1') {
                q.push({i,j-1});
                grid[i][j-1] = '0';
            }
            if (j < m-1 && grid[i][j+1] == '1') {
                q.push({i,j+1});
                grid[i][j+1] = '0';
            }
        }
    }
};
```