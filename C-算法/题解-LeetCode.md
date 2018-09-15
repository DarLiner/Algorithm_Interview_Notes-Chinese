题解-LeetCode
===

RoadMap
---
- [二叉树](#二叉树)
  - [DFS](#dfs)
- [数组](#数组)
  - [双指针](#双指针)
  - [多级排序](#多级排序)
  - [其他](#其他)
- [暴力搜索](#暴力搜索)
  - [DFS](#dfs)
  - [BFS](#bfs)


Index
---
<!-- TOC -->

- [二叉树](#二叉树)
  - [124. 二叉树中的最大路径和（DFS）](#124-二叉树中的最大路径和dfs)
- [数组](#数组)
  - [560. 和为K的子数组（前缀和 + Map）](#560-和为k的子数组前缀和--map)
  - [838. 推多米诺（双指针）](#838-推多米诺双指针)
  - [15. 三数之和（双指针）](#15-三数之和双指针)
  - [16. 最接近的三数之和（双指针）](#16-最接近的三数之和双指针)
  - [729. 我的日程安排表 I（多级排序）](#729-我的日程安排表-i多级排序)
  - [26. 删除排序数组中的重复项](#26-删除排序数组中的重复项)
- [暴力搜索](#暴力搜索)
  - [200. 岛屿的个数（DFS | BFS）](#200-岛屿的个数dfs--bfs)

<!-- /TOC -->

## 二叉树

### 124. 二叉树中的最大路径和（DFS）
> https://leetcode-cn.com/problems/binary-tree-maximum-path-sum/description/

**题目描述**
```
给定一个非空二叉树，返回其最大路径和。

本题中，路径被定义为一条从树中任意节点出发，达到任意节点的序列。
该路径至少包含一个节点，且不一定经过根节点。

输入: [1,2,3]

       1
      / \
     2   3

输出: 6
```

**暴力求解的思路**
- 利用一个子函数，求出每个节点**最大深度路径和**（做法类似求**树的深度**）
  - 注意，因为节点中的值可能为负数，所以最大深度路径和不一定都会到达叶子
  - 同样，最大深度路径和也可能为负数，此时应该返回 0
- 接着对每个节点，**经过该节点**的最大路径和为
  ```
  该节点的值 + 左子树的最大深度路径和 + 右子树的最大深度路径和
  ```
- **空树的最大路径和**应该为负无穷（作为递归基）；但实际用例中没有空树的情况

**C++**
- 初始版本（AC）
  ```C++
  class Solution {
      const int inf = 0x3f3f3f3f;

      int maxDeep(TreeNode* root) {
          if (!root) return 0;

          // 避免负数的情况
          return max(0, root->val + max({ 0, maxDeep(root->left), maxDeep(root->right) }));
      }
  public:
      int maxPathSum(TreeNode* root) {
          if (root == nullptr) return -inf;  // 空树返回负无穷

          int path_sum = root->val + maxDeep(root->right) + maxDeep(root->left);

          return max({ path_sum, maxPathSum(root->left), maxPathSum(root->right) });
      }
  };
  ```
- **改进**
  - 使用一个变量保存中间结果
  ```
  class Solution {
      // C++11 支持 就地 初始化
      const int inf = 0x3f3f3f3f;
      int ret = -inf;

      int maxDeepSum(TreeNode* node) {
          if (node == nullptr)
              return 0;

          int l_sum = max(0, maxDeepSum(node->left));
          int r_sum = max(0, maxDeepSum(node->right));

          ret = max(ret, node->val + l_sum + r_sum);
          return node->val + max(l_sum, r_sum);
      }
  public:
      int maxPathSum(TreeNode* root) {
          maxDeepSum(root);
          return ret;
      }
  };
  ```

**优化方案**
- 记忆化搜索（树DP）；简单来说，就是保存中间结果


## 数组

### 560. 和为K的子数组（前缀和 + Map）
> https://leetcode-cn.com/problems/subarray-sum-equals-k/description/

**问题描述**
```
给定一个整数数组和一个整数 k，你需要找到该数组中和为 k 的连续的子数组的个数。

示例 1 :

输入:nums = [1,1,1], k = 2
输出: 2 , [1,1] 与 [1,1] 为两种不同的情况。
```

**思路**
- 前缀和 + map
- 难点在于重复的情况也要记录

**C++**
```C++
class Solution {
public:
    int subarraySum(vector<int>& nums, int k) {
        
        int sum = 0;
        int cnt = 0;
        map<int, int> bok;
        bok[0] = 1;
        for (int i=0; i<nums.size(); i++) {
            sum += nums[i];
            cnt += bok[sum-k];  // 注意顺序，必须先更新 cnt，再更新 bok[sum]
            bok[sum]++;         // 如果 bok[sum] 不存在，会默认插入并初始化为 0，所以可以这么写
            // 等价于以下代码
            // if (bok.count(sum))
            //     bok[sum] += 1;
            // else
            //     bok[sum] = 1;
        }
        
        return cnt;
    }
};
```

### 838. 推多米诺（双指针）
> https://leetcode-cn.com/problems/push-dominoes/description/

**问题描述**
```
一行中有 N 张多米诺骨牌，我们将每张多米诺骨牌垂直竖立。

在开始时，我们同时把一些多米诺骨牌向左或向右推。

每过一秒，倒向左边的多米诺骨牌会推动其左侧相邻的多米诺骨牌。

同样地，倒向右边的多米诺骨牌也会推动竖立在其右侧的相邻多米诺骨牌。

如果同时有多米诺骨牌落在一张垂直竖立的多米诺骨牌的两边，由于受力平衡， 该骨牌仍然保持不变。

就这个问题而言，我们会认为正在下降的多米诺骨牌不会对其它正在下降或已经下降的多米诺骨牌施加额外的力。

给定表示初始状态的字符串 "S" 。如果第 i 张多米诺骨牌被推向左边，则 S[i] = 'L'；如果第 i 张多米诺骨牌被推向右边，则 S[i] = 'R'；如果第 i 张多米诺骨牌没有被推动，则 S[i] = '.'。

返回表示最终状态的字符串。

示例 1：
  输入：".L.R...LR..L.."
  输出："LL.RR.LLRRLL.."

示例 2：
  输入："RR.L"
  输出："RR.L"
  说明：第一张多米诺骨牌没有给第二张施加额外的力。

提示：
  0 <= N <= 10^5
  表示多米诺骨牌状态的字符串只含有 'L'，'R'; 以及 '.';
```

**思路**
- 如果给原始输入左右分别加上一个 "L" 和 "R"，那么共有以下 4 种可能
  ```
  'R......R' => 'RRRRRRRR'
  'L......L' => 'LLLLLLLL'
  'L......R' => 'L......R'
  'R......L' => 'RRRRLLLL' or 'RRRR.LLLL'
  ```
  > [[C++/Java/Python] Two Pointers](https://leetcode.com/problems/push-dominoes/discuss/132332/C++JavaPython-Two-Pointers) - LeetCode

**C++**
```C++
class Solution {
public:
    string pushDominoes(string d) {
        string s = "L" + d + "R";
        string ret = "";

        int lo = 0, hi = 1;
        for (; hi < s.length(); hi++) {
            if (s[hi] == '.')
                continue;

            if (lo > 0)         // 注意这一步操作
                ret += s[lo];

            int delta = hi - lo - 1;
            if (s[lo] == s[hi])
                ret += string(delta, s[lo]);  // string 的一种构造函数，以 s[lo] 为每个字符，生成长度为 h_l 的字符串
            else if (s[lo] == 'L' && s[hi] == 'R')
                ret += string(delta, '.');
            else if (s[lo] == 'R' && s[hi] == 'L')
                ret += string(delta / 2, 'R') + string(delta & 1, '.') + string(delta / 2, 'L');

            lo = hi;
        }

        return ret;
    }
};
```

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
