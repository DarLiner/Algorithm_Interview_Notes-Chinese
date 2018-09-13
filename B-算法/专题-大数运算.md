专题-大数运算
===

Index
---
<!-- TOC -->

- [大数取模](#大数取模)
  - [取模运算的性质](#取模运算的性质)
  - [快速幂取模](#快速幂取模)
- [大数加/减/乘/除](#大数加减乘除)

<!-- /TOC -->

## 大数取模

### 取模运算的性质
<div align="center"><a href="http://www.codecogs.com/eqnedit.php?latex=\fn_cs&space;\begin{aligned}&space;&(a&plus;b)\;\%\;m=(a\;\%\;m&plus;b\;\%\;m)\;\%\;m\\&space;&(a-b)\;\%\;m=(a\;\%\;m-b\;\%\;m{\color{Red}\;&plus;\;m})\;\%\;m\\&space;&ab\;\%\;m=(a\;\%\;m)(b\;\%\;m)\;\%\;m&space;\end{aligned}"><img src="../_assets/公式_20180811215806.png" height="" /></a></div>

- 因为 `(a%n) - (b%n)` 可能小于 `n`，所以 `+n`
- 因为 `(a%n)(b%n)` 可能溢出，计算前应该强转为 `long long`

**Code - C++**
- 输入 `a` 为长度小于 1000 的字符串，`b` 为小于 `100000` 的整数
  ```C++
  int big_mod(const string& a, int b) {
      long ret = 0;  // 防止 ret * 10 溢出
      for (auto c : a) {
          ret = ((ret * 10) % b + (c - '0') % b) % b;  // ret = ((ret * 10) + (c - '0')) % b
      }
      return (int)ret;
  }

  /* 示例说明
  1234 % 11 == ((((0*10 + 1)*10 + 2)*10 + 3)*10 + 4) % 11
            == ((((0*10 + 1)*10 + 2)*10 + 3)*10 % 11 + 4 % 11) % 11
            == ((((0*10 + 1)*10 + 2)*10 % 11 + 3 % 11)*10 % 11 + 4 % 11) % 11
            == ((((0*10 + 1)*10 % 11 + 2 % 11)*10 % 11 + 3 % 11)*10 % 11 + 4 % 11) % 11
            == ((((0*10 % 11 + 1 % 11)*10 % 11 + 2 % 11)*10 % 11 + 3 % 11)*10 % 11 + 4 % 11) % 11
  ```

### 快速幂取模
- 计算 `a^n % b`
- **基本方法**：根据取模的性质 3 —— `ab % m == (a%m)(b%m) % m`
  ```C++
  int big_mod(int a, int n, int b) {
      long long ret = 1;
      while(n--) {
          ret *= a % b;
          ret %= b;
      }
      return (int)ret;
      
  }
  ```
  - 时间复杂度 `O(N)`
- **快速幂取模**
  ```C++
  int big_mod(int a, int n, int b) {
      long long ret = 1;
      while(n) {
          if (n & 1)
              ret = (ret*a) % b;
          a = (a*a) % b;
          n >>= 1;
      }
      return (int)ret;
  }
  ```
  - 代码跟快速幂很像
  - 示例说明
    ```
    2^10 % 11 == (2^5 % 11)(2^5 % 11) % 11
              == ((2 % 11)(2^4 % 11))((2 % 11)(2^4 % 11)) % 11
              == ...
    ```

## 大数加/减/乘/除
> [大数的四则运算（加法、减法、乘法、除法） - 落枫飘飘](https://www.cnblogs.com/wuqianling/p/5387099.html) - 博客园 