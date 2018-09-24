Cpp-库函数
===

Index
---
<!-- TOC -->

- [常用操作](#常用操作)
  - [去重](#去重)
- [string 相关](#string-相关)
  - [模式匹配](#模式匹配)

<!-- /TOC -->

## 常用操作

### 去重


## string 相关

### 模式匹配

**第一次匹配的位置**
```C++
// 返回第一次匹配的位置，返回 -1 表示不匹配
size_t str_match(const string& S, const string& T) {
    auto index = S.find(T);

    return index;
}
```

**统计匹配的次数**
```C++
// 统计匹配的次数
size_t str_count(const string& S, const string& T) {

    size_t cnt = 0;
    for (size_t i = 0; (i = S.find(T, i)) != string::npos; i++, cnt++);

    return cnt;
}
```
- 经测试，比手写的 **KMP 算法**更快！

