专题-字符串
===

Index
---
<!-- TOC -->

- [模式匹配 TODO](#模式匹配-todo)
- [进制转换](#进制转换)
    - [10进制转任意进制](#10进制转任意进制)
    - [任意进制转10进制](#任意进制转10进制)
    - [长地址转短地址](#长地址转短地址)
- [功能函数](#功能函数)
    - [`atoi()`](#atoi)
- [表达式转化（中缀，后缀，前缀）](#表达式转化中缀后缀前缀)
    - [中缀转后缀](#中缀转后缀)

<!-- /TOC -->

## 模式匹配 TODO

**模式匹配基本方法**
> [字符串模式匹配算法——BM、Horspool、Sunday、KMP、KR、AC算法 - 单车博客园](https://www.cnblogs.com/dancheblog/p/3517338.html) - 博客园 
- 双指针
- KMP
- ...

## 进制转换

### 10进制转任意进制
```python
from string import digits, ascii_uppercase, ascii_lowercase

Alphabet = digits + ascii_lowercase + ascii_uppercase
print(Alphabet)  # "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
```

**递归方法**
```python
def ten2any(n, b=62):
    """"""
    assert b <= 62

    n, index = divmod(n, b)  # n = n // b, index = n % b
    if n > 0:
        return ten2any(n, b) + Alphabet[index]
    else:
        return Alphabet[index]
```

**迭代方法**
```python
def ten2any_2(n, b=62):
    """"""
    ret = ""
    while n > 0:
        n, index = divmod(n, b)
        ret = Alphabet[index] + ret

    return ret
```

### 任意进制转10进制

**迭代方法**
```python
from string import digits, ascii_uppercase, ascii_lowercase

Alphabet = digits + ascii_lowercase + ascii_uppercase
print(Alphabet)  # "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

def any2ten(s, base=62):
    """"""
    n = 0
    for i, c in enumerate(reversed(s)):  # reversed(s)
        index = Alphabet.index(c)
        n += index * pow(base, i)

    return n
```

### 长地址转短地址
> [短 URL 系统是怎么设计的？](https://www.zhihu.com/question/29270034/answer/46446911) - 知乎

**基本思路**
- 发号策略：给每一个收录的长地址，分配一个自增索引
- 将分配的索引转换为一个 62 进制数（10数字+26大写字母+26小写字母）
- **注意**，发号机制不会判断是否重复

**重复长地址怎么处理？**
- 如果要求相同的长地址要对应唯一的短地址，那么唯一的方法就是维护一个映射表
- 但是维护一个映射表可能比忽略重复需要更多的空间
- 一个**折衷**的方法是——维护一个“最近”的长对短映射，比如采用一小时过期的机制来实现 LRU 淘汰
  - 当一个地址被频繁使用，那么它始终在这个 key-value 表中，总能返回当初生成那个短地址
  - 如果它使用并不频繁，那么长对短的 key 会过期，LRU 机制就会自动淘汰掉它

**如何保证发号器的大并发高可用？**
- 如果做成分布式的，那么多节点要保持同步加 1，多点同时写入，以 CAP 理论看，很难做到
- 一个简单的处理方式是，使用多个发号器，以**奇偶**/**尾数**区分
  - 比如一个发单号，一个发双号；
  - 或者实现 1000 个发号器，分别发尾号为 0 到 999 的号；每发一个号，每个发号器加 1000，而不是加 1


## 功能函数

### `atoi()`
**功能简述**
- 将字符串（C风格）转换成整型；
- 会跳过前面的空格字符，直到遇上数字或正负号才开始转换；
- 如果遇到的第一个字符不是数字，则返回 0，并结束转化；
- 当遇到**非数字**或**结束符**('\0') 时结束转化，并将结果返回（整型）
- 如果发生溢出，则输出 INT_MAX 或 INT_MIN;
- 内置 atoi 不会处理 NULL 指针

**合法样例**：
```
"123"           -> 123
"+123"          -> 123
"-123"          -> -123
"123abc"        -> 123
"   123abc"     -> 123
"a123"          -> 0
```

**核心代码（C++）**
```C++
while (*p >= '0' && *p <= '9') {
    ret = ret * 10 + (*p - '0');
    p++;
}
```
- 除了核心代码，更重要的是**异常处理**和**溢出判断**

**完整代码**
```C++
int atoi_my(const char* const cs) {
    if (cs == nullptr) return 0;

    int ret = 0;
    auto *p = cs;  // cs 为常指针

    // 跳过前面的空格
    while (isspace(*p)) p++;

    // 判断正负
    int sign = 1;   // 默认正数
    if (*p == '-') sign = -1;
    if (*p == '-' || *p == '+') p++;

    // 核心代码：循环转换整数（加入溢出判断）
    int tmp;  // 保存临时结果，用于溢出判断
    while (*p >= '0' && *p <= '9') {
        tmp = ret * 10 + (*p - '0');
        if (tmp / 10 != ret) {  // 溢出判断
            return sign > 0 ? INT_MAX : INT_MIN;
        }
        ret = tmp;
        p++;
    }
    // 核心代码（无溢出判断）
    //while (*p >= '0' && *p <= '9') {
    //    ret = ret * 10 + (*p - '0');
    //    p++;
    //}

    return sign * ret;
}
```


## 表达式转化（中缀，后缀，前缀）
> [前缀、中缀、后缀表达式](https://blog.csdn.net/antineutrino/article/details/6763722) - CSDN博客 

**为什么要把中缀表达式转化为后缀，前缀？**
- 计算机无法直接计算带有括号，以及区分优先级的表达式，或者说很难计算。
- 使用后缀，前缀，消除了括号和优先级。

**如何计算后缀，前缀表达式？**
- **手工求法**：
  > [中缀表达式与前、后缀表达式转化简单的技巧[转] - Hslim](https://www.cnblogs.com/Hslim/p/5008460.html) - 博客园 
  - 示例：`a+b*c-(d+e)`
  - 第一步：按照运算符的优先级对所有的运算单位加括号
    ```
    ((a+(b*c))-(d+e))
    ```
  - 第二步：转换前缀与后缀表达式
    ```
    后缀：把运算符号移动到对应的括号后面
          ((a(bc)* )+ (de)+ )-
      把括号去掉：
          abc*+de+- 
    前缀：把运算符号移动到对应的括号前面
          -( +(a *(bc)) +(de))
      把括号去掉：
          -+a*bc+de
    ```
- **计算机方法**：
  - [中缀转后缀](#中缀转后缀)

### 中缀转后缀

**思路**
- 从左到右遍历中缀表达式，遇到**操作数**则输出；遇到操作符，若当前操作符的**优先级大于**栈顶操作符优先级，进栈；否则，弹出栈顶操作符，当前操作符进栈。（这只是一段比较**粗糙**的描述，更多细节请参考链接或下面的源码）
  > [中、前、后缀表达式](https://blog.csdn.net/lin74love/article/details/65631935) - CSDN博客 

**C++**（未测试）
<!-- - 只测试了部分样例，如果有**未通过的样例**请告诉我
- 以下代码中有一些**循环**可能是多余的；如果字符串合法，应该只需要判断一次，而不需要循环处理（未验证） -->
```C++
#include <iostream>
#include <string>
#include <sstream>
#include <stack>
#include <queue>
#include <set>
using namespace std;

//set<char> l1{ '+', '-' };
//set<char> l2{ '*', '/' };
//
//vector<set<char>> l{ l1, l2 };

int get_level(char c) {
    switch (c) {
    case '+':
    case '-':
        return 1;
    case '*':
    case '/':
        return 2;
    //case '(':
    //    return 3;
    default:
        return -1;
    }
}

string infix2suffix(const string& s) {
    stack<char> tmp;        // 符号栈
    queue<string> ans;      // 必须使用 string 队列，因为可能存在多位数字
    //stringstream ret;       // 用字符流模拟队列

    bool num_flag = false;      // 用于判断数字的末尾 
                                //初始设为 false 是为了避免第一个字符是括号
    //int v = 0;            // 保存数值
    string v{ "" };         // 用字符串保存更好，这样还能支持字母形式的表达式
    for (auto c : s) {
        // 处理数字
        if (isalnum(c)) {           // 处理多位数字
            v.append(string(1, c)); // 注意，char 字符不能直接转 string
            num_flag = true;
        }
        else {
            if (num_flag) {     // 因为可能存在多位数字，所以数字需要等遇到第一个非数字字符才入队
                ans.push(v);
                //ret << v << ' ';
                v.clear();
                num_flag = false;
            }

            // 处理运算符的过程
            if (c == ')') {  // 如果遇到右括号，则依次弹出栈顶符号，直到遇到**第一个**左括号并弹出（坑点 1：可能存在连续的左括号）
                while (!tmp.empty()) {
                    if (tmp.top() == '(') {
                        tmp.pop();
                        break;
                    }
                    ans.push(string(1, tmp.top()));
                    //ret << tmp.top() << ' ';
                    tmp.pop();
                }
            } // 注意这两个判断的顺序（坑点 2：右括号是始终不用入栈的，所以应该先处理右括号）
            else if (tmp.empty() || tmp.top() == '(' || c == '(') {  // 如果符号栈为空，或栈顶为 ')'，或遇到左括号
                tmp.push(c);                                         // 则将该运算符入栈
            }
            else {
                while (!tmp.empty() && get_level(tmp.top()) >= get_level(c)) {  // 如果栈顶元素的优先级大于等于当前运算符，则弹出
                    if (tmp.top() == '(')  // （坑点 3：左括号的优先级是大于普通运算符的，但它不应该在这里弹出）
                        break;
                    ans.push(string(1, tmp.top()));
                    //ret << tmp.top() << ' ';
                    tmp.pop();
                }
                tmp.push(c);
            }
        }
    }

    if (num_flag) {             // 表达式的最后一个数字入栈
        ans.push(v);
        //ret << v << ' ';
    }

    while (!tmp.empty()) {      // 字符串处理完后，依次弹出栈中的运算符
        if (tmp.top() == '(')   // 这个判断好像是多余的
            tmp.pop();
        ans.push(string(1, tmp.top()));
        //ret << tmp.top() << ' ';
        tmp.pop();
    }

    //return ret.str();

    stringstream ret;
    while (!ans.empty()) {
        ret << ans.front() << ' ';
        ans.pop();
    }
    return ret.str();
}

void solve() {
    // 只测试了以下样例，如果有反例请告诉我

    cout << infix2suffix("12+(((23)+3)*4)-5") << endl;  // 12 23 3 + 4 * + 5 -
    cout << infix2suffix("1+1+1") << endl;              // 1 1 + 1 +
    cout << infix2suffix("(1+1+1)") << endl;            // 1 1 + 1 +
    cout << infix2suffix("1+(2-3)*4+10/5") << endl;     // 1 2 3 - 4 * + 10 5 / +
    cout << infix2suffix("az-(b+c/d)*e") << endl;       // az b c d / + e * -
}
```
