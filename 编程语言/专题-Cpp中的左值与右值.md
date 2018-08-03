C++中的左值与右值
===
<!-- TOC -->

- [如何快速判断左值与右值](#如何快速判断左值与右值)
- [左值与右值的本质/含义](#左值与右值的本质含义)
- [`move()` 与 `forward()`](#move-与-forward)
- [Reference](#reference)

<!-- /TOC -->


## 如何快速判断左值与右值
- **能被 `&` 取地址的就是左值**
  ```
  int foo;               // foo 是一个左值
  cout << &foo;          // 可以被取地址
  foo = foo + 5;         // foo + 5 是一个左值
  cout << &(foo+5);      // err
  cout << &1;            // err
  ```
  - 多数常数、字符等字面量都是右值，但**字符串是左值**
  - 虽然**字符串字面量**是左值；但它是 **const 左值**（只读对象），所以也不能对它赋值
    ```Cpp
    cout << &'a';     // err: lvalue required as unary '&' operand
    cout << &"abc";   // OK
    "abc" = "cba";    // err: assignment of read-only location
    ```
    > 为什么字符串字面量是对象？——节省内存，同一份字符串字面量引用的是用一块内存
- 所有具名变量一定是左值，即使声明时使用的是右值引用；

  所有**临时变量**都是右值
  ```Cpp
  T foo() {
      T t;
      // ...
      return t; 
  }

  T&& tmp = foo();  // tmp 是右值引用，但它是一个左值
  ```


## 左值与右值的本质/含义
- 左值表示是“**对象**”（object），右值表示“**值**”（value）——**“对象”内存储着“值”**
- 左值 `->` 右值的转换可看做“读取对象的值”（reading the value of an object）
- 任何一个 C++ 程序都是围绕对象进行操作：1) 反复地操作各种类型的对象 2) 这些对象在运行时创建在明确的内存范围内 3) 这些对象内存储着值。
- 存在不属于任何对象的值（临时量），这些就是右值
  - 字面量 5 表示一个抽象的值，但它不属于任何对象；
  - 大多数字面量都是右值，**除了字符串**，但可以把字符串字面量当做右值处理
  - `x + y` 表示两个对象的和，但它也不存储在任何对象中


## `move()` 与 `forward()`
- `std::move()` 允许你把任何表达式当做**右值**处理
  - 把“对象”当做“值”来处理效率更高——比如在复制对象时，实际上在"move"内存中的内容，而不是"copy"它们
  ```Cpp
  widget foo() {  // 这里的返回值不是引用
      widget w;
      // ...
      return w;   // w 是一个左值，但会被当做右值处理，这里是隐式完成的（开启返回值优化）
  }
  ```
- `std::forward` 允许你在处理时，保留表达式作为“对象”（左值）或“值”（右值）的特性
  ```Cpp
  void wrapper(widget&& x) {
      some_op(std::forward<widget>(x));
  }
  ```
  - 如果传入的是左值，x 会被推断为 `widget&` 类型
  - 如果传入的是右值，x 会被推断为 `widget&&` 类型
  - 如果不使用 `forward()`，这两种情况下，x 都会被当做左值
  - 如果使用了 `forward()`，x 会被当做它原本的值类型（左值或右值）


## Reference
- [The lvalue/rvalue metaphor](https://josephmansfield.uk/articles/lvalue-rvalue-metaphor.html) — Joseph Mansfield
- [关于C++左值和右值区别有没有什么简单明了的规则可以一眼辨别？](https://www.zhihu.com/question/39846131/answer/85277628) - 知乎
- [从4行代码看右值引用 - qicosmos(江南)](https://www.cnblogs.com/qicosmos/p/4283455.html) - 博客园 
