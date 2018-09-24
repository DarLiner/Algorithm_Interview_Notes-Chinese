C++-面向对象编程
===

面向对象编程概述
---
- 面向对象编程的两个主要特征：**继承**、**多态**
- C++ 中是通过间接利用“**指向父类**”的**指针**或**引用**来操作其子类对象，从而达到多态的目的；如果直接操作某个实例，那么多态将无从谈起
- 具体来说，**多态**是通过**动态绑定**机制，达到在**运行时**确定实际被调用的是哪个子类类型，进而调用对应的 **override** 方法

Reference
---
- 《Essential C++》 第 4/5 章 - Lippman, 侯捷

Index
---
<!-- TOC -->

- [不通过继承实现多态](#不通过继承实现多态)
- [抽象基类](#抽象基类)

<!-- /TOC -->

## 不通过继承实现多态
> 4.11 指针，指向 Class member function
- 核心想法：手工调整函数指针，模拟**动态绑定**的效果
  <details><summary><b>示例：动态序列（点击展开）</b></summary> 

  ```Cpp
  class num_sequence {
  public:
      // PtrType 是一个指针，指向 num_sequence 的成员函数，
      //  该成员函数必须只接受一个 int 型参数，以及返回类型为 void
      typedef void (num_sequence::*PtrType)(int);

      enum { cnt_seq = 2 };              // 预定义了两种序列
      enum ns_type {
          ns_fibonacci, ns_square
      };

      // 构造函数：默认指向斐波那契数列
      num_sequence(): _pmf(func_tbl[ns_fibonacci]) { }

      // 调整指针指向
      void set_sequence(ns_type nst) {
          switch (nst) {
          case ns_fibonacci: case ns_square:
              _pmf = func_tbl[nst];
              break;
          default:
              cerr << "invalid sequence type\n";
          }
      }

      void print(int n) {
          (this->*_pmf)(n); // 通过指针选择需要调用的函数
      }

      // _pmf 可以指向以下任何一个函数
      void fibonacci(int n) {
          int f = 1;
          int g = 1;
          for (int i = 2; i <= n; i++)
              g = g + f, f = g - f;
          cout << f << endl;
      }

      void square(int n) {
          cout << n * n << endl;
      }

  private:
      PtrType _pmf;
      static PtrType func_tbl[cnt_seq];  // 保存所有序列函数的指针
                                         // 为了兼容性，不推荐写成 `static vector<vector<int>没有空格> _seq;`
  };

  // static 成员变量初始化
  num_sequence::PtrType
  num_sequence::func_tbl[cnt_seq] = {
      &num_sequence::fibonacci,
      &num_sequence::square,
  };

  int main() {

      auto ns = num_sequence();
      ns.print(5);  // 5
      ns.set_sequence(num_sequence::ns_square);  // 调整函数指针以获得多态的效果
      ns.print(5);  // 25

      cout << endl;
      system("PAUSE");
      return 0;
  }
  ```

  </details>
  
  > [源文件](../code/cpp/面向对象-不通过继承实现多态-动态序列.cpp)

## 抽象基类
- 仅仅为了**设计**/**定义规范**而存在