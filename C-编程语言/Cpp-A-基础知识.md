**RoadMap**
---
<!-- TOC -->

- [指针与引用](#指针与引用)
  - [左值引用与右值引用](#左值引用与右值引用)
- [static 与 const](#static-与-const)
  - [const 相关代码](#const-相关代码)
- [this 指针](#this-指针)
- [inline 内联函数](#inline-内联函数)
  - [编译器对 inline 函数的处理步骤](#编译器对-inline-函数的处理步骤)
  - [inline 的优缺点](#inline-的优缺点)
  - [虚函数可以内联吗？](#虚函数可以内联吗)
- [assert 与 sizeof](#assert-与-sizeof)
- [C++ 中 struct、union、class](#c-中-structunionclass)
  - [C 与 C++ 中的结构体](#c-与-c-中的结构体)
  - [C++ 中 struct 和 class 的区别](#c-中-struct-和-class-的区别)
  - [联合体 union](#联合体-union)
- [用 C 实现 C++ 中的封装、继承和多态](#用-c-实现-c-中的封装继承和多态)
- [友元函数与友元类](#友元函数与友元类)
- [枚举类型 enum](#枚举类型-enum)
- [其他](#其他)
  - [#pragma pack(n)](#pragma-packn)
  - [位域 Bit mode](#位域-bit-mode)
  - [关键字 volatile](#关键字-volatile)
  - [关键字 extern "C"](#关键字-extern-c)
  - [关键字 explicit](#关键字-explicit)
  - [关键字 using](#关键字-using)
  - [范围解析运算符 ::](#范围解析运算符-)
  - [关键字 decltype](#关键字-decltype)

<!-- /TOC -->

## 指针与引用

### 左值引用与右值引用
> [C++专题-左值与右值](./Cpp-C-左值与右值.md)

## static 与 const
> https://github.com/huihut/interview#const

**static 作用**
1. 修饰普通变量，修改变量的存储区域和生命周期，使变量存储在静态区，在 main 函数运行前就分配了空间，如果有初始值就用初始值初始化它，如果没有初始值系统用默认值初始化它。
1. 修饰普通函数，表明函数的作用范围，仅在定义该函数的文件内才能使用。在多人开发项目时，为了防止与他人命令函数重名，可以将函数定位为 static。
1. 修饰成员变量，修饰成员变量使所有的对象只保存一个该变量，而且不需要生成对象就可以访问该成员。
1. 修饰成员函数，修饰成员函数使得不需要生成对象就可以访问该函数，但是在 static 函数内不能访问非静态成员。

**const 的作用**
1. 修饰变量，说明该变量不可以被改变；
1. 修饰指针，分为指向常量的指针和指针常量；
1. 常量引用，经常用于形参类型，即避免了拷贝，又避免了函数对值的修改；
1. 修饰成员函数，说明该成员函数内不能修改成员变量。

### const 相关代码
<!-- <details><summary><b>const 使用</b>（点击展开）</summary>  -->
<!-- </details> -->
```cpp
// 类
class A {
private:
    const int a;                // 常对象成员，只能在初始化列表赋值

public:
    // 构造函数
    A() { };
    A(int x) : a(x) { };        // 初始化列表赋值常对象成员

    // const 可用于对重载函数的区分
    int getValue();             // 普通成员函数
    int getValue() const;       // 常成员函数，不得修改类中的任何数据成员的值
    // 工程上为了安全性有时会分别实现 const 和 非const 两个版本
    // > 下标操作符为什么要定义const和非const两个版本？_百度知道 https://zhidao.baidu.com/question/517798128.html
};

// 全局函数
void function() {
    // 对象
    A b;                        // 普通对象，可以调用全部成员函数
    const A a;                  // 常对象，只能调用常成员函数、更新常成员变量
    const A *p = &a;            // 常指针，注意 a 是一个常对象
    const A &q = a;             // 常引用

    // 指针
    char greeting[] = "Hello";
    char* p1 = greeting;                // 指针变量，指向字符数组 变量
    const char* p2 = greeting;          // 指针变量，指向字符数组 常量
    char* const p3 = greeting;          // 常指针，指向字符数组 变量
    const char* const p4 = greeting;    // 常指针，指向字符数组 常量
}

// 函数
void function1(const int Var);           // 传递过来的参数在函数内不可变
void function2(const char* Var);         // 参数指针所指内容为常量
void function3(char* const Var);         // 参数指针为常指针
void function4(const int& Var);          // 引用参数在函数内为常量

// 函数返回值
const int function5();      // 返回一个常数
const int* function6();     // 返回一个指向常量的指针变量，使用：const int *p = function6();
int* const function7();     // 返回一个指向变量的常指针，使用：int* const p = function7();
```

## this 指针
> https://github.com/huihut/interview#this-指针

## inline 内联函数
> https://github.com/huihut/interview#inline-内联函数

**内联函数的特点**
- 相当于把内联函数中的内容复制到了调用该函数的地方（编译时完成）；
  - 相当于避免了函数调用的开销，直接执行函数体；
  - 加速了程序的运行，但是消耗了空间
- 相当于宏，但比宏多了**类型检查**，使具有函数特性；
- 不能包含循环、递归、switch 等复杂操作；
  - 如果包含了，那么相当于失去了内联的作用；
  - 内联只是对编译器的**建议**，是否内联取决于编译器
- 定义在类中的函数，除了虚函数，都会自动隐式地当成内联函数。
  > 但这不代表虚函数无法内联 > [虚函数可以内联吗？](#虚函数可以内联吗)
- 内联函数通常定义在头文件中
  - inline 函数对编译器而言必须是可见的，以便它能够在调用点展开该函数。
  - 如果定义在 cpp 文件中，那么在每个调用该内联函数的文件内都要再重新定义一次，且定义必须一致
  - inline 函数允许多次定义
<!-- - 在 C++ 中，inline 是**定义**时有效的关键字 -->

**`inline` 的使用**
```Cpp
// 定义在头文件中
inline 
int func() {}

// 声明在头文件中
inline 
int func1();
// 定义在源文件 1 中
inline
int func1() {}
// 定义在源文件 2 中
inline
int func1() {}
// 源文件 1 和 2 中的定义必须一致
```
- 关于 inline 关键字应该放在声明还是定义处，重说纷纭，保险起见都加上
- 当然，最好的做法是直接定义在头文件中

### 编译器对 inline 函数的处理步骤
- 将 inline 函数体复制到 inline 函数调用点处；
- 为所用 inline 函数中的局部变量分配内存空间；
- 将 inline 函数的的输入参数和返回值映射到调用方法的局部变量空间中；
- 如果 inline 函数有多个返回点，将其转变为 inline 函数代码块末尾的分支（使用 GOTO）。

### inline 的优缺点
**优点（相比宏定义）**
- 内联函数类似宏函数，在调用处展开，省去了**调用开销**（参数压栈、栈帧开辟与回收，结果返回等），从而提高程序运行速度。
- 相比宏函数来说，内联函数在代码展开时，会进行安全检查或自动类型转换（同普通函数），而宏定义不会。
- 在类中声明同时定义的成员函数，会自动转化为内联函数，因此内联函数可以访问类的成员变量，宏定义则不能。
- 内联函数在运行时可调试，宏定义不可以。

**缺点**
- 代码膨胀。
  - 内联是以代码膨胀（内存）为代价，来消除函数调用带来的开销。
  - 如果执行函数体内代码的时间，相比于函数调用的开销较大，那么效率的收获会很少。
- inline 函数无法随函数库升级。
  - inline 函数的改变需要重新编译，不像 non-inline 可以直接链接。
- 是否内联，程序员不可控。
  - 内联函数只是对编译器的建议，是否对函数内联，决定权在于编译器。

### 虚函数可以内联吗？
> [Are "inline virtual" member functions ever actually "inlined"?](http://www.cs.technion.ac.il/users/yechiel/c++-faq/inline-virtuals.html) - C++ FAQ 
- 虚函数可以是内联函数
- 但是当虚函数表现多态性时不能内联。
  - inline 是在编译期建议编译器内联，而虚函数的多态性在运行期，编译器无法知道运行期调用哪个代码，因此虚函数表现为多态性时（运行期）不可以内联。
- 虚函数**唯一可以内联的情况**是：
  - 编译器知道所调用的对象是哪个类，如 `Base::who()`，这只有在编译器具有实际对象而不是对象的指针或引用时才会发生。

**虚函数的内联**
```Cpp
class Base {
public:
    inline virtual 
    void who() {
        cout << "I am Base\n";
    }

    virtual ~Base() { }
};

class Derived : public Base {
public:
    inline void who() {  // 不写 inline 时也会隐式内联
        cout << "I am Derived\n";
    }
};

int main() {
    // 此处的虚函数 who()，是通过 Base类的具体对象 b来调用的，
    // 因此编译期间就能确定，所以它可以是内联的，但最终是否内联取决于编译器。
    Base b;
    b.who();

    // 此处的虚函数是通过指针调用的，呈现多态性，
    // 需要在运行时期间才能确定，所以不能为内联。
    Base *ptr = new Derived();
    ptr->who();

    // 因为 Base 有虚析构函数（virtual ~Base() {}），
    // 所以 delete 时，会先调用派生类（Derived）析构函数，
    // 再调用基类（Base）析构函数，防止内存泄漏。
    delete ptr;
    ptr = nullptr;

    system("pause");
    return 0;
}
```

## assert 与 sizeof
- 断言 `assert()` **是宏而非函数**，定义在头文件 `<assert.h>/<cassert>` 中
- `sizeof()` **是操作符而非函数** `sizeof int == sizeof(int)`
  - sizeof 对数组，得到整个数组所占空间大小。
  - sizeof 对指针，得到指针本身所占空间大小。
  - 特别的
    ```Cpp
    void foo(int arr[]) {  // 实际上是 int* arr
        cout << sizeof(arr) << endl;  // 得到的是指针的大小，而非数组的大小
    }
    ```


## C++ 中 struct、union、class

### C 与 C++ 中的结构体
- C 中的结构体
  ```C
  typedef 
  struct Student {
      int age; 
  } Stu;
  ```
  等价于
  ```C
  struct Student { 
      int age; 
  };
  typedef struct Student Stu;
  ```
  实际上就是为 `struct Student` 这个比较长的声明定义了一个别名 `Stu`
  ```C
  struct Student s1;
  Stu s2;
  ```
  因为 C 中定义结构体，必须带上 `struct`，所以还可以定义 `void Student() {}` 不冲突

- C++ 中的结构体
  ```Cpp
  typedef
  struct Student { 
      int age; 
  } Stu;

  Student s1;         // 正确，"struct" 关键字可省略
  struct Student s2;  // 正确
  Stu s3;             // 正确，使用别名

  void f( Student me );  // 正确
  ```
  如果定义了同名的函数，那么 struct 关键字不可省略
  ```Cpp
  typedef 
  struct Student { 
      int age; 
  } Stu;

  void Student() {}           // 正确，定义后名为 "Student" 的函数

  // void Stu() {}            // 错误，符号 "Stu" 已经被定义为一个 "struct Student" 的别名

  int main() {
      Student(); 
      struct Student s1;
      Stu s2;
      return 0;
  }
  ```

### C++ 中 struct 和 class 的区别
- 一般来说，struct 适合作为一个数据结构的实现体，class 更适合作为一个对象的实现体
- 但最本质的区别在于默认的**访问控制权限**
  - 默认的继承访问权限—— struct 是 public，class 是 private
  - 默认的成员访问权限—— struct 是 public，class 是 private

### 联合体 union
- **联合体**（union）是一种节省空间的特殊的类
  - 一个 union 可以有多个数据成员，但是在任意时刻只有一个数据成员可以有值
  - 当某个成员被赋值后其他成员变为**未定义状态**
- 联合体的特点
  - 默认访问控制符为 public
  - 可以含有构造函数、析构函数
  - 不能含有引用类型的成员
  - 不能继承自其他类，不能作为基类
  - 不能含有虚函数
  - 匿名 union 在定义所在作用域可直接访问 union 成员
  - 匿名 union 不能包含 protected 成员或 private 成员
  - 全局匿名联合必须是静态（static）的

```Cpp
union UnionTest { // 联合体
    UnionTest() : i(10) {};
    int i;
    double d;
};

static union {    // 全局静态匿名联合体
    int i;
    double d;
};

int main() {
    UnionTest u;

    union {       // 局部匿名联合体
        int i;
        double d;
    };

    std::cout << u.i << std::endl;  // 输出 UnionTest 联合的 10

    ::i = 20; // 匿名 union 在定义所在作用域可直接访问
    std::cout << ::i << std::endl;  // 输出全局静态匿名联合的 20

    i = 30;
    std::cout << i << std::endl;    // 输出局部匿名联合的 30

    return 0;
}
```


## 用 C 实现 C++ 中的封装、继承和多态
> 董的博客 » [C语言实现封装、继承和多态](http://dongxicheng.org/cpp/ooc/) 


## 友元函数与友元类
> [友元(友元函数、友元类和友元成员函数) C++](https://www.cnblogs.com/zhuguanhao/p/6286145.html) - zhuguanhao - 博客园

**友元小结**
- 能访问私有成员
- 破坏封装性
- 友元关系不可传递
- 友元关系的单向性
- 友元声明的形式及数量不受限制

**友元函数**
- 不是类的成员函数，却能访问该类所有成员（包括私有成员）的函数
- 类授予**它的友元函数**特别的访问权，这样该友元函数就能访问到类中的所有成员。
  ```Cpp
  class A {
  public:
      friend void set_data(int x, A &a);      // 友元函数的声明
      int get_data() { return data; }
  private:
      int data;
  };

  void set_data(int x, A &a) {                // 友元函数的定义
      a.data = x; 
      cout << a.data << endl;                 // 无障碍读写类的私有成员
  }

  int main(void) {
      class A a;
      set_data(1, a);
      // cout << a.data;  // err
      cout << a.get_data() << endl; 
      return 0;
  }
  ```

**友元类**
- 一个类的友元类可以访问该类的所有成员（包括私有成员）
- 
- 注意点
  - 友元关系不能被继承
  - 友元关系不能传递
  - 友元关系是单向的
```Cpp
class A {
public:
    friend class C;    // 友元类的声明：C 是 A 的友元类
private:
    int data;
};

class C {              // 友元类的定义，可以访问 A 中的成员
public:
    void set_A_data(int x, A &a) { 
        a.data = x; 
    }

    int get_A_data(A& a) {
      return a.data;
    }
};

int main(void) {
    class A a;
    class C c;

    c.set_A_data(1, a);
    cout << c.get_A_data(a) << endl;  // 1

    return 0;
}
```

**友元成员函数**
- 使类 B 中的成员函数成为类A的友元函数（但 B 不是 A 的友元类），这样只有类 B 的该成员函数可以访问类 A 的所有成员
- 不推荐使用，原因如下
- 注意声明与定义的顺序
  ```Cpp
  class A;    // 类 A 的声明，因为 A 的友元函数，即 B 的成员函数要用到 A，所以必须先声明类 A
              // 但此时还不能定义 A，因为 B 还没有定义，B 的成员函数就无从谈起
  class B {   // 类 B 的定义
  public:
      void set_A_data(int x, A &a);   // 类 A 的友元函数，同时是 B 的成员函数
                                      // 但只能声明，还不能定义，因为 A 还没有定义，访问 A 的成员就无从谈起
  };

  class A {
  public:
      friend void B::set_A_data(int x, A &a);   // 将 B 的成员函数声明为 A 的友元成员函数；所以必须先定义 B
  private:
      int data;
      void print_data() { cout << data << endl; }
  };

  void B::set_A_data(int x, A &a) {   // 只有在定义类 A 后才能定义该函数
      a.data = x;          // 访问 A 的私有成员变量
      a.print_data();      // 访问 A 的私有成员函数
  }

  int main(void) {
      class A a;
      class B b;

      b.set_A_data(1, a);

      return 0;
  }
  ```


## 枚举类型 enum
> [C++枚举类型](https://www.baidu.com/s?wd=Cpp枚举类型)_百度搜索 
- 限制作用域的枚举类型，使用关键字 `enum class`
  ```Cpp
  enum class open_modes { input, output, append };
  ```
- 不限作用域的枚举类型
  ```Cpp
  enum color { red, yellow, green };
  enum { floatPrec = 6, doublePrec = 10 };
  ```



## 其他

### #pragma pack(n)
> https://github.com/huihut/interview#pragma-packn

### 位域 Bit mode
> https://github.com/huihut/interview#位域

### 关键字 volatile 
> https://github.com/huihut/interview#volatile

### 关键字 extern "C"
> https://github.com/huihut/interview#extern-c

### 关键字 explicit 
> https://github.com/huihut/interview#explicit显式构造函数

### 关键字 using
> https://github.com/huihut/interview#using

### 范围解析运算符 ::
> https://github.com/huihut/interview#-范围解析运算符

### 关键字 decltype
> https://github.com/huihut/interview#decltype