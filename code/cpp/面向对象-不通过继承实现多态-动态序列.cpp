#include <iostream>
using namespace std;

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

