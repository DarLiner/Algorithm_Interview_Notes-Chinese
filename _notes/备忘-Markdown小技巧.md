备忘-Markdown小技巧
===

Index
---
<!-- TOC -->

- [自动更新目录](#自动更新目录)
- [图片居中](#图片居中)
- [隐藏代码块](#隐藏代码块)
- [HTML 表格](#html-表格)
- [Latex 公式](#latex-公式)

<!-- /TOC -->

## 自动更新目录
- VSCode 插件 [`Markdown TOC`](https://marketplace.visualstudio.com/items?itemName=AlanWalk.markdown-toc)

## 图片居中
- 不带链接
    ```
    <div align="center"><img src="../_assets/公式_20180717114628.png" height="" /></div>
    ```
- 带链接
    ```
    <div align="center"><a href=""><img src="../_assets/公式_20180717114628.png" height="" /></a></div>
    ```
- `height=""`用于控制图片的大小

## 隐藏代码块
```
<details><summary><b>示例：动态序列（点击展开）</b></summary> 

// 代码块，注意上下都要保留空行

</details>
```

## HTML 表格
```
<table style="width:100%; table-layout:fixed;">
    <tr>
        <td>No padding, no strides</td>
        <td>Arbitrary padding, no strides</td>
        <td>Half padding, no strides</td>
        <td>Full padding, no strides</td>
    </tr>
    <tr>
        <td><img width="150px" src="../_assets/no_padding_no_strides.gif"></td>
        <td><img width="150px" src="../_assets/arbitrary_padding_no_strides.gif"></td>
        <td><img width="150px" src="../_assets/same_padding_no_strides.gif"></td>
        <td><img width="150px" src="../_assets/full_padding_no_strides.gif"></td>
    </tr>
    <tr>
        <td>No padding, strides</td>
        <td>Padding, strides</td>
        <td>Padding, strides (odd)</td>
    </tr>
    <tr>
        <td><img width="150px" src="../_assets/no_padding_strides.gif"></td>
        <td><img width="150px" src="../_assets/padding_strides.gif"></td>
        <td><img width="150px" src="../_assets/padding_strides_odd.gif"></td>
    </tr>
</table>
```

## Latex 公式
> 在线 LaTeX 公式编辑器 http://www.codecogs.com/latex/eqneditor.php

**斜体加粗**
```
\boldsymbol{x}
```
**期望**
```
\mathbb{E}
```
**矩阵对齐**
```
\begin{array}{ll}
 & \\
 & \\
\end{array}
```
**转置**
```
^\mathsf{T}
```
**省略号**
```
水平方向    \cdots   
竖直方向    \vdots   
对角线方向  \ddots
```
**按元素相乘**
```
\circ
或
\odot
```