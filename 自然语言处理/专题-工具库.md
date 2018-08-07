专题-工具库
===

Index
---
<!-- TOC -->

- [工具列表](#工具列表)
- [jieba 分词](#jieba-分词)
  - [分词](#分词)

<!-- /TOC -->

# 工具列表
- Stanford Core NLP
  - 语义分析
- NLTK
  - 分词（西文）、分句、读取语义树
  - 词干提取
- jieba
  - 中文分词、词性标注

# jieba 分词
> fxsjy/[jieba: 结巴中文分词](https://github.com/fxsjy/jieba) 

## 分词
**代码示例**
```Python
import jieba

# 全模式
seg_list = jieba.cut("我来到北京清华大学", cut_all=True)
print("【全模式】: " + "/ ".join(seg_list))  

# 精确模式（默认）
seg_list = jieba.cut("我来到北京清华大学", cut_all=False)
# seg_list = jieba.cut("他来到了网易杭研大厦")
print("【精确模式】: " + "/ ".join(seg_list))  
print(", ".join(seg_list))

# 新词识别


# 搜索引擎模式
seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")
print(", ".join(seg_list))
```