---
title: 机器学习算法——决策树(DecisionTree)
date: 2017-11-23 04:52:51
tags:
- machine-learning
categories:
- study
---

## 基本概念
决策树是常见的机器学习算法，具有及其良好的可解释性，如同人的思维一样，比如决定今天时否出门，如果下雨，那么再看风大不大，如果又刮大风，那么就不出门：

<!--more-->

![demo](demo.png)



决策树按照实现方法的不同，分为CART、ID3、C4.5

决策树按照预测值y的类型，又可以分为

* 分类树：预测值y代表类别，是离散值
* 回归树：预测值y代表数值，如身高，为连续值


## 算法原理

**决策树的构造过程，主要包含三步**：

* 如何选择节点属性
* 如何分裂属性
* 达的终止条件，终止决策树构造

**选择属性**方法有很多，不同的决策树主要区别就在于属性选择的方法，Quora上针对此也有详细的回答：[what-are-the-differences-between-ID3-C4-5-and-CART](https://www.quora.com/What-are-the-differences-between-ID3-C4-5-and-CART)

* ID3——信息增益
	* 缺点：偏向于多值特征
* C4.5——信息增益率
* CART——Gini Impurity（分类）and Variance reduction（回归），具体可参见wiki


![compare](compare.webp)



**分裂属性**的方法相对固定，主要有三种不同情况：

1. 属性是离散值且不要求生成二叉决策树。此时用属性的每一个划分作为一个分支。
2. 属性是离散值且要求生成二叉决策树。此时使用属性划分的一个子集进行测试，按照“属于此子集”和“不属于此子集”分成两个分支。
3. 属性是连续值。此时确定一个值作为分裂点split_point，按照>split_point和<=split_point生成两个分支。

**决策树构造的终止条件**是什么？在两种情况下我们停止进一步对子集进行划分，其一是划分已经达的可以理想效果了，另外一种就是进一步划分收效甚微，不值得再继续了。用专业术语总结终止条件有以下几个：

* 子集的熵达的阈值
* 子集规模够小
* 进一步划分的增益小于阈值


上述决策树的介绍是分类树，回归树则有所不同：
* 属性选择：基于残差平方和（MSE），选择一个属性，按照该属性分类后的两部分，分别计算MSE，两部分MSE之和最小的属性，作为分类属性，将数据分为两部分
* 属性分裂：按照上述连续值分裂的方式进行即可
* 终止条件：与分类树类似
* 预测：取树节点中所有数据的均值，作为预测的Y值，残差的计算也是基于此Y值


## 优缺点总结

wiki：https://en.wikipedia.org/wiki/Decision_tree_learning

* 优点——预测复杂度为树深度
* 优点——可读性描述性好
* 缺点——学习时的计算量和内存消耗随着样本规模变大
* 缺点——连续数据难处理
* 缺点——多分类问题，准确率不够理想

## 实际使用trick

1. 属性为连续变量时，如何使用互信息选择

	* 方法A：假设属性符合某概率分布，通过概率密度函数计算
	* 方法B：先将D中元素按照特征属性排序，则每两个相邻元素的中间点可以看做潜在分裂点，从第一个潜在分裂点开始，分裂D并计算两个集合的期望信息，具有最小期望信息的点称为这个属性的最佳分裂点，其信息期望作为此属性的信息期望。

2. 属性用完，但仍有子集不是纯净集
	* 多数票决：以子集中的多数类别作为该叶子节点类别

3. 树深度不要过深：避免数据中的噪声和离群点导致的过拟合

	* 前剪枝：在构造过程中，当某个节点满足剪枝条件，则直接停止此分支的构造
	* 后剪枝：先构造完成完整的决策树，再通过某些条件遍历树进行剪枝

4. 剪枝策略

	* Pessimistic Error Pruning：悲观错误剪枝
	* Minimum Error Pruning：最小误差剪枝
	* Cost-Complexity Pruning：代价复杂剪枝
	* Error-Based Pruning：基于错误的剪枝，即对每一个节点，都用一组测试 数据集进行测试，如果分裂之后，能够降低错误率，再继续分裂为两棵子树，否则直接作为叶子节点。
	* Critical Value Pruning：关键值剪枝，就是设置信息Gain的阈值作为停止条件

## Reference

* wiki：[https://en.wikipedia.org/wiki/Decision_tree](https://en.wikipedia.org/wiki/Decision_tree)
* wiki（learning）：https://en.wikipedia.org/wiki/Decision_tree_learning
* github src：https://github.com/eriklindernoren/ML-From-Scratch

