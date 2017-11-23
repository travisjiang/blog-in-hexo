---
title: 机器学习算法——GBDT
date: 2017-11-23 23:07:56
tags:
- machine-learning
categories:
- study
---


## 基本概念

GBDT(Gradient Boosting Decision Tree) 又叫 MART（Multiple Additive Regression Tree)，是一种迭代的决策树算法，该算法由多棵决策树组成，所有树的结论累加起来做最终答案。它在被提出之初就和SVM一起被认为是泛化能力较强的算法。

<!--more-->

GBDT与随机森林都是由Decision Tree组合而成，区别是什么？

* GBDT中的基决策树是回归树，GBDT是用boosting方法ensemble所有的基决策树，必须是串行的训练各个基决策树
* 随机森林是用bagging方法ensemble所有基决策树，相当于并行训练多个决策树，再采用投票的方式决定最后的预测结果

## 算法原理

GBDT的核心概念有三个，理解这三个概念就一定能搞懂GBDT：

* Regression Decision Tree（回归树）
* Gradient Boosting（梯度提升）
* Shrinkage（收缩）

这里只总结一下每个概念最关键的点，详细介绍参考Reference中的推荐链接。

### Regression Decision Tree（回归树）

回归树与分类树类似，每个节点的预测值为节点包含所有样本的均值。衡量节点的标准则由分类树中的最大熵变为残差平方和：

$$ MSE = \sum_{i\in treeleft}{(y_{1i}-y'_1)^2} + \sum_{j\in treeright}{(y_{2j} - y'_2)^2}$$

其中$y'_1$表示左子树的预测值，$y'_2$表示右子节点的预测值，选取使MSE最小的属性作为节点分裂的属性

![Regression DT example](example1.png)



### Gradient Boosting（梯度提升）

如何组合多颗回归树来得到要一个ensemble的模型？

Boosting是组合（ensemble）多个弱分类器，从而得到 一个强分类器的方式，所谓“三个臭皮匠顶个诸葛亮”嘛！

其基本思想是，每一个弱分类器都基于先与它训练的上一个弱分类器，调整分类器的训练参数，使用同样数据集训练得出，常见Boosting方式有两种：

* AdaBoosting
	* 计算弱分类器的MSE，**基于MSE，调整训练样本的比重**，使分类错误的样本在新的MSE计算中占比更重，从而训练新的弱分类器时会更加重视这些样本；
	* 累加所有弱分类器的结果，得到最终结果，而**每一个分类器累加时的权重，也是由该分类器的MSE计算得到**
* Gradient Boosting
	* 不调整训练样本的比重，也不给每个回归树设置不同的权重
	* 调整的是训练的目标值$y_n$：假设真实值为y，第n-1个回归树对训练集X的预测结果为$y'_{n-1}$；则训练第n个回归树时，预测的目标不是y，而是残差（残差=真实值-预测值），也就是**是前n-1个分类器预测值之和，与真实值y的残差**：$y-\sum^{n-1}_{i=1}{y'_i}$


![Gradient Boosting example](example2.png)



* 如上图，假设训练集是A、B、C、D，目标是预测其年龄14，16，24,26
* 图中第一棵树的预测值目标为真实年龄；第二棵树的预测目标则变为真实年龄与第一棵树预测值的残差。
* 最终累加所有回归树的预测值，就是GBDT模型的预测值

### Shrinkage

Shrinkage的思想是：每次走一小步逐渐逼近结果的效果，要比每次迈一大步很快逼近结果的方式更容易避免过拟合。因此它不相信每一个回归树的预测结果，而是仅仅采用每棵树预测结果的一小部分，累加到最终预测值。这样可以训练更多回归树。

公式对比更清晰，假设$y'_n$表示第n棵树的实际预测值，$y_n$表示第n棵树的目标预测值，y为样本真实值，就是第n+1棵树的预测目标：

* 不使用Shrinkage时：
	* $y_{n+1} = y-\sum^n_{i=1}{y'_i}$
* 使用Shrinkage时：
	* $y_{n+1} = y-\sum^n_{i=1}{y'_i * step}$

step一般比较小，如0.01~0.001，有点类似学习率的意思，让学习过程是缓慢逼近而不是陡变的。

## 实际应用

1. GBDT使用几乎所有回归问题（线性/非线性），也可以用于分类问题（设定阈值，大于阈值是正例，反之是负例）
2. 用作特征组合与特征选择：CTR广告预估中，先训练一个GBDT模型，然后GBDT中**每个树的每个叶节点的路径**对应的特征组合（或者特征），就可以作为关键特征，从而达到特征工程的目的。

![GBDT and LR](example3.png)

3. 用于搜索排序，RankNet

## FAQ

为什么GBDT的每个树深度为6就已经可以达到很高的精度，而随机森林与单棵决策树算法，高精度需要树深度很大？


* 对于Bagging算法来说，由于我们会并行地训练很多不同的分类器的目的就是降低这个方差(variance) ，因为采用了相互独立的基分类器多了以后，h的值自然就会靠近。所以对于每个基分类器来说，目标就是如何降低这个偏差（bias),所以我们会采用深度很深甚至不剪枝的决策树。
* 对于Boosting来说，每一步我们都会在上一轮的基础上更加拟合原数据，所以模型本身可以保证低偏差（bias），所以对于每个基分类器来说，问题就在于如何选择variance更小的分类器，即更简单的分类器，所以我们选择了深度很浅的决策树。
* GBDT作为boosting方法，本身可以保证偏差，所以对于每个基分类器，主要目标是降低方差，从树深度一般不超过6




## Reference

* http://blog.csdn.net/suranxu007/article/details/49910323（推荐，讲的比本文清楚）
* http://blog.csdn.net/lilyth_lilyth/article/details/48032119（CRT预估中GBDT作为特征提取，与LR融合的方案）
* http://www.jianshu.com/p/005a4e6ac775
* https://en.wikipedia.org/wiki/Gradient_boosting#Gradient_tree_boosting


