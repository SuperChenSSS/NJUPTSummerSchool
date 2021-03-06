### Day 6: 讲座报告

#### 游戏人工智能：陈兴国博士

![](img/Day6/IMG_0508.JPG)

游戏的四个特性：

- 有趣
- 不确定
- 规则
- 虚构

游戏分类：

- 完全博弈：棋、AlphaGo
- 非完全博弈：卡牌游戏

人数角度：

- 个人游戏：俄罗斯方块，2048等
- 多人游戏：中国象棋，卡牌等

游戏的复杂度：

![](img/Day6/IMG_0510.JPG)

- 算法的复杂度：
  - P：在多项式时间内可以完成的问题
  - NP：在多项式时间内非确定性可以完成的问题
  - NP-Complete
  - NP-Hard
- 游戏的复杂度
  - 俄罗斯方块的NP-Complete问题

**P = NP?**

Cook's theorem

- SAT问题的求解
- NP中最困难的问题可转换为P问题

经典问题：

- 欧拉回路：给定一个图，从某一点出发每条边走且仅走一次，最后回到出发点
  - 充要条件：连通、所有节点度数为偶数
  - Floyd算法
- Hamilton回路
  - 给定一个图，问能否找一条经过每个顶点一次且恰好一次，最后走回来的路
  - NP-Complete问题
- 3-Partition
- TSP问题：旅行商问题
  - 给定所有顶点和距离，问访问所有节点的最短路径路线
  - NP-Hard问题

俄罗斯方块：

- 七种方块等概率出现
- 以1的概率失败
  - 轮流掉落
- NP-Complete问题的解决：
  - 消除一行，有时候很简单，有时候很难
  - 求解算法的目标：难
    - 给定初始形状
    - 给定顺序
- 找到一个带变量n的问题
- 给定一个解（下落位置和旋转的序列）
  - NP
- 利用三分问题求解，再进行求和操作

游戏的复杂度

- 算法复杂度
- NP-Hard问题

玩游戏的人工智能

- 形式化
- 马尔科夫博弈
  - MDP流程：Agent 环境

流程如图所示：

![](img/Day6/IMG_0511.JPG)

游戏怎么玩：

- 看得远
- 看得准

搜索：

- min-max搜索

![](img/Day6/IMG_0513.JPG)

核心思想：对手选对自己最不利的，我方选对自己最有利的

- 一个最差情况的解

  - 在零项和情况下，退化成纳什均衡


- $\alpha - \beta $ 剪枝搜索

把一些不可能的情况剪枝掉

如何看得准：

- 评估函数：蒙特卡罗算法

![](img/Day6/IMG_0514.JPG)

**机器学习分类（反馈）**

- 监督学习
- 半监督学习
- 无监督学习
- 强化学习、游戏

训练的数据：

- 监督学习中样例Instance：<x,y>
- MDP中的数据
  - 一步< V(s) ,r + rV(s') >
  - 目标最小化MSE = $\sum_{S}(r+rV(s') - V(s))^2$

总结：

- 游戏的复杂度
- 玩游戏的人工智能
  - 模型
  - 常见思路

展望：

- 更高效、更优的解
- 如何游戏开发中提高效率
- 智能算法与传统文化

#### 分布式数据库——大数据管理的基础支撑：韩京宇教授

![](img/Day6/IMG_0515.JPG)

- 基础概念
- 数据分布技术
- 查询优化技术
- 事务提交技术
- 技术发展

基础概念：

两种类型：物理上分布，逻辑上集中；物理上分布，逻辑上分布

DDS主要特点：

- 数据是分布的
- 分布的数据是相关的
- 由DDBMS统一管理

物理分布、逻辑分布的数据库：

1. 实现现存的，分布的，异构的多个数据集合融合应用
2. 各个节点独立自治，且协商合作
3. 无全局模式

数据分布方式：

1. 划分式
2. 重复式
3. 混合式

关系数据库的数据分布单位：

- 水平分割
- 垂直分割

要求：

1. 完备性
2. 可重构性
3. 不相交性

分布式查询优化：

- IO代价
- CPU代价

![](img/Day6/IMG_0516.JPG)