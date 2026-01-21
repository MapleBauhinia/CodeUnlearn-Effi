# 面向大模型生成代码的效率提升方法  
**Improving the Runtime Efficiency of LLM-Generated Code via Code Behavior Modeling and Machine Unlearning**

---

## 项目背景与动机

随着大语言模型（Large Language Models, LLMs）在程序生成任务中的广泛应用，其生成代码在**功能正确性**方面已取得显著进展。然而，大量研究与实践表明，LLM-generated code 在**执行效率**（如运行时间、内存占用）层面仍存在系统性不足，尤其体现在算法选择、数据结构使用及语言特性利用等方面。

本项目旨在从**代码行为（Code Behavior）**这一可解释视角出发，系统性刻画并消除大模型生成代码中的低效模式，探索一种**可控、可扩展、可解释**的代码效率提升范式。

---

## 项目简介

本项目提出了一种结合**代码行为建模（Code Behavior Modeling）**与**机器遗忘学习（Machine Unlearning）**的统一框架，用于提升大模型生成代码在以下指标上的表现：

- 平均执行时间（Runtime）
- 峰值与平均内存开销（Memory Usage）
- 算法复杂度与可扩展性（Scalability）

核心思想在于：  
> **显式识别并“遗忘”低效代码行为，同时系统性强化高效代码行为的生成概率。**

---

## 整体技术路线概览

项目整体流程如下：

1. 构建 **低效 / 高效代码行为分类体系（Taxonomy）**
2. 收集并运行大规模真实代码实现，进行效率评测
3. 构建低效–高效代码对，并进行细粒度行为标注
4. 基于机器遗忘学习设计三阶段训练策略
5. 引入静态与动态分析信号，引导模型规避低效行为

---

## 代码行为建模（Code Behavior Modeling）

### 1. 低效代码行为定义与分类

通过系统调研大模型代码生成、代码效率评测基准（如 EffiBench）以及程序分析领域的相关研究，我们提出了**大模型生成代码低效行为分类体系**：

### **Taxonomy 1: Inefficient Code Behaviors**

#### 1. Inefficient Function or API Usage
- Suboptimal method or API selection  
- Unnecessary or excessive recursion  

#### 2. Algorithmic Inefficiencies
- Brute-force or suboptimal algorithmic strategies  
- Absence of established optimizations (e.g., early exit, pruning, space–time trade-offs)  
- Insufficient mathematical abstraction  
- Inefficient conditional logic  
- Avoidable nested-loop complexity  
- Unnecessary multi-pass processing  
- Redundant recomputation  

#### 3. Inefficient Data Structure Usage
- Inappropriate data structure selection  
  - e.g., list instead of set for membership testing  
  - list instead of deque for queue operations  
- Inefficient operations on chosen structures  
- Inefficient string concatenation (e.g., `s += char` causing O(n²))  
- Repeated slicing in loops  
- Redundant data creation, duplication, or conversion  

#### 4. Underutilization of Language-Specific Features
- Failure to leverage built-in functions or libraries  
- Lack of idiomatic constructs  
  - e.g., Python comprehensions, generators  

#### 5. Memory Inefficiencies
- Unnecessary buffering or intermediate storage  
- Creation of large or avoidable temporary objects  

#### 6. Other Inefficiencies
- Lack of input-scale awareness  
- Inefficient I/O handling  
- Inefficient exception-handling patterns  
- Redundant or dead code constructs  

---

### 2. 高效代码行为分类体系

在低效行为分析的基础上，我们进一步构建了**对偶的高效代码行为体系**，用于后续正向强化学习：

### **Taxonomy 2: Efficient Code Behaviors**

#### 1. Function or API Optimizations
- Optimal API or method selection  
- Avoidance of unnecessary or deep recursion  

#### 2. Algorithmic Optimizations
- Replacement with efficient algorithms  
  - e.g., two-pointer, divide-and-conquer, greedy, dynamic programming  
- Use of standard optimization techniques  
- Mathematical abstraction and formula-based optimization  
- Optimized conditional logic  
- Multi-pass → single-pass transformation  
- Elimination of redundant recomputation  

#### 3. Data Structure Optimizations
- Proper data structure selection  
  - e.g., hash/set/dict, heap for top-k  
- Efficient structure-specific operations  
- Efficient string handling  
- In-place modification when feasible  

#### 4. Language-Specific Feature Utilization
- Effective use of built-in libraries  
- Idiomatic language constructs  

#### 5. Memory Optimizations
- Fixed-size or bounded buffers  
- Avoidance of unnecessary intermediates  
- Streaming or chunk-based processing  
- Preallocation or object reuse  

#### 6. Other Optimizations
- Input-scale–aware guarding  
- Efficient I/O processing  
- Robust yet lightweight exception handling  

---

## 代码数据集构建与效率评测

### 1. 数据集收集

- 数据来源：LeetCode 第 1–3000 道算法与数据结构题目  
- 数据内容：
  - 题目描述
  - 官方测试用例
  - 多种 Python 代码实现
- 数据清洗：
  - 主动避开 EffiBench 中已包含题目，防止评测基准污染  

### 2. 代码运行与效率评估

- 在隔离的 Python 沙箱环境中执行所有代码实现
- 记录：
  - 平均执行时间
  - 内存使用情况
- 去除：
  - 噪声过大的异常实现
- 对同一题目的多种实现进行**效率排名**

### 3. 低效–高效代码对构建

- 针对每道题目，设计**基于百分位的采样策略**
- 从效率排序结果中构建多个：
  - 低效代码 × 高效代码 配对
- 保证：
  - 行为差异显著
  - 功能语义一致  

---

## 代码行为标注

- 使用当前代码生成能力较强的模型 **Claude Sonnet 4.5**
- 对低效–高效代码对进行自动化行为标注
- 标注内容包括：
  - 具体代码行为类别
  - 行为解释（Explanation）
  - 成因机制分析（Mechanism）
  - 综合效率影响总结（Summary）
- 随后进行人工审核与一致性校验，确保标注质量  

---

## 基于机器遗忘学习的效率提升方法

### 整体训练策略（三阶段）

#### 阶段一：低效行为遗忘（Unlearning Phase）
- 数据：低效代码数据集（覆盖 Taxonomy 1）
- 方法：
  - 针对不同行为类型，设计差异化遗忘损失函数
  - 强约束遗忘目标
  - 融合 explanation / mechanism 等附属信息  

#### 阶段二：能力保持（Capability Preservation）
- 数据：简单、通用的编程任务
- 方法：
  - KL Divergence 正则
  - 防止灾难性遗忘
  - 保护通用代码生成能力  

#### 阶段三：高效行为强化（Reinforcement Phase）
- 数据：高效代码数据集（覆盖 Taxonomy 2）
- 方法：
  - 逐 token 交叉熵或强化学习
  - 提升高效行为的生成概率  

---

## 低效行为引导机制（进行中）

我们正在探索引入**程序分析信号**以进一步增强行为控制能力：

### 策略1：
在低效代码数据集上，针对不同的代码行为使用不同的**机器遗忘学习损失函数**，应用损失函数 (强约束) + 附属信息(explanation、mechanism 等深入分析信息)，纠正低效行为。

### 策略2：
引入基于 AST 等辅助结构的静态检测和基于 GraphCodeBERT 的动态检测，对生成代码进行低效行为评分，引导大模型避开低效行为。

- 静态分析：
  - AST-based pattern detection
- 动态分析：
  - GraphCodeBERT / learned efficiency detector
- 训练方式：
  - Policy Gradient
  - 行为评分函数 **Bc(y)**

**训练流程示意：**
θ_generate → y
→ AST Static / Model Dynamic Detector
→ Inefficiency Score Bc(y)
→ Policy Gradient
→ log pθ(y|x)
→ θ


---

## 当前状态

- [✅] 低效 / 高效代码行为分类体系构建  
- [✅] 大规模代码收集与效率评测  
- [✅] 低效–高效代码对与行为标注  
- [❓] 机器遗忘学习算法设计与验证（进行中）  

---

## 展望

本项目致力于推动以下研究方向的发展：

- 面向 **效率与可解释性** 的代码生成模型训练范式
- LLM-generated code 的行为级安全性与质量控制
- 程序分析与大模型训练的深度融合

欢迎对 **Code Intelligence / Program Analysis / Trustworthy AI** 感兴趣的研究者交流与合作。

