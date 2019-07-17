# Lecture 15: Matrix Factorization

## 1. Linear Network Hypothesis

推荐系统：

* 所拥有的数据：一些用户对一些电影的打分；
* 所需要学习的能力：预测某个用户对一个没有看过的电影的评分。

对第m个电影来说的数据集$\mathcal{D}_m​$为：
$$
\Big\{(\tilde{\mathbf{x}}_n = (n),\ y_n=r_{nm}):\ user\ n\ rated\ movie\ m\Big\}
$$
数据中的输入特征，就是一个简单的编号，例如：1126,5566,6211等——**抽象的特征(abstract feature)**。当然，不一定：①某个人(n)看过所有的电影；②某部电影(m)被所有人看过——因此，$r_{nm}$可能有一部分是缺失的。

**对分类特征(categorical feature)进行二元向量编码(binary vector encoding)**

一些常见的分类特征：

* ID
* 血型：A，B，AB，O
* 程序语言：Java，C，C++，Python……

目前所学的大多数模型都是在**数值特征**上操作的：

* 线性模型；
* 线性模型的延伸：类神经网络；
* 除了，决策树(RF，GBDT)，可以直接处理分类特征。

因此，需要进行编码，将分类特征转换为数值特征——encoding——其中，binary vector encoding：

* $A = [1\ 0\ 0\ 0]^T$
* $B = [0\ 1\ 0\ 0]^T$
* $AB = [0\ 0\ 1\ 0]^T$
* $O = [0\ 0\ 0\ 1]^T$

经过编码后，数据集变为：
$$
\Big\{(\tilde{\mathbf{x}}_n = BinaryVectorEncoding(n),\ y_n=r_{nm}):\ user\ n\ rated\ movie\ m\Big\}
$$
将所有电影的资料整合起来，$\mathcal{D}$：
$$
\Big\{(\tilde{\mathbf{x}}_n = BinaryVectorEncoding(n),\ y_n=[r_{n1}\ ?\ ?\ r_{n4}\ r_{n5}\ \cdots\ r_{nM}]^T)\Big\}
$$
idea：使用类神经网络进行特征萃取

![15-1](.\pic\15-1.png)

其中，$\mathbf{x}$是表示某个用户的稀疏向量，只有一个位置是1，其余是0；$\mathbf{y}$是该用户对于所有M个电影的评分向量，中间层就是萃取出的特征。由于输入向量中只有一个位置是1，因此可以忽略中间层的tanh变换：

![15-2](.\pic\15-2.png)

重命名权重矩阵：

* 第一层的所有权重：$\Big[w_{ni}^{(1)}\Big] = \mathbf{V}^T$，$\mathbf{V}^T$是$N × \tilde{d}​$
* 第二层的所有权重：$\Big[w_{im}^{(2)}\Big] = \mathbf{W}$，$\mathbf{W}$是$\tilde{d} × M​$

Hypothesis：
$$
h(\mathbf{x}) = \mathbf{W}^T\mathbf{V}\mathbf{x}
$$
对某一个用户，其输出为：
$$
h(\mathbf{x}_n) = \mathbf{W}^T\mathbf{V}\mathbf{x} = \mathbf{W}^T \mathbf{v}_n
$$
其中$\mathbf{v}_n​$是矩阵$\mathbf{V}​$的第n列。

接下来，就是学习$\mathbf{V}​$与$\mathbf{W}​$。

## 2. Basic Matrix Factorization

对第m部电影，线性模型为(取输出的第m个分量)：
$$
h_m(\mathbf{x}) = \mathbf{w}_m^T\mathbf{V}\mathbf{x} = \mathbf{w}_m^T\Phi(\mathbf{x})
$$
所有的电影模型，共用一个转换$\Phi$，将抽象的特征x转换成具体的描述。

对于所有$D_m$(第m个电影来说的数据集)，我们希望：
$$
r_{nm} = y_n \approx \mathbf{w}_m^T \mathbf{v}_n
$$
将求解模型转换为最佳化$E_{in}​$的问题：
$$
E_{in}(\{\mathbf{w}_m\},\{\mathbf{v}_n\}) = constant · \sum_{user\ n\ rated\ movie\ m} \Big(r_{nm} - \mathbf{w}_m^T \mathbf{v}_n \Big)^2
$$
解决上述最佳化问题，等价于通过最上面非常简单的两层线性网络，同时学到了两个东西：

* transform：$\mathbf{V}​$
* linear models：$\mathbf{W}$

**矩阵分解(Matrix Factorization)**
$$
r_{nm} \approx \mathbf{w}_m^T \mathbf{v}_n = \mathbf{v}_n^T \mathbf{w}_m
$$
也就是：
$$
\mathbf{R} \approx \mathbf{V}^T\mathbf{W}
$$
如下图：

![15-3](.\pic\15-3.png)

从一些已经知道的某些用户对某些电影的评分(known rating)出发，学习一些factors——$\mathbf{v}_n$和$\mathbf{w}_m$，最后能够用他们来做预测：

![15-4](.\pic\15-4.png)

现在，需要解决的最优化问题是：
$$
\underset{\mathbf{W},\mathbf{V}}{min}\ E_{in}(\{\mathbf{w}_m\},\{\mathbf{v}_n\}) = \sum_{m=1}^M \Big( \sum_{(\mathbf{x}_n, r_{nm}) \in \mathcal{D}_m}(r_{nm}-\mathbf{w}_m^T \mathbf{v}_n)^2 \Big)
$$
上述最优化问题有两组变量——alternating minimization：

* 当$\mathbf{v}_n$固定，最小化$\mathbf{w}_m$，等价于在电影m的数据集$\mathcal{D}_m$上最小化$E_{in}$——per-movie linear regression without $w_0​$；
* 当$\mathbf{w}_m$固定，最小化$\mathbf{v}_n$，等价于在用户m的数据集$\mathcal{D}_n$上最小化$E_{in}$——per-user linear regression without $v_0​$——对称的。

该算法被称为__alternating least squares __：

![15-5](.\pic\15-5.png)

与矩阵分解类似的是线性自编码器：前者常用于降维，后者常用于抽取隐藏的特征：

![15-6](.\pic\15-6.png)



## 3. Stochastic Gradient Descent

除了使用ALS算法，我们还可以考虑使用SGD解决下面的最优化问题。
$$
\underset{\mathbf{W},\mathbf{V}}{min}\ E_{in}(\{\mathbf{w}_m\},\{\mathbf{v}_n\}) = \sum_{m=1}^M \Big( \sum_{(\mathbf{x}_n, r_{nm}) \in \mathcal{D}_m}(r_{nm}-\mathbf{w}_m^T \mathbf{v}_n)^2 \Big)
$$
SGD：随机选择一个样本，然后用该样本错误衡量err的梯度来更新参数，迭代至收敛。优点是：

* 有效率；
* 易执行；
* 可以轻易地扩展到其他的err。

对某一笔资料的错误衡量：
$$
err(user\ n,\ movie\ m,\ rating\ r_{nm}) = (r_{nm} - \mathbf{w}_m^T \mathbf{v}_n)^2
$$
对所有的变数做偏微分，求得梯度，然后更新：变数是$\mathbf{v}_1, \mathbf{v}_2, \cdots, \mathbf{v}_N$，$\mathbf{w}_1, \mathbf{w}_2, \cdots, \mathbf{w}_M$

* $\nabla_{\mathbf{v}_n} err = -2(r_{nm}-\mathbf{w}^T_m \mathbf{v}_n)\mathbf{w}_m$
* $\nabla_{\mathbf{w}_m} err = -2(r_{nm}-\mathbf{w}^T_m \mathbf{v}_n)\mathbf{v}_n$
* 其他的向量不用更新，因为梯度是0。

又可以把要更新的两个向量的梯度写成：
$$
-(residual)(the\ other\ feature\ vector)
$$
综上，矩阵分解SGD算法为：

![15-7](.\pic\15-7.png)

适用于大数据。

## 4. Summary of Extraction Models





