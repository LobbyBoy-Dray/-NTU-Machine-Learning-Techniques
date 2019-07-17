# Lecture 14：Radial Basis Function Network

> 课件链接：[Hsuan-Tien Lin - radial basis function network](https://www.csie.ntu.edu.tw/~htlin/course/ml19spring/doc/214_handout.pdf)
>
> **Radial Basis Function Network(径向基函数网络)**
>
> * RBF Network Hypothesis：径向基函数网络的假说形式
> * RBF Network Learning：径向基函数网络的学习过程
> * k-Means Algorithm：k均值演算法
> * k-Means and RBF Network in Action：k均值与径向基函数网络的实践

## 一. RBF Network Hypothesis：径向基函数网络的假说形式

**回顾：Gaussian SVM——基于高斯核函数的支撑向量机**
$$
g_{SVM}(\mathbf{x}) = sign\Bigg(\sum_{SV}\alpha_ny_n exp(-\gamma||\mathbf{x}-\mathbf{x}_n||^2) + b\Bigg)
$$
Gaussian SVM的实质是在一个无线多维的空间内找一个large-margin的边界作为最优的分离超平面。从结果上看，Gaussian SVM是一堆高斯函数的线性组合，组合系数是$\alpha_ny_n$，而每个高斯函数的中心均为一个样本点(支撑向量)。

之前提到，高斯核又被称为径向基函数核，即Radial Basis Function Kernel。其中的radial是指函数值只与某个距离有关，该距离即输入样本点$\mathbf{x}$与中心点$\mathbf{x}_n$的距离；basis function是指将要把这些函数进行线性组合，在Gaussian SVM中组合系数为$\alpha_ny_n$。

**另一种看待Gaussian SVM回传函数的方式**

令：
$$
g_n(\mathbf{x}) = y_n exp(-\gamma||\mathbf{x}-\mathbf{x}_n||^2)
$$
那么Gaussian SVM可以写作：
$$
g_{SVM}(\mathbf{x}) = sign\Bigg(\sum_{SV}\alpha_n g_n(\mathbf{x}) + b\Bigg)
$$
即Gaussian SVM可以看做是线性聚合——linear aggregation，聚合的元素是一个个小g——radial hypotheses。每一个小g都依据一个样本点(支撑向量)定义，其含义是：**根据输入样本与支撑向量样本点的相似性(距离)决定"投票票数"的多少**。例如，某个小g是定义在y为-1的某个支撑向量上的；如果输入样本x与该支撑向量的距离很小，那么高斯函数那一项就会比较大，那么就会给y，即-1，乘到比较大的正数，导致该小g的输出是一个很负的负数，也可以理解为该g投了非常偏向-1类别的票——依据是小g认为该输入样本与其背后的支撑向量样本距离比较近，因此"应该很相似"。

综上，Gaussian SVM可以看做是一堆radial hypotheses的线性聚合。而**径向基函数(RBF)网络也是一堆radial hypotheses的线性聚合**。

![](.\pic\14-1.png)

由上图可以看出，RBF网络与神经网络的输出层是一样的，均为线性聚合。但它们的**隐藏层有着显著的区别**：神经网络隐藏层的每个单元在：①做内积②通过转换函数如tanh；而RBF网络的每个单元在：①计算距离②通过径向基函数如Gaussian。

**RBF网络的hypothesis**
$$
h(\mathbf{x}) = Output\Bigg(\sum_{m=1}^M \beta_m RBF(\mathbf{x},\mathbf{\mu}_m) + b\Bigg)
$$
b可以暂时略去。因此，需要决定的参数有：

* （M：有多少个单元）；
* （RBF：径向基函数的选择）；
* （Output：取决于面向的问题）；

* $\mathbf{\mu}_m$：每个单元的中心(center)；
* $\beta_m$：每个单元的投票权重。

用上述hypothesis的观点看Gaussian SVM：

* M是支撑向量的数量；
* RBF是高斯函数；
* Output是sign，因为在做二元分类问题；
* 每个单元的中心即为每个支撑向量；
* 每个单元的投票权重为$\alpha_n y_n$。

学习RBF网络的过程，就是在RBF、Output给定的情形下，决定$\mathbf{\mu}_m$与$\beta_m$。

**关于相似性(Similarity)的解释**

在SVM相关章节中我们提到，kernel其实在描述两个样本点的相似性，这种描述是通过"偷吃步"的方法计算Z空间中的内积完成的。但并不是所有描述两个样本点相似性的函数都是合法的kernel——合法的kernel需要满足Mercer’s condition。

在刚刚，我们又接触了一种描述两个样本点的相似性的工具——RBF，即径向基函数。不同于kernel通过计算Z空间内积的方式，RBF透过X空间的距离(的函数)来衡量两个样本点的相似性。因为距离越近，往往相似性越大，因此RBF函数往往是关于距离单调递减的。

当然，衡量相似性的方式不只有kernel与RBF，还有其他的一些函数。它们的关系如下图所示。其中，高斯函数既属于kernel类，也属于RBF类：

![](.\pic\14-2.png)

## 二. RBF Network Learning：径向基函数网络的学习过程

**完全RBF网络：Full RBF Network**
$$
h(\mathbf{x}) = Output\Bigg(\sum_{m=1}^M \beta_m RBF(\mathbf{x},\mathbf{\mu}_m)\Bigg)
$$
完全RBF网络是指M=N且$\mu_m = \mathbf{x}_m$，即分别以每一个训练样本点为center构造RBF函数。这种做法的含义是：每一个训练样本点都会对输入的预测产生影响，影响的效力用$\beta_m$体现。例如，我们考虑所有训练样本的相似性的投票权相同，即取$\beta_m = 1·y_m$，对于二元分类问题：
$$
g_{uniform}(\mathbf{x}) = sign\Bigg(\sum_{m=1}^N y_m exp(-\gamma||\mathbf{x}-\mathbf{x}_m||^2)\Bigg)
$$
对于该"均匀"的RBF网络，由于每个训练样本的投票权相同，因此离输入$\mathbf{x}$最近的那个点很有可能就直接决定了预测结果——**因为离输入最近的点RBF函数值最大，而高斯函数又是衰退很快的函数**。因此，我们可能不需要计算上式中的求和项，而是直接找到离输入$\mathbf{x}$最近的那**一个**训练样本点，然后用它的标签作为预测结果：
$$
g_{nbor}(\mathbf{x}) = y_m\ such\ that\ \mathbf{x}\ closest\ to\ \mathbf{x}_m
$$
这样寻找"最近邻"的方法，称为**nearest neighbor model**。最近邻方法也可以进行延伸，得到**k近邻模型**——找离输入点最近的k个训练样本，然后做均匀投票动作。近邻模型是一种偷懒的方法，似乎并没有进行训练，只是在进行预测的时候才开始计算每个训练样本与输入点的相似性(距离)。





最佳化$\beta_m5$









## 三. k-Means Algorithm：k均值演算法









## 四. k-Means and RBF Network in Action：k均值与径向基函数网络的实践











