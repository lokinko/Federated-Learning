# Federated Learning

>*人工智能（Artificial Intelligence, AI）进入以深度学习为主导的大数据时代，基于大数据的机器学习既推动了AI的蓬勃发展，也带来了一系列安全隐患。这些隐患来源于深度学习本身的学习机制，无论是在它的模型建造（训练）阶段，还是在模型推理和使用阶段。这些安全隐患如果被有意或无意地滥用，后果将十分严重。*
---

**联邦学习是一种 <font color=#B22222>隐私保护、数据本地存储与计算</font> 的机器学习算法。**

**联邦平均算法：**
* ___Server___ 端初始化模型参数：___initialize___ <a href="https://www.codecogs.com/eqnedit.php?latex=w_0" target="_blank"><img src="https://latex.codecogs.com/svg.latex?w_0" title="w_0" /></a>
* 每个更新轮次(___each round___):
    - 选取本轮参与的用户数：<a href="https://www.codecogs.com/eqnedit.php?latex=m\Leftarrow&space;max(C\cdot&space;K,1)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?m\Leftarrow&space;max(C\cdot&space;K,1)" title="m\Leftarrow max(C\cdot K,1)" /></a>
    - 将其打乱顺序为集合：<a href="https://www.codecogs.com/eqnedit.php?latex=S_t:(random\&space;set\&space;of&space;\&space;m\&space;clients)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?S_t:(random\&space;set\&space;of&space;\&space;m\&space;clients)" title="S_t:(random\ set\ of \ m\ clients)" /></a>
        * 对于每个用户 <a href="https://www.codecogs.com/eqnedit.php?latex=k&space;\in&space;S_t" target="_blank"><img src="https://latex.codecogs.com/svg.latex?k&space;\in&space;S_t" title="k \in S_t" /></a>，并行计算
            - <a href="https://www.codecogs.com/eqnedit.php?latex=w^k_{t&plus;1}\leftarrow&space;ClientUpdate(k,w_t)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?w^k_{t&plus;1}\leftarrow&space;ClientUpdate(k,w_t)" title="w^k_{t+1}\leftarrow ClientUpdate(k,w_t)" /></a>
        * <a href="https://www.codecogs.com/eqnedit.php?latex=w^k_{t&plus;1}&space;\leftarrow&space;\sum^K_{k=1}&space;\frac{n_k}{n}w_{t&plus;1}^k" target="_blank"><img src="https://latex.codecogs.com/svg.latex?w^k_{t&plus;1}&space;\leftarrow&space;\sum^K_{k=1}&space;\frac{n_k}{n}w_{t&plus;1}^k" title="w^k_{t+1} \leftarrow \sum^K_{k=1} \frac{n_k}{n}w_{t+1}^k" /></a>

* ___Client___ 端更新：___ClientUpdate(k,w_t)___
    * 将每个 ___client___ 的数据按照 ___batch_size___ 划分为 **B** 组.  <a href="https://www.codecogs.com/eqnedit.php?latex=\mathcal{B}\leftarrow(split\&space;\mathcal{P}_k\&space;into\&space;batches\&space;of\&space;size\&space;B)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\mathcal{B}\leftarrow(split\&space;\mathcal{P}_k\&space;into\&space;batches\&space;of\&space;size\&space;B)" title="\mathcal{B}\leftarrow(split\ \mathcal{P}_k\ into\ batches\ of\ size\ B)" /></a>
    * 每个 ___epoch___ 的每个 ___batch___ 更新一次本地权重. <a href="https://www.codecogs.com/eqnedit.php?latex=w\leftarrow&space;w-\eta&space;\cdot&space;\nabla&space;\mathbb{l}(w;b)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?w\leftarrow&space;w-\eta&space;\cdot&space;\nabla&space;\mathbb{l}(w;b)" title="w\leftarrow w-\eta \cdot \nabla \mathbb{l}(w;b)" /></a>
    * ___return w to server___

## 文献参考  
### 1. 文献综述
* [Advances and Open Problems in Federated Learning](https://arxiv.org/pdf/1912.04977.pdf)
* [Federated Machine Learning: Concept and Applications](https://arxiv.org/pdf/1902.04885.pdf)
* [Threats to Federated Learning: A Survey](https://arxiv.org/pdf/2003.02133.pdf)
* [A Survey on Federated Learning Systems: Vision, Hype and Reality for Data Privacy and Protection](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1907.09693v3.pdf)
* [Survey of Personalization Techniques for Federated Learning](https://arxiv.org/pdf/2003.08673.pdf)
  
### 2. Communication-Efficient & Converge fast
* [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/pdf/1602.05629.pdf) 
* [Federated Learning for Wireless Communications: Motivation, Opportunities and Challenges](https://arxiv.org/pdf/1908.06847v3.pdf)
### 3. 针对模型和数据隐私的攻击/保护
* [Practical Secure Aggregation for Privacy-Preserving Machine Learning](https://eprint.iacr.org/2017/281.pdf)
### 4. 数据分布不均、数据异构
### 5. 激励机制
* [Fair Resource Allocation in Federated Learning](https://arxiv.org/pdf/1905.10497v1.pdf)
### 6. 其他
* [Federated Adversarial Domain Adaptation](https://arxiv.org/abs/1911.02054)
* [Federated Learning with Matched Averaging](https://arxiv.org/abs/2002.06440)
* [Bayesian Nonparametric Federated Learning of Neural Networks](https://arxiv.org/abs/1905.12022v1)
