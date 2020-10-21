# Federated Learning

>*人工智能（Artificial Intelligence, AI）进入以深度学习为主导的大数据时代，基于大数据的机器学习既推动了AI的蓬勃发展，也带来了一系列安全隐患。这些隐患来源于深度学习本身的学习机制，无论是在它的模型建造（训练）阶段，还是在模型推理和使用阶段。这些安全隐患如果被有意或无意地滥用，后果将十分严重。*
---

**联邦学习是一种 <font color=#B22222>隐私保护、数据本地存储与计算</font> 的机器学习算法。**

**联邦平均算法：**
* ___Server___ 端初始化模型参数：___initialize___ $w_0$
* 每个更新轮次(___each round___):
    - 选取本轮参与的用户数：<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mi>m</mi><mo stretchy="false">⇐</mo><mi>m</mi><mi>a</mi><mi>x</mi><mo stretchy="false">(</mo><mi>C</mi><mo>⋅</mo><mi>K</mi><mo>,</mo><mn>1</mn><mo stretchy="false">)</mo></math>
    - 将其打乱顺序为集合：<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><msub><mi>S</mi><mi>t</mi></msub><mo>:</mo><mo stretchy="false">(</mo><mi>r</mi><mi>a</mi><mi>n</mi><mi>d</mi><mi>o</mi><mi>m</mi><mtext>&nbsp;</mtext><mi>s</mi><mi>e</mi><mi>t</mi><mtext>&nbsp;</mtext><mi>o</mi><mi>f</mi><mtext>&nbsp;</mtext><mi>m</mi><mtext>&nbsp;</mtext><mi>c</mi><mi>l</mi><mi>i</mi><mi>e</mi><mi>n</mi><mi>t</mi><mi>s</mi><mo stretchy="false">)</mo></math>
        * 对于每个用户 <math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mi>k</mi><mo>∈</mo><msub><mi>S</mi><mi>t</mi></msub></math>，并行计算
            - <math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><msubsup><mi>w</mi><mrow><mi>t</mi><mo>+</mo><mn>1</mn></mrow><mi>k</mi></msubsup><mo stretchy="false">←</mo><mi>C</mi><mi>l</mi><mi>i</mi><mi>e</mi><mi>n</mi><mi>t</mi><mi>U</mi><mi>p</mi><mi>d</mi><mi>a</mi><mi>t</mi><mi>e</mi><mo stretchy="false">(</mo><mi>k</mi><mo>,</mo><msub><mi>w</mi><mi>t</mi></msub><mo stretchy="false">)</mo></math>
        * <math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><msubsup><mi>w</mi><mrow><mi>t</mi><mo>+</mo><mn>1</mn></mrow><mi>k</mi></msubsup><mo stretchy="false">←</mo><munderover><mo data-mjx-texclass="OP">∑</mo><mrow><mi>k</mi><mo>=</mo><mn>1</mn></mrow><mi>K</mi></munderover><mfrac><msub><mi>n</mi><mi>k</mi></msub><mi>n</mi></mfrac><msubsup><mi>w</mi><mrow><mi>t</mi><mo>+</mo><mn>1</mn></mrow><mi>k</mi></msubsup></math>

* ___Client___ 端更新：___ClientUpdate(k,w_t)___
    * 将每个 ___client___ 的数据按照 ___batch_size___ 划分为 **B** 组.  <math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi data-mjx-variant="-tex-calligraphic" mathvariant="script">B</mi></mrow><mo stretchy="false">←</mo><mo stretchy="false">(</mo><mi>s</mi><mi>p</mi><mi>l</mi><mi>i</mi><mi>t</mi><mtext>&nbsp;</mtext><msub><mrow><mi data-mjx-variant="-tex-calligraphic" mathvariant="script">P</mi></mrow><mi>k</mi></msub><mtext>&nbsp;</mtext><mi>i</mi><mi>n</mi><mi>t</mi><mi>o</mi><mtext>&nbsp;</mtext><mi>b</mi><mi>a</mi><mi>t</mi><mi>c</mi><mi>h</mi><mi>e</mi><mi>s</mi><mtext>&nbsp;</mtext><mi>o</mi><mi>f</mi><mtext>&nbsp;</mtext><mi>s</mi><mi>i</mi><mi>z</mi><mi>e</mi><mtext>&nbsp;</mtext><mi>B</mi><mo stretchy="false">)</mo></math>
    * 每个 ___epoch___ 的每个 ___batch___ 更新一次本地权重. <math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mi>w</mi><mo stretchy="false">←</mo><mi>w</mi><mo>−</mo><mi>η</mi><mo>⋅</mo><mi mathvariant="normal">∇</mi><mrow><mi data-mjx-variant="-tex-calligraphic" mathvariant="script">l</mi></mrow><mo stretchy="false">(</mo><mi>w</mi><mo>;</mo><mi>b</mi><mo stretchy="false">)</mo></math>
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
