# Federated Learning

>*人工智能（Artificial Intelligence, AI）进入以深度学习为主导的大数据时代，基于大数据的机器学习既推动了AI的蓬勃发展，也带来了一系列安全隐患。这些隐患来源于深度学习本身的学习机制，无论是在它的模型建造（训练）阶段，还是在模型推理和使用阶段。这些安全隐患如果被有意或无意地滥用，后果将十分严重。*

---

**联邦学习是一种 <font color=#B22222>隐私保护、数据本地存储与计算</font> 的机器学习算法。**

# 文献参考


## Introduction
* [Federated Learning Comic](https://federated.withgoogle.com/)
* [Federated Learning: Collaborative Machine Learning without Centralized Training Data ](http://ai.googleblog.com/2017/04/federated-learning-collaborative.html)
* [GDPR, Data Shotrage and AI (AAAI-19)](https://aaai.org/Conferences/AAAI-19/invited-speakers/#yang)
* [Federated Learning: Machine Learning on Decentralized Data (Google I/O'19)](https://www.youtube.com/watch?v=89BGjQYA0uE)
* [Federated Learning White Paper V1.0](https://www.fedai.org/static/flwp-en.pdf)
* [Federated learning: distributed machine learning with data locality and privacy](https://blog.fastforwardlabs.com/2018/11/14/federated-learning.html)

## Survey
* [Federated Learning: Challenges, Methods, and Future Directions](https://arxiv.org/abs/1908.07873)
* [Federated Learning Systems: Vision, Hype and Reality for Data Privacy and Protection](https://arxiv.org/abs/1907.09693)
* [Federated Learning in Mobile Edge Networks: A Comprehensive Survey](https://arxiv.org/abs/1909.11875)
* [Federated Learning for Wireless Communications: Motivation, Opportunities and Challenges](https://arxiv.org/abs/1908.06847)
* [Convergence of Edge Computing and Deep Learning: A Comprehensive Survey](https://arxiv.org/pdf/1907.08349.pdf)
* [Advances and Open Problems in Federated Learning](https://arxiv.org/pdf/1912.04977.pdf)
* [Federated Machine Learning: Concept and Applications](https://arxiv.org/pdf/1902.04885.pdf)
* [Threats to Federated Learning: A Survey](https://arxiv.org/pdf/2003.02133.pdf)
* [Survey of Personalization Techniques for Federated Learning](https://arxiv.org/pdf/2003.08673.pdf)

## Benchmarks
* [LEAF: A Benchmark for Federated Settings](https://arxiv.org/abs/1812.01097)](https://github.com/TalwalkarLab/leaf) [Recommend]
* [A Performance Evaluation of Federated Learning Algorithms](https://www.researchgate.net/profile/Gregor_Ulm/publication/329106719_A_Performance_Evaluation_of_Federated_Learning_Algorithms/links/5c0fabcfa6fdcc494febf907/A-Performance-Evaluation-of-Federated-Learning-Algorithms.pdf)
* [Edge AIBench: Towards Comprehensive End-to-end Edge Computing Benchmarking](https://arxiv.org/abs/1908.01924)



## Model Aggregation
* [One-Shot Federated Learning](https://arxiv.org/abs/1902.11175)
* [Federated Learning with Unbiased Gradient Aggregation and Controllable Meta Updating ](https://arxiv.org/abs/1910.08234) (NIPS 2019 Workshop)
* [Bayesian Nonparametric Federated Learning of Neural Networks](https://arxiv.org/abs/1905.12022) (ICML 2019)
* [Agnostic Federated Learning](https://arxiv.org/abs/1902.00146) (ICML 2019)
* [Federated Learning with Matched Averaging](https://openreview.net/forum?id=BkluqlSFDS) (ICLR 2020)
* [Astraea: Self-balancing federated learning for improving classification accuracy of mobile deep learning applications](https://arxiv.org/abs/1907.01132)



## Statistical Heterogeneity
* [FedPD: A Federated Learning Framework with Optimal Rates andAdaptivity to Non-IID Data](https://arxiv.org/pdf/2005.11418.pdf)
* [Decentralized Learning of Generative Adversarial Networks from Non-iid Data](https://arxiv.org/pdf/1905.09684.pdf)
* [Overcoming Forgetting in Federated Learning on Non-IID Data](https://arxiv.org/pdf/1910.07796.pdf)
* [Towards Class Imbalance in Federated Learning](https://arxiv.org/pdf/2008.06217.pdf)
* [Communication-Efficient On-Device Machine Learning:Federated Distillation and Augmentationunder Non-IID Private Data](https://arxiv.org/pdf/1811.11479v1.pdf)
* [Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization](https://arxiv.org/pdf/2007.07481.pdf)
* [Federated Adversarial Domain Adaptation](https://arxiv.org/abs/1911.02054)
* [Federated Learning with Only Positive Labels](https://arxiv.org/pdf/2004.10342.pdf)
* [Federated Learning with Non-IID Data](https://arxiv.org/abs/1806.00582) 
* [The Non-IID Data Quagmire of Decentralized Machine Learning](https://arxiv.org/abs/1910.00189)
* [Robust and Communication-Efficient Federated Learning from Non-IID Data](https://arxiv.org/pdf/1903.02891) [IEEE transactions on neural networks and learning systems]
* [FedMD: Heterogenous Federated Learning via Model Distillation](https://arxiv.org/abs/1910.03581) (NIPS 2019 Workshop)
* [First Analysis of Local GD on Heterogeneous Data](https://arxiv.org/abs/1909.04715)
* [SCAFFOLD: Stochastic Controlled Averaging for On-Device Federated Learning](https://arxiv.org/abs/1910.06378)
* [Federated Optimization for Heterogeneous Networks](https://arxiv.org/pdf/1812.06127)
* [On the Convergence of FedAvg on Non-IID Data](https://arxiv.org/abs/1907.02189) [[OpenReview]](https://openreview.net/forum?id=HJxNAnVtDS)
* [Agnostic Federated Learning](https://arxiv.org/abs/1902.00146) (ICML 2019)
* [Local SGD Converges Fast and Communicates Little](https://arxiv.org/abs/1805.09767)
* [Improving Federated Learning Personalization via Model Agnostic Meta Learning](https://arxiv.org/abs/1909.12488) (NIPS 2019 Workshop)
* [Adaptive Gradient-Based Meta-Learning Methods](https://arxiv.org/abs/1906.02717) (NIPS 2019 Workshop)
* [Federated Adversarial Domain Adaptation](https://arxiv.org/abs/1911.02054) (ICLR 2020)
* [LoAdaBoost: Loss-Based AdaBoost Federated Machine Learning on Medical Data](https://arxiv.org/pdf/1811.12629)
* [On Federated Learning of Deep Networks from Non-IID Data: Parameter Divergence and the Effects of Hyperparametric Methods](https://openreview.net/forum?id=SJeOAJStwB) [Rejected in ICML 2020]
* [Overcoming Forgetting in Federated Learning on Non-IID Data](https://arxiv.org/abs/1910.07796) [NIPS 2019 Workshop]
* [FedMAX: Activation Entropy Maximization Targeting Effective Non-IID Federated Learning](#workshop) [NIPS 2019 Workshop]
* [Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification](https://arxiv.org/abs/1909.06335) [NIPS 2019 Workshop]
* [Fair Resource Allocation in Federated Learning](https://arxiv.org/abs/1905.10497)
* [Communication-efficient on-device machine learning: Federated distillation and augmentation under non-iid private data](https://arxiv.org/abs/1811.11479)
* [Think Locally, Act Globally: Federated Learning with Local and Global Representations](https://arxiv.org/abs/2001.01523) [NIPS 2019 Workshop]

### Meta Learning
* [Federated Meta-Learning with Fast Convergence and Efficient Communication](https://arxiv.org/abs/1802.07876)
* [Federated Meta-Learning for Recommendation](https://www.semanticscholar.org/paper/Federated-Meta-Learning-for-Recommendation-Chen-Dong/8e21d353ba283bee8fd18285558e5e8df39d46e8#paper-header)
* [Adaptive Gradient-Based Meta-Learning Methods](https://arxiv.org/abs/1906.02717)

### Multi-task Learning
* [MOCHA: Federated Multi-Task Learning](https://arxiv.org/abs/1705.10467) [[NIPS 2017]](https://papers.nips.cc/paper/7029-federated-multi-task-learning) [[Slides]](http://learningsys.org/nips17/assets/slides/mocha-NIPS.pdf)
* [Variational Federated Multi-Task Learning](https://arxiv.org/abs/1906.06268)
* [Federated Kernelized Multi-Task Learning](https://mlsys.org/Conferences/2019/doc/2018/30.pdf)
* [Clustered Federated Learning: Model-Agnostic Distributed Multi-Task Optimization under Privacy Constraints](https://arxiv.org/abs/1910.01991) [NIPS 2019 Workshop]

### Convergence
* [A Linear Speedup Analysis of Distributed Deep Learning with Sparse and Quantized Communication](https://papers.nips.cc/paper/7519-a-linear-speedup-analysis-of-distributed-deep-learning-with-sparse-and-quantized-communication) [NIPS 2018]
* [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/pdf/1602.05629.pdf)
* [FetchSGD: Communication-Efficient Federated Learning with Sketching](https://arxiv.org/pdf/2007.07682.pdf)
* [Federated Learning for Wireless Communications: Motivation, Opportunities and Challenges](https://arxiv.org/pdf/1908.06847v3.pdf)
* [On the Convergence of FedAvg on Non-IID Data](https://arxiv.org/pdf/1907.02189.pdf)
* [SCAFFOLD: Stochastic Controlled Averaging for On-Device Federated Learning](https://arxiv.org/abs/1910.06378)
* [Federated Optimization for Heterogeneous Networks](https://arxiv.org/pdf/1812.06127)
* [On the Convergence of FedAvg on Non-IID Data](https://arxiv.org/abs/1907.02189) [[OpenReview]](https://openreview.net/forum?id=HJxNAnVtDS)
* [Can Decentralized Algorithms Outperform Centralized Algorithms? A Case Study for Decentralized Parallel Stochastic Gradient Descent](https://arxiv.org/abs/1705.09056) [NIPS 2017]
* [Communication Efficient Decentralized Training with Multiple Local Updates](https://arxiv.org/abs/1910.09126)
* [First Analysis of Local GD on Heterogeneous Data](https://arxiv.org/abs/1909.04715)
* [MATCHA: Speeding Up Decentralized SGD via Matching Decomposition Sampling](https://arxiv.org/abs/1905.09435)
* [Local SGD Converges Fast and Communicates Little](https://arxiv.org/abs/1805.09767)
* [SlowMo: Improving Communication-Efficient Distributed SGD with Slow Momentum](https://arxiv.org/abs/1910.00643)
* [Adaptive Federated Learning in Resource Constrained Edge Computing Systems](https://arxiv.org/abs/1804.05271) [IEEE Journal on Selected Areas in Communications, 2019]
* [Parallel Restarted SGD with Faster Convergence and Less Communication: Demystifying Why Model Averaging Works for Deep Learning](https://arxiv.org/abs/1807.06629) [AAAI 2018]
* [On the Linear Speedup Analysis of Communication Efficient Momentum SGD for Distributed Non-Convex Optimization](https://arxiv.org/abs/1905.03817) [ICML 2019]
* [Communication-efficient on-device machine learning: Federated distillation and augmentation under non-iid private data](https://arxiv.org/abs/1811.11479)
* [Convergence of Distributed Stochastic Variance Reduced Methods without Sampling Extra Data](https://arxiv.org/abs/1905.12648) [NIPS 2019 Workshop]


## System
* [Towards Federated Learning at Scale: System Design](https://arxiv.org/abs/1902.01046) **[Must Read]**
* [Demonstration of Federated Learning in a Resource-Constrained Networked Environment](https://ieeexplore.ieee.org/document/8784064)
* [Federated Learning Systems: Vision, Hype and Reality for Data Privacy and Protection](https://arxiv.org/abs/1907.09693)
* [Applied Federated Learning: Improving Google Keyboard Query Suggestions](https://arxiv.org/abs/1812.02903)
* [Federated Learning and Differential Privacy: Software tools analysis, the Sherpa.ai FL framework and methodological guidelines for preserving data privacy](https://arxiv.org/abs/2007.00914) (Startup)


## Communication Efficiency
* [*Communication-Efficient Learning of Deep Networks from Decentralized Data*](https://arxiv.org/abs/1602.05629)](https://github.com/roxanneluo/Federated-Learning) [Google] **[Must Read]**
* [Two-Stream Federated Learning: Reduce the Communication Costs](https://ieeexplore.ieee.org/document/8698609) [2018 IEEE VCIP] 
* [Client-Edge-Cloud Hierarchical Federated Learning](https://arxiv.org/abs/1905.06641)
* [PowerSGD: Practical Low-Rank Gradient Compression for Distributed Optimization](https://arxiv.org/abs/1905.13727) [NIPS 2019], Thijs Vogels, Sai Praneeth Karimireddy, and Martin Jaggi. 
* [Deep Gradient Compression: Reducing the Communication Bandwidth for Distributed Training](https://arxiv.org/abs/1712.01887) [ICLR 2018] Yujun Lin, Song Han, Huizi Mao, Yu Wang, and William J Dally
* [The Error-Feedback Framework: Better Rates for SGD with Delayed Gradients and Compressed Communication](https://arxiv.org/abs/1909.05350) Sebastian U Stich and Sai Praneeth Karimireddy, 2019.
* [A Communication Efficient Collaborative Learning Framework for Distributed Features](https://arxiv.org/abs/1912.11187) [NIPS 2019 Workshop]
* [Active Federated Learning](https://arxiv.org/abs/1909.12641) [NIPS 2019 Workshop]
* [Communication-Efficient Distributed Optimization in Networks with Gradient Tracking and Variance Reduction](https://arxiv.org/abs/1909.05844) [NIPS 2019 Workshop]
* [Gradient Descent with Compressed Iterates](https://arxiv.org/abs/1909.04716) [NIPS 2019 Workshop]

### Compression
* [Robust and Communication-Efficient Federated Learning from Non-IID Data](https://arxiv.org/pdf/1903.02891), 2019
* [Expanding the Reach of Federated Learning by Reducing Client Resource Requirements](https://arxiv.org/abs/1812.07210) Sebastian Caldas, Jakub Konecny, H Brendan McMahan, and Ameet Talwalkar, 2018
* [Federated Learning: Strategies for Improving Communication Efficiency](https://arxiv.org/abs/1610.05492) [NIPS2016 Workshop] [Google]
* [Natural Compression for Distributed Deep Learning](https://arxiv.org/abs/1905.10988) Samuel Horvath, Chen-Yu Ho, Ludovit Horvath, Atal Narayan Sahu, Marco Canini, and Peter Richtarik, 2019.
* [FedPAQ: A Communication-Efficient Federated Learning Method with Periodic Averaging and Quantization](https://arxiv.org/abs/1909.13014), 2019
* [ATOMO: Communication-efficient Learning via Atomic Sparsification](https://arxiv.org/abs/1806.04090)  [NIPS 2018], H. Wang, S. Sievert, S. Liu, Z. Charles, D. Papailiopoulos, and S. Wright.
* [vqSGD: Vector Quantized Stochastic Gradient Descent](https://arxiv.org/abs/1911.07971) Venkata Gandikota, Raj Kumar Maity, and Arya Mazumdar, 2019.
* [QSGD: Communication-efficient SGD via gradient quantization and encoding](https://arxiv.org/abs/1610.02132) [NIPS 2017], Dan Alistarh, Demjan Grubic, Jerry Li, Ryota Tomioka, and Milan Vojnovic.
* [cpSGD: Communication-efficient and differentially-private distributed SGD](https://arxiv.org/abs/1805.10559)
* [Federated Optimization: Distributed Machine Learning for On-Device Intelligence](https://arxiv.org/abs/1610.02527) [Google]
* [Distributed Mean Estimation with Limited Communication](https://arxiv.org/abs/1611.00429) [ICML 2017], Ananda Theertha Suresh, Felix X. Yu, Sanjiv Kumar, and H Brendan McMahan. 
* [Randomized Distributed Mean Estimation: Accuracy vs Communication](https://arxiv.org/abs/1611.07555) Frontiers in Applied Mathematics and Statistics, Jakub Konecny and Peter Richtarik, 2016
* [Error Feedback Fixes SignSGD and other Gradient Compression Schemes](https://arxiv.org/abs/1901.09847) [ICML 2019], Sai Praneeth Karimireddy, Quentin Rebjock, Sebastian Stich, and Martin Jaggi.
* [ZipML: Training Linear Models with End-to-End Low Precision, and a Little Bit of Deep Learning](http://proceedings.mlr.press/v70/zhang17e.html) [ICML 2017], H. Zhang, J. Li, K. Kara, D. Alistarh, J. Liu, and C. Zhang.

### Important-Based Updating
* [eSGD: Communication Efficient Distributed Deep Learning on the Edge](https://www.usenix.org/conference/hotedge18/presentation/tao) [USENIX 2018 Workshop (HotEdge 18)]
* [CMFL: Mitigating Communication Overhead for Federated Learning](http://home.cse.ust.hk/~lwangbm/CMFL.pdf)


### Decentralization
* [Communication Compression for Decentralized Training](https://arxiv.org/abs/1803.06443) [NIPS 2018], H. Tang, S. Gan, C. Zhang, T. Zhang, and J. Liu.
* [𝙳𝚎𝚎𝚙𝚂𝚚𝚞𝚎𝚎𝚣𝚎: Decentralization Meets Error-Compensated Compression](https://arxiv.org/abs/1907.07346) Hanlin Tang, Xiangru Lian, Shuang Qiu, Lei Yuan, Ce Zhang, Tong Zhang, and Ji Liu, 2019


## Resource Allocation

### Participants Selection

* [Client Selection for Federated Learning with Heterogeneous Resources in Mobile Edge](https://arxiv.org/abs/1804.08333) (FedCS)
* [Hybrid-FL for Wireless Networks: Cooperative Learning Mechanism Using Non-IID Data](https://arxiv.org/abs/1905.07210)
* [Ask to upload some data from client to server Efficient Training Management for Mobile Crowd-Machine Learning: A Deep Reinforcement Learning Approach](https://arxiv.org/abs/1812.03633)
* [Reward function: accumulated data, energy consumption, training accuracy Fair Resource Allocation in Federated Learning](https://arxiv.org/abs/1905.10497)
* [Low-latency Broadband Analog Aggregation For Federated Edge Learning](https://arxiv.org/abs/1812.11494)
* [Federated Learning over Wireless Fading Channels](https://arxiv.org/pdf/1907.09769.pdf)
* [Federated Learning via Over-the-Air Computation](https://arxiv.org/abs/1812.11750)

### Adaptive Aggregation

* [Asynchronous Federated Learning for Geospatial Applications](https://link.springer.com.remotexs.ntu.edu.sg/chapter/10.1007/978-3-030-14880-5_2) [ECML PKDD Workshop 2018] 
* [Asynchronous Federated Optimization](https://arxiv.org/abs/1903.03934)
* [Adaptive Federated Learning in Resource Constrained Edge Computing Systems](https://arxiv.org/abs/1804.05271) [IEEE Journal on Selected Areas in Communications, 2019]

### Incentive Mechanism

* [Incentive Mechanism for Reliable Federated Learning: A Joint Optimization Approach to Combining Reputation and Contract Theory](https://ieeexplore.ieee.org/document/8832210)
* [Motivating Workers in Federated Learning: A Stackelberg Game Perspective](https://arxiv.org/abs/1908.03092)
* [Incentive Design for Efficient Federated Learning in Mobile Networks: A Contract Theory Approach [2019]](https://arxiv.org/abs/1905.07479)
* [Fair Resource Allocation in Federated Learning](https://arxiv.org/pdf/1905.10497v1.pdf)

## Vertical Federated Learning

* [A Quasi-Newton Method Based Vertical Federated Learning Framework for Logistic Regression](https://arxiv.org/abs/1912.00513) [NIPS 2019 Workshop]

## Adversarial Attacks
* [Can You Really Backdoor Federated Learning?](https://arxiv.org/abs/1911.07963)
* [Model Poisoning Attacks in Federated Learning](https://dais-ita.org/sites/default/files/main_secml_model_poison.pdf) [NIPS workshop 2018]

## Data Privacy and Confidentiality

* [Gradient-Leaks: Understanding and Controlling Deanonymization in Federated Learning](https://arxiv.org/abs/1805.05838) [NIPS 2019 Workshop]
* [Quantification of the Leakage in Federated Learning](https://arxiv.org/pdf/1910.05467.pdf)


### Courses

* [Applied Cryptography](https://www.udacity.com/course/applied-cryptography--cs387)

### Differential Privacy

* [A Brief Introduction to Differential Privacy](https://medium.com/georgian-impact-blog/a-brief-introduction-to-differential-privacy-eacf8722283b)
* [Deep Learning with Differential Privacy*](http://doi.acm.org/10.1145/2976749.2978318)
  * Martin Abadi, Andy Chu, Ian Goodfellow, H. Brendan McMahan, Ilya Mironov, Kunal Talwar, and Li Zhang.
* [Learning Differentially Private Recurrent Language Models](https://arxiv.org/abs/1710.06963)
* [Federated Learning with Bayesian Differential Privacy](https://arxiv.org/abs/1911.10071) [NIPS 2019 Workshop]
* [Private Federated Learning with Domain Adaptation](https://arxiv.org/abs/1912.06733) [NIPS 2019 Workshop]
* [cpSGD: Communication-efficient and differentially-private distributed SGD](https://arxiv.org/abs/1805.10559)
* [Federated Learning with Bayesian Differential Privacy](https://arxiv.org/pdf/1911.10071.pdf) [NIPS 2019 Workshop]



### Secure Multi-party Computation

#### Secret Sharing
* [Simple Introduction to Sharmir's Secret Sharing and Lagrange Interpolation](https://www.youtube.com/watch?v=kkMps3X_tEE)
* [Secret Sharing, Part 1](https://mortendahl.github.io/2017/06/04/secret-sharing-part1/): Shamir's Secret Sharing & Packed Variant
* [Secret Sharing, Part 2](https://mortendahl.github.io/2017/06/24/secret-sharing-part2/): Improve efficiency
* [Secret Sharing, Part 3](https://mortendahl.github.io/2017/08/13/secret-sharing-part3/)


#### Build Safe AI Series
* [Building Safe A.I.](http://iamtrask.github.io/2017/03/17/safe-ai/)
  * A Tutorial for Encrypted Deep Learning
  * Use Homomorphic Encryption (HE)

* [Private Deep Learning with MPC](https://mortendahl.github.io/2017/04/17/private-deep-learning-with-mpc/)
  * A Simple Tutorial from Scratch
  * Use Multiparty Compuation (MPC)

* [Private Image Analysis with MPC](https://mortendahl.github.io/2017/09/19/private-image-analysis-with-mpc/)
  * Training CNNs on Sensitive Data
  * Use SPDZ as MPC protocol

#### MPC related Paper
[Helen: Maliciously Secure Coopetitive Learning for Linear Models](https://arxiv.org/abs/1907.07212) (NIPS 2019 Workshop)



### Privacy Preserving Machine Learning

* [Privacy-Preserving Deep Learning](https://www.comp.nus.edu.sg/~reza/files/Shokri-CCS2015.pdf)
* [Privacy Partition: A Privacy-Preserving Framework for Deep Neural Networks in Edge Networks](http://mews.sv.cmu.edu/papers/archedge-18.pdf)
* [Practical Secure Aggregation for Privacy-Preserving Machine Learning](https://eprint.iacr.org/2017/281.pdf) (Google)
  * Secure Aggregation: The problem of computing a multiparty sum where no party reveals its update in the clear—even to the aggregator
  * Goal: securely computing sums of vectors, which has a constant number of rounds, low communication overhead, robustness to failures, and which requires only one server with limited trust
  * Need to have basic knowledge of cryptographic algorithms such as secret sharing, key agreement, etc.
* [Practical Secure Aggregation for Federated Learning on User-Held Data](https://arxiv.org/abs/1611.04482) (Google)
  * Highly related to *Practical Secure Aggregation for Privacy-Preserving Machine Learning*
  * Proposed 4 protocol one by one with gradual improvement to meet the requirement of secure aggregation propocol.
* [SecureML: A System for Scalable Privacy-Preserving Machine Learning](https://eprint.iacr.org/2017/396.pdf)
* [Practical Secure Aggregation for Privacy-Preserving Machine Learning](https://eprint.iacr.org/2017/281.pdf)
* [DeepSecure: Scalable Provably-Secure Deep Learning](https://arxiv.org/abs/1705.08963)
* [Chameleon: A Hybrid Secure Computation Framework for Machine Learning Applications](https://arxiv.org/pdf/1801.03239.pdf)


## Other Learning

* [Federated Reinforcement Learning](https://arxiv.org/abs/1901.08277)
* [Secure Federated Transfer Learning](https://arxiv.org/abs/1812.03337)
* [Federated Learning with Matched Averaging](https://arxiv.org/abs/2002.06440)
* [Bayesian Nonparametric Federated Learning of Neural Networks](https://arxiv.org/abs/1905.12022v1)
* [Multi-Center Federated Learning](https://arxiv.org/pdf/2005.01026.pdf)
* [FedGAN: Federated Generative Adversarial Networks for Distributed Data](https://arxiv.org/pdf/2006.07228.pdf)
* [Generative models for effective ML on private decentralized datasets](https://arxiv.org/pdf/1911.06679.pdf)
* [FEDFMC: Sequential efficient Federated Learning on Non-IID data](https://arxiv.org/pdf/2006.10937v1.pdf) 

## Frameworks

* [PySyft](https://github.com/OpenMined/PySyft)
  * [A Generic Framework for Privacy Preserving Peep Pearning](https://arxiv.org/abs/1811.04017)
* [Tensorflow Federated](https://www.tensorflow.org/federated)
* [FATE](https://github.com/FederatedAI/FATE)
* [Baidu PaddleFL](https://github.com/PaddlePaddle/PaddleFL)
* [Nvidia Clara SDK](https://developer.nvidia.com/clara)


## Workshops

* [NIPS 2019 Workshop on Federated Learning for Data Privacy and Confidentiality 1](https://slideslive.com/38921898/workshop-on-federated-learning-for-data-privacy-and-confidentiality-1)
* [NIPS 2019 Workshop on Federated Learning for Data Privacy and Confidentiality 2](https://slideslive.com/38921899/workshop-on-federated-learning-for-data-privacy-and-confidentiality-2)
* [NIPS 2019 Workshop on Federated Learning for Data Privacy and Confidentiality 3](https://slideslive.com/38921900/workshop-on-federated-learning-for-data-privacy-and-confidentiality-3)


## Applications

* [Federated Learning Approach for Mobile Packet Classification](https://arxiv.org/abs/1907.13113)
* [Federated Learning for Ranking Browser History Suggestions](https://arxiv.org/abs/1911.11807) [NIPS 2019 Workshop]

### Healthcare
* [HHHFL: Hierarchical Heterogeneous Horizontal Federated Learning for Electroencephalography](https://arxiv.org/abs/1909.05784) [NIPS 2019 Workshop]
* [Learn Electronic Health Records by Fully Decentralized Federated Learning](https://arxiv.org/abs/1912.01792) [NIPS 2019 Workshop]
* [Patient Clustering Improves Efficiency of Federated Machine Learning to predict mortality and hospital stay time using distributed Electronic Medical Records](https://arxiv.org/ftp/arxiv/papers/1903/1903.09296.pdf) [[News]](https://venturebeat.com/2019/03/25/federated-learning-technique-predicts-hospital-stay-and-patient-mortality/)
  * MIT CSAI, Harvard Medical School, Tsinghua University
* [Federated learning of predictive models from federated Electronic Health Records.](https://www.ncbi.nlm.nih.gov/pubmed/29500022)
  * Boston University, Massachusetts General Hospital
* [FedHealth: A Federated Transfer Learning Framework for Wearable Healthcare](https://arxiv.org/pdf/1907.09173.pdf)
  * Microsoft Research Asia
* [Multi-Institutional Deep Learning Modeling Without Sharing Patient Data: A Feasibility Study on Brain Tumor Segmentation](https://arxiv.org/pdf/1810.04304.pdf)
  * Intel
* [NVIDIA Clara Federated Learning to Deliver AI to Hospitals While Protecting Patient Data](https://blogs.nvidia.com/blog/2019/12/01/clara-federated-learning/)
  * Nvidia
* [What is Federated Learning](https://blogs.nvidia.com/blog/2019/10/13/what-is-federated-learning/)
  * Nvidia
* [Split learning for health: Distributed deep learning without sharing raw patient data](https://arxiv.org/pdf/1812.00564)
* [Two-stage Federated Phenotyping and Patient Representation Learning](https://www.aclweb.org/anthology/W19-5030.pdf) [ACL 2019]
* [Federated Tensor Factorization for Computational Phenotyping](https://dl.acm.org/doi/10.1145/3097983.3098118) SIGKDD 2017
* [FedHealth- A Federated Transfer Learning Framework for Wearable Healthcare](https://arxiv.org/abs/1907.09173) [ICJAI19 workshop]
* [Multi-Institutional Deep Learning Modeling Without Sharing Patient Data- A Feasibility Study on Brain Tumor Segmentation](https://arxiv.org/abs/1810.04304) [MICCAI'18 Workshop]
* [Federated Patient Hashing](https://aaai.org/ojs/index.php/AAAI/article/view/6121) [AAAI'20]
  
### Natual Language Processing

Google

* [Federated Learning for Mobile Keyboard Prediction](https://arxiv.org/abs/1811.03604)
* [Applied Federated Learning: Improving Google Keyboard Query Suggestions](https://arxiv.org/abs/1812.02903)
* [Federated Learning Of Out-Of-Vocabulary Words](https://arxiv.org/abs/1903.10635)
* [Federated Learning for Emoji Prediction in a Mobile Keyboard](https://arxiv.org/abs/1906.04329)

Snips

* [Federated Learning for Wake Keyword Spotting](https://arxiv.org/pdf/1810.05512.pdf)](https://medium.com/snips-ai/federated-learning-for-wake-word-detection-c8b8c5cdd2c5)](https://github.com/snipsco/keyword-spotting-research-datasets)

### Computer Vision

* [Performance Optimization for Federated Person Re-identification via Benchmark Analysis](https://arxiv.org/abs/2008.11560) [ACMMM20]](https://github.com/cap-ntu/FedReID)
* [Real-World Image Datasets for Federated Learning](https://arxiv.org/abs/1910.11089)
  * Webank & Extreme Vision
* [FedVision- An Online Visual Object Detection Platform Powered by Federated Learning](https://arxiv.org/abs/2001.06202) [IAAI20]
* [Federated Learning for Vision-and-Language Grounding Problems](http://web.pkusz.edu.cn/adsp/files/2019/11/AAAI-FenglinL.1027.pdf) [AAAI20]

### Recommendation

 * [Federated Collaborative Filtering for Privacy-Preserving Personalized Recommendation System](https://arxiv.org/abs/1901.09888)
    * Huawei
 * [Federated Meta-Learning with Fast Convergence and Efficient Communication](https://arxiv.org/abs/1802.07876)
    * Huawei

### Industrial

* [Turbofan POC: Predictive Maintenance of Turbofan Engines using Federated Learning](https://github.com/matthiaslau/Turbofan-Federated-Learning-POC)
* [Turbofan Tycoon Simulation by Cloudera/FastForwardLabs](https://turbofan.fastforwardlabs.com/)
* [Firefox Search Bar](https://florian.github.io/federated-learning/)
  * Detail explaination of their implementationn of Federated Learning in production.

## Company

* [Privacy.ai](https://privacy.ai/)
* [OpenMined](https://www.openmined.org/)
* [Arkhn: Healthcare data](https://arkhn.org/en/)
* [Owkin: Medical research](https://owkin.com/)
* [Snips: Voice platform](https://snips.ai/)
* [XAIN: Automated Invoicing](https://www.xain.io/)
* [S20: Multiple third party collaboration](https://www.s20.ai/)
