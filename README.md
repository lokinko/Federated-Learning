# Federated Learning

>*人工智能（Artificial Intelligence, AI）进入以深度学习为主导的大数据时代，基于大数据的机器学习既推动了AI的蓬勃发展，也带来了一系列安全隐患。这些隐患来源于深度学习本身的学习机制，无论是在它的模型建造（训练）阶段，还是在模型推理和使用阶段。这些安全隐患如果被有意或无意地滥用，后果将十分严重。*

---

**联邦学习是一种 <font color=#B22222>隐私保护、数据本地存储与计算</font> 的机器学习算法。**

# 文献参考


## Part 1: Introduction
* [Federated Learning Comic](https://federated.withgoogle.com/)
* [Federated Learning: Collaborative Machine Learning without Centralized Training Data](http://ai.googleblog.com/2017/04/federated-learning-collaborative.html)
* [GDPR, Data Shotrage and AI (AAAI-19)](https://aaai.org/Conferences/AAAI-19/invited-speakers/#yang)
* [Federated Learning: Machine Learning on Decentralized Data (Google I/O'19)](https://www.youtube.com/watch?v=89BGjQYA0uE)
* [Federated Learning White Paper V1.0](https://www.fedai.org/static/flwp-en.pdf)
* [Federated learning: distributed machine learning with data locality and privacy](https://blog.fastforwardlabs.com/2018/11/14/federated-learning.html)

## Part 2: Survey
* [Federated Learning: Challenges, Methods, and Future Directions](https://arxiv.org/abs/1908.07873)
* [Federated Learning Systems: Vision, Hype and Reality for Data Privacy and Protection](https://arxiv.org/abs/1907.09693)
* [Federated Learning in Mobile Edge Networks: A Comprehensive Survey](https://arxiv.org/abs/1909.11875)
* [Federated Learning for Wireless Communications: Motivation, Opportunities and Challenges](https://arxiv.org/abs/1908.06847)
* [Convergence of Edge Computing and Deep Learning: A Comprehensive Survey](https://arxiv.org/pdf/1907.08349.pdf)
* [Advances and Open Problems in Federated Learning](https://arxiv.org/pdf/1912.04977.pdf)
* [Federated Machine Learning: Concept and Applications](https://arxiv.org/pdf/1902.04885.pdf)
* [Threats to Federated Learning: A Survey](https://arxiv.org/pdf/2003.02133.pdf)
* [Survey of Personalization Techniques for Federated Learning](https://arxiv.org/pdf/2003.08673.pdf)
* [SECure: A Social and Environmental Certificate for AI Systems](https://arxiv.org/pdf/2006.06217.pdf)
* [From Federated Learning to Fog Learning: Towards Large-Scale Distributed Machine Learning in Heterogeneous Wireless Networks](https://arxiv.org/pdf/2006.03594.pdf)
* [Federated Learning for 6G Communications: Challenges, Methods, and Future Directions](https://arxiv.org/pdf/2006.02931.pdf)
* [A Review of Privacy Preserving Federated Learning for Private IoT Analytics](https://arxiv.org/pdf/2004.11794.pdf)
* [Towards Utilizing Unlabeled Data in Federated Learning: A Survey and Prospective](https://arxiv.org/pdf/2002.11545.pdf)
* [Federated Learning for Resource-Constrained IoT Devices: Panoramas and State-of-the-art](https://arxiv.org/pdf/2002.10610.pdf)
* [Privacy-Preserving Blockchain Based Federated Learning with Differential Data Sharing](https://arxiv.org/pdf/1912.04859.pdf)
* [An Introduction to Communication Efficient Edge Machine Learning](https://arxiv.org/pdf/1912.01554.pdf)
* [Federated Learning for Healthcare Informatics](https://arxiv.org/pdf/1911.06270.pdf)
* [Federated Learning for Coalition Operations](https://arxiv.org/pdf/1910.06799.pdf)
* [No Peek: A Survey of private distributed deep learning](https://arxiv.org/pdf/1812.03288.pdf)
* [Communication-Efficient Edge AI: Algorithms and Systems](http://arxiv.org/pdf/2002.09668.pdf)

## Part 3: Benchmarks
* [LEAF: A Benchmark for Federated Settings](https://arxiv.org/abs/1812.01097)(https://github.com/TalwalkarLab/leaf) [Recommend]
* [A Performance Evaluation of Federated Learning Algorithms](https://www.researchgate.net/profile/Gregor_Ulm/publication/329106719_A_Performance_Evaluation_of_Federated_Learning_Algorithms/links/5c0fabcfa6fdcc494febf907/A-Performance-Evaluation-of-Federated-Learning-Algorithms.pdf)
* [Edge AIBench: Towards Comprehensive End-to-end Edge Computing Benchmarking](https://arxiv.org/abs/1908.01924)

## Part 4: Converge
### 4.1 Model Aggregation
* [One-Shot Federated Learning](https://arxiv.org/abs/1902.11175)
* [Federated Learning with Unbiased Gradient Aggregation and Controllable Meta Updating](https://arxiv.org/abs/1910.08234) (NIPS 2019 Workshop)
* [Bayesian Nonparametric Federated Learning of Neural Networks](https://arxiv.org/abs/1905.12022) (ICML 2019)
* [FedBE: Making Bayesian Model Ensemble Applicable to Federated Learning](https://openreview.net/forum?id=dgtpE6gKjHn) (ICLR 2021)
* [Agnostic Federated Learning](https://arxiv.org/abs/1902.00146) (ICML 2019)
* [Federated Learning with Matched Averaging](https://openreview.net/forum?id=BkluqlSFDS) (ICLR 2020)
* [Astraea: Self-balancing federated learning for improving classification accuracy of mobile deep learning applications](https://arxiv.org/abs/1907.01132)

### 4.2 Convergence Research
* [A Linear Speedup Analysis of Distributed Deep Learning with Sparse and Quantized Communication](https://papers.nips.cc/paper/7519-a-linear-speedup-analysis-of-distributed-deep-learning-with-sparse-and-quantized-communication) （NIPS 2018）
* [Achieving Linear Speedup with Partial Worker Participation in Non-IID Federated Learning](https://openreview.net/forum?id=jDdzh5ul-d) (ICLR 2021)
* [FetchSGD: Communication-Efficient Federated Learning with Sketching](https://arxiv.org/pdf/2007.07682.pdf)
* [FL-NTK: A Neural Tangent Kernel-based Framework for Federated Learning Convergence Analysis](https://arxiv.org/abs/2105.05001) (ICML 2021)
* [Federated Optimization for Heterogeneous Networks](https://arxiv.org/pdf/1812.06127)
* [On the Convergence of FedAvg on Non-IID Data](https://arxiv.org/abs/1907.02189) [[OpenReview]](https://openreview.net/forum?id=HJxNAnVtDS)
* [Communication Efficient Decentralized Training with Multiple Local Updates](https://arxiv.org/abs/1910.09126)
* [Local SGD Converges Fast and Communicates Little](https://arxiv.org/abs/1805.09767)
* [SlowMo: Improving Communication-Efficient Distributed SGD with Slow Momentum](https://arxiv.org/abs/1910.00643)
* [Parallel Restarted SGD with Faster Convergence and Less Communication: Demystifying Why Model Averaging Works for Deep Learning](https://arxiv.org/abs/1807.06629) (AAAI 2018）
* [On the Linear Speedup Analysis of Communication Efficient Momentum SGD for Distributed Non-Convex Optimization](https://arxiv.org/abs/1905.03817) (ICML 2019）
* [Communication-efficient on-device machine learning: Federated distillation and augmentation under non-iid private data](https://arxiv.org/abs/1811.11479)
* [Convergence of Distributed Stochastic Variance Reduced Methods without Sampling Extra Data](https://arxiv.org/abs/1905.12648) （NIPS 2019 Workshop)

### 4.3 Statistical Heterogeneity
* [FedPD: A Federated Learning Framework with Optimal Rates andAdaptivity to Non-IID Data](https://arxiv.org/pdf/2005.11418.pdf)
* [FedBN: Federated Learning on Non-IID Features via Local Batch Normalization](https://openreview.net/forum?id=6YEQUn0QICG) (ICLR 2021)
* [FedMix: Approximation of Mixup under Mean Augmented Federated Learning](https://openreview.net/forum?id=Ogga20D2HO-) (ICLR 2021)
* [HeteroFL: Computation and Communication Efficient Federated Learning for Heterogeneous Clients](https://openreview.net/forum?id=TNkPBBYFkXg) (ICLR 2021)
* [Decentralized Learning of Generative Adversarial Networks from Non-iid Data](https://arxiv.org/pdf/1905.09684.pdf)
* [Towards Class Imbalance in Federated Learning](https://arxiv.org/pdf/2008.06217.pdf)
* [Communication-Efficient On-Device Machine Learning:Federated Distillation and Augmentationunder Non-IID Private Data](https://arxiv.org/pdf/1811.11479v1.pdf)
* [Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization](https://arxiv.org/pdf/2007.07481.pdf)
* [Federated Adversarial Domain Adaptation](https://arxiv.org/abs/1911.02054)
* [Federated Learning with Only Positive Labels](https://arxiv.org/pdf/2004.10342.pdf)
* [Federated Learning with Non-IID Data](https://arxiv.org/abs/1806.00582) 
* [The Non-IID Data Quagmire of Decentralized Machine Learning](https://arxiv.org/abs/1910.00189)
* [Robust and Communication-Efficient Federated Learning from Non-IID Data](https://arxiv.org/pdf/1903.02891) (IEEE transactions on neural networks and learning systems)
* [FedMD: Heterogenous Federated Learning via Model Distillation](https://arxiv.org/abs/1910.03581) (NIPS 2019 Workshop)
* [First Analysis of Local GD on Heterogeneous Data](https://arxiv.org/abs/1909.04715)
* [SCAFFOLD: Stochastic Controlled Averaging for On-Device Federated Learning](https://arxiv.org/abs/1910.06378)
* [Improving Federated Learning Personalization via Model Agnostic Meta Learning](https://arxiv.org/abs/1909.12488) (NIPS 2019 Workshop)
* [Personalized Federated Learning with First Order Model Optimization](https://openreview.net/forum?id=ehJqJQk9cw) (ICLR 2021)
* [LoAdaBoost: Loss-Based AdaBoost Federated Machine Learning on Medical Data](https://arxiv.org/pdf/1811.12629)
* [On Federated Learning of Deep Networks from Non-IID Data: Parameter Divergence and the Effects of Hyperparametric Methods](https://openreview.net/forum?id=SJeOAJStwB)
* [Overcoming Forgetting in Federated Learning on Non-IID Data](https://arxiv.org/abs/1910.07796) （NIPS 2019 Workshop)
* [FedMAX: Activation Entropy Maximization Targeting Effective Non-IID Federated Learning](#workshop) （NIPS 2019 Workshop)
* [Adaptive Federated Optimization.](https://arxiv.org/pdf/2003.00295.pdf)(ICLR 2021 (Under Review))
* [Stochastic, Distributed and Federated Optimization for Machine Learning. FL PhD Thesis. By Jakub](https://arxiv.org/pdf/1707.01155.pdf)
* [Collaborative Deep Learning in Fixed Topology Networks](https://arxiv.org/pdf/1706.07880.pdf)
* [FedCD: Improving Performance in non-IID Federated Learning.](https://arxiv.org/pdf/2006.09637.pdf)
* [Life Long Learning: FedFMC: Sequential Efficient Federated Learning on Non-iid Data.](https://arxiv.org/pdf/2006.10937.pdf)
* [Robust Federated Learning: The Case of Affine Distribution Shifts.](https://arxiv.org/pdf/2006.08907.pdf)
* [Exploiting Shared Representations for Personalized Federated Learning](https://arxiv.org/abs/2102.07078) (ICML 2021)
* [Personalized Federated Learning using Hypernetworks](https://arxiv.org/abs/2103.04628) (ICML 2021)
* [Ditto: Fair and Robust Federated Learning Through Personalization](https://onikle.com/articles/359482) (ICML 2021)
* [Data-Free Knowledge Distillation for Heterogeneous Federated Learning](https://arxiv.org/abs/2105.10056) (ICML 2021)
* [Bias-Variance Reduced Local SGD for Less Heterogeneous Federated Learning](https://arxiv.org/abs/2102.03198) (ICML 2021)
* [Heterogeneity for the Win: One-Shot Federated Clustering](https://arxiv.org/abs/2103.00697) (ICML 2021)
* [Clustered Sampling: Low-Variance and Improved Representativity for Clients Selection in Federated Learning](https://arxiv.org/abs/2105.05883) (ICML 2021)
* [Federated Deep AUC Maximization for Hetergeneous Data with a Constant Communication Complexity](https://arxiv.org/abs/2102.04635) (ICML 2021)
* [Federated Learning of User Verification Models Without Sharing Embeddings](https://arxiv.org/abs/2104.08776) (ICML 2021)
* [One for One, or All for All: Equilibria and Optimality of Collaboration in Federated Learning](https://arxiv.org/abs/2103.03228) (ICML 2021)
* [Ensemble Distillation for Robust Model Fusion in Federated Learning.](https://arxiv.org/pdf/2006.07242.pdf)
* [XOR Mixup: Privacy-Preserving Data Augmentation for One-Shot Federated Learning.](https://arxiv.org/pdf/2006.05148.pdf)
* [An Efficient Framework for Clustered Federated Learning.](https://arxiv.org/pdf/2006.04088.pdf)
* [Continual Local Training for Better Initialization of Federated Models.](https://arxiv.org/pdf/2005.12657.pdf)
* [FedPD: A Federated Learning Framework with Optimal Rates and Adaptivity to Non-IID Data.](https://arxiv.org/pdf/2005.11418.pdf)
* [Global Multiclass Classification from Heterogeneous Local Models.](https://arxiv.org/pdf/2005.10848.pdf)
* [Multi-Center Federated Learning.](https://arxiv.org/pdf/2005.01026.pdf)
* [Federated Semi-Supervised Learning with Inter-Client Consistency & Disjoint Learning](https://openreview.net/forum?id=ce6CFXBh30h) (ICLR 2021)
* [(*) FedMAX: Mitigating Activation Divergence for Accurate and Communication-Efficient Federated Learning. CMU ECE.](https://arxiv.org/pdf/2004.03657.pdf)
* [(*) Adaptive Personalized Federated Learning](https://arxiv.org/pdf/2003.13461.pdf)
* [Semi-Federated Learning](https://arxiv.org/pdf/2003.12795.pdf)
* [Device Heterogeneity in Federated Learning: A Superquantile Approach.](https://arxiv.org/pdf/2002.11223.pdf)
* [Personalized Federated Learning for Intelligent IoT Applications: A Cloud-Edge based Framework](https://arxiv.org/pdf/2002.10671.pdf)
* [Three Approaches for Personalization with Applications to Federated Learning](https://arxiv.org/pdf/2002.10619.pdf)
* [Personalized Federated Learning: A Meta-Learning Approach](https://arxiv.org/pdf/2002.07948.pdf)
* [Towards Federated Learning: Robustness Analytics to Data Heterogeneity](https://arxiv.org/pdf/2002.05038.pdf)
* [Salvaging Federated Learning by Local Adaptation](https://arxiv.org/pdf/2002.04758.pdf)
* [FOCUS: Dealing with Label Quality Disparity in Federated Learning.](https://arxiv.org/pdf/2001.11359.pdf)
* [Overcoming Noisy and Irrelevant Data in Federated Learning.](https://arxiv.org/pdf/2001.08300.pdf)(ICPR 2020)
* [Real-Time Edge Intelligence in the Making: A Collaborative Learning Framework via Federated Meta-Learning.](https://arxiv.org/pdf/2001.03229.pdf)
* [(*) Think Locally, Act Globally: Federated Learning with Local and Global Representations. NeurIPS 2019 Workshop on Federated Learning distinguished student paper award](https://arxiv.org/pdf/2001.01523.pdf)
* [Federated Learning with Personalization Layers](https://arxiv.org/pdf/1912.00818.pdf)
* [Federated Evaluation of On-device Personalization](https://arxiv.org/pdf/1910.10252.pdf)
* [Measure Contribution of Participants in Federated Learning](https://arxiv.org/pdf/1909.08525.pdf)
* [(*) Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification](https://arxiv.org/pdf/1909.06335.pdf)
* [Multi-hop Federated Private Data Augmentation with Sample Compression](https://arxiv.org/pdf/1907.06426.pdf)
* [Distributed Training with Heterogeneous Data: Bridging Median- and Mean-Based Algorithms](https://arxiv.org/pdf/1906.01736.pdf)
* [High Dimensional Restrictive Federated Model Selection with multi-objective Bayesian Optimization over shifted distributions](https://arxiv.org/pdf/1902.08999.pdf)
* [Robust Federated Learning Through Representation Matching and Adaptive Hyper-parameters](https://arxiv.org/pdf/1912.13075.pdf)
* [Towards Efficient Scheduling of Federated Mobile Devices under Computational and Statistical Heterogeneity](https://arxiv.org/pdf/2005.12326.pdf)
* [Client Adaptation improves Federated Learning with Simulated Non-IID Clients](https://arxiv.org/pdf/2007.04806.pdf)

### 4.4 Adaptive Aggregation

* [Asynchronous Federated Learning for Geospatial Applications](https://link.springer.com.remotexs.ntu.edu.sg/chapter/10.1007/978-3-030-14880-5_2) (ECML PKDD Workshop 2018） 
* [Asynchronous Federated Optimization](https://arxiv.org/abs/1903.03934)
* [Adaptive Federated Learning in Resource Constrained Edge Computing Systems](https://arxiv.org/abs/1804.05271) (IEEE Journal on Selected Areas in Communications, 2019）
* [The Distributed Discrete Gaussian Mechanism for Federated Learning with Secure Aggregation](https://arxiv.org/abs/2102.06387) (ICML 2021)

## Part 5: Security
### 5.1 Adversarial Attacks
* [Can You Really Backdoor Federated Learning? ](https://arxiv.org/abs/1911.07963)(NeruIPS 2019)
* [Model Poisoning Attacks in Federated Learning](https://dais-ita.org/sites/default/files/main_secml_model_poison.pdf) (NIPS workshop 2018）
* [An Overview of Federated Deep Learning Privacy Attacks and Defensive Strategies.](https://arxiv.org/pdf/2004.04676.pdf)
* [How To Backdoor Federated Learning.](https://arxiv.org/pdf/1807.00459.pdf)(AISTATS 2020)
* [Deep Models Under the GAN: Information Leakage from Collaborative Deep Learning.](https://arxiv.org/pdf/1702.07464.pdf)(ACM CCS 2017)
* [Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates](https://arxiv.org/pdf/1803.01498.pdf)
* [Deep Leakage from Gradients.](https://papers.nips.cc/paper/9617-deep-leakage-from-gradients.pdf)(NIPS 2019)
* [Comprehensive Privacy Analysis of Deep Learning: Passive and Active White-box Inference Attacks against Centralized and Federated Learning.](https://arxiv.org/pdf/1812.00910.pdf)
* [Beyond Inferring Class Representatives: User-Level Privacy Leakage From Federated Learning.](https://arxiv.org/pdf/1812.00535.pdf)(INFOCOM 2019)
* [Analyzing Federated Learning through an Adversarial Lens.](https://arxiv.org/pdf/1811.12470.pdf)(ICML 2019）
* [Mitigating Sybils in Federated Learning Poisoning.](https://arxiv.org/pdf/1808.04866.pdf)(RAID 2020)
* [RSA: Byzantine-Robust Stochastic Aggregation Methods for Distributed Learning from Heterogeneous Datasets.](https://arxiv.org/abs/1811.03761)(AAAI 2019)
* [A Framework for Evaluating Gradient Leakage Attacks in Federated Learning.](https://arxiv.org/pdf/2004.10397.pdf)
* [Local Model Poisoning Attacks to Byzantine-Robust Federated Learning.](https://arxiv.org/pdf/1911.11815.pdf)
* [Backdoor Attacks on Federated Meta-Learning](https://arxiv.org/pdf/2006.07026.pdf)
* [Towards Realistic Byzantine-Robust Federated Learning.](https://arxiv.org/pdf/2004.04986.pdf)
* [Data Poisoning Attacks on Federated Machine Learning.](https://arxiv.org/pdf/2004.10020.pdf)
* [Exploiting Defenses against GAN-Based Feature Inference Attacks in Federated Learning.](https://arxiv.org/pdf/2004.12571.pdf)
* [Byzantine-Resilient High-Dimensional SGD with Local Iterations on Heterogeneous Data.](https://arxiv.org/pdf/2006.13041.pdf)
* [FedMGDA+: Federated Learning meets Multi-objective Optimization.](https://arxiv.org/pdf/2006.11489.pdf)
* [Free-rider Attacks on Model Aggregation in Federated Learning.](https://arxiv.org/pdf/2006.11901.pdf)
* [FDA3 : Federated Defense Against Adversarial Attacks for Cloud-Based IIoT Applications.](https://arxiv.org/pdf/2006.15632.pdf)
* [Privacy-preserving Weighted Federated Learning within Oracle-Aided MPC Framework.](https://arxiv.org/pdf/2003.07630.pdf)
* [BASGD: Buffered Asynchronous SGD for Byzantine Learning.](https://arxiv.org/pdf/2003.00937.pdf)
* [Stochastic-Sign SGD for Federated Learning with Theoretical Guarantees.](https://arxiv.org/pdf/2002.10940.pdf)
* [Learning to Detect Malicious Clients for Robust Federated Learning.](https://arxiv.org/pdf/2002.00211.pdf)
* [Robust Aggregation for Federated Learning.](https://arxiv.org/pdf/1912.13445.pdf)
* [Towards Deep Federated Defenses Against Malware in Cloud Ecosystems.](https://arxiv.org/pdf/1912.12370.pdf)
* [Attack-Resistant Federated Learning with Residual-based Reweighting.](https://arxiv.org/pdf/1912.11464.pdf)
* [Free-riders in Federated Learning: Attacks and Defenses.](https://arxiv.org/pdf/1911.12560.pdf)
* [Robust Federated Learning with Noisy Communication.](https://arxiv.org/pdf/1911.00251.pdf)
* [Abnormal Client Behavior Detection in Federated Learning.](https://arxiv.org/pdf/1910.09933.pdf)
* [Eavesdrop the Composition Proportion of Training Labels in Federated Learning.](https://arxiv.org/pdf/1910.06044.pdf)
* [Byzantine-Robust Federated Machine Learning through Adaptive Model Averaging.](https://arxiv.org/pdf/1909.05125.pdf)
* [An End-to-End Encrypted Neural Network for Gradient Updates Transmission in Federated Learning.](https://arxiv.org/pdf/1908.08340.pdf)
* [Secure Distributed On-Device Learning Networks With Byzantine Adversaries.](https://arxiv.org/pdf/1906.00887.pdf)
* [Robust Federated Training via Collaborative Machine Teaching using Trusted Instances.](https://arxiv.org/pdf/1905.02941.pdf)
* [Dancing in the Dark: Private Multi-Party Machine Learning in an Untrusted Setting.](https://arxiv.org/pdf/1811.09712.pdf)
* [Inverting Gradients - How easy is it to break privacy in federated learning?](https://arxiv.org/pdf/2003.14053.pdf)

### 5.2 Data Privacy and Confidentiality

* [Gradient-Leaks: Understanding and Controlling Deanonymization in Federated Learning](https://arxiv.org/abs/1805.05838) （NIPS 2019 Workshop)
* [Quantification of the Leakage in Federated Learning](https://arxiv.org/pdf/1910.05467.pdf)

## Part 6: Communication Efficiency
* [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)](https://github.com/roxanneluo/Federated-Learning) [Google] **[Must Read]**
* [Two-Stream Federated Learning: Reduce the Communication Costs](https://ieeexplore.ieee.org/document/8698609) (2018 IEEE VCIP)
* [Federated Learning Based on Dynamic Regularization](https://openreview.net/forum?id=B7v4QMR6Z9w) (ICLR 2021)
* [Federated Learning via Posterior Averaging: A New Perspective and Practical Algorithms](https://openreview.net/forum?id=GFsU8a0sGB) (ICLR 2021)
* [Adaptive Federated Optimization](https://openreview.net/forum?id=LkFG3lB13U5) (ICLR 2021)
* [PowerSGD: Practical Low-Rank Gradient Compression for Distributed Optimization](https://arxiv.org/abs/1905.13727) （NIPS 2019）
* [Deep Gradient Compression: Reducing the Communication Bandwidth for Distributed Training](https://arxiv.org/abs/1712.01887) (ICLR 2018)
* [The Error-Feedback Framework: Better Rates for SGD with Delayed Gradients and Compressed Communication](https://arxiv.org/abs/1909.05350)
* [A Communication Efficient Collaborative Learning Framework for Distributed Features](https://arxiv.org/abs/1912.11187) （NIPS 2019 Workshop)
* [Active Federated Learning](https://arxiv.org/abs/1909.12641) （NIPS 2019 Workshop)
* [Communication-Efficient Distributed Optimization in Networks with Gradient Tracking and Variance Reduction](https://arxiv.org/abs/1909.05844) （NIPS 2019 Workshop)
* [Gradient Descent with Compressed Iterates](https://arxiv.org/abs/1909.04716) （NIPS 2019 Workshop)
* [LAG: Lazily Aggregated Gradient for Communication-Efficient Distributed Learning](https://arxiv.org/abs/1805.09965)
* [Exact Support Recovery in Federated Regression with One-shot Communication](https://arxiv.org/pdf/2006.12583.pdf)
* [DEED: A General Quantization Scheme for Communication Efficiency in Bits](https://arxiv.org/pdf/2006.11401.pdf)
* [Personalized Federated Learning with Moreau Envelopes](https://arxiv.org/pdf/2006.08848.pdf)
* [Towards Flexible Device Participation in Federated Learning for Non-IID Data.](https://arxiv.org/pdf/2006.06954.pdf)
* [A Primal-Dual SGD Algorithm for Distributed Nonconvex Optimization](https://arxiv.org/pdf/2006.03474.pdf)
* [FedSplit: An algorithmic framework for fast federated optimization](https://arxiv.org/pdf/2005.05238.pdf)
* [Distributed Stochastic Non-Convex Optimization: Momentum-Based Variance Reduction](https://arxiv.org/pdf/2005.00224.pdf)
* [On the Outsized Importance of Learning Rates in Local Update Methods.](https://arxiv.org/pdf/2007.00878.pdf)
* [Federated Learning with Compression: Unified Analysis and Sharp Guarantees.](https://arxiv.org/pdf/2007.01154.pdf)
* [From Local SGD to Local Fixed-Point Methods for Federated Learning](https://arxiv.org/pdf/2004.01442.pdf)
* [Federated Residual Learning.](https://arxiv.org/pdf/2003.12880.pdf)
* [Acceleration for Compressed Gradient Descent in Distributed and Federated Optimization.](https://arxiv.org/pdf/2002.11364.pdf)[ICML 2020]
* [Client Selection for Federated Learning with Heterogeneous Resources in Mobile Edge](https://arxiv.org/abs/1804.08333) (FedCS)
* [Hybrid-FL for Wireless Networks: Cooperative Learning Mechanism Using Non-IID Data](https://arxiv.org/abs/1905.07210)
* [LASG: Lazily Aggregated Stochastic Gradients for Communication-Efficient Distributed Learning](https://arxiv.org/pdf/2002.11360.pdf)
* [Uncertainty Principle for Communication Compression in Distributed and Federated Learning and the Search for an Optimal Compressor](https://arxiv.org/pdf/2002.08958.pdf)
* [Dynamic Federated Learning](https://arxiv.org/pdf/2002.08782.pdf)
* [Distributed Optimization over Block-Cyclic Data](https://arxiv.org/pdf/2002.07454.pdf)
* [Federated Composite Optimization](https://arxiv.org/abs/2011.08474) (ICML 2021)
* [Distributed Non-Convex Optimization with Sublinear Speedup under Intermittent Client Availability](https://arxiv.org/pdf/2002.07399.pdf)
* [Federated Learning of a Mixture of Global and Local Models](https://arxiv.org/pdf/2002.05516.pdf)
* [Faster On-Device Training Using New Federated Momentum Algorithm](https://arxiv.org/pdf/2002.02090.pdf)
* [FedDANE: A Federated Newton-Type Method](https://arxiv.org/pdf/2001.01920.pdf)
* [Distributed Fixed Point Methods with Compressed Iterates](https://arxiv.org/pdf/1912.09925.pdf)
* [Primal-dual methods for large-scale and distributed convex optimization and data analytics](https://arxiv.org/pdf/1912.08546.pdf)
* [Parallel Restarted SPIDER - Communication Efficient Distributed Nonconvex Optimization with Optimal Computation Complexity](https://arxiv.org/pdf/1912.06036.pdf)
* [Representation of Federated Learning via Worst-Case Robust Optimization Theory](https://arxiv.org/pdf/1912.05571.pdf)
* [On the Convergence of Local Descent Methods in Federated Learning](https://arxiv.org/pdf/1910.14425.pdf)
* [SCAFFOLD: Stochastic Controlled Averaging for Federated Learning](https://arxiv.org/pdf/1910.06378.pdf)
* [Accelerating Federated Learning via Momentum Gradient Descent](https://arxiv.org/pdf/1910.03197.pdf)
* [Robust Federated Learning in a Heterogeneous Environment](https://arxiv.org/pdf/1906.06629.pdf)
* [Scalable and Differentially Private Distributed Aggregation in the Shuffled Model](https://arxiv.org/pdf/1906.08320.pdf)
* [Differentially Private Learning with Adaptive Clipping](https://arxiv.org/pdf/1905.03871.pdf)
* [Semi-Cyclic Stochastic Gradient Descent](https://arxiv.org/pdf/1904.10120.pdf)
* [Federated Optimization in Heterogeneous Networks](https://arxiv.org/pdf/1812.06127.pdf)
* [Partitioned Variational Inference: A unified framework encompassing federated and continual learning](https://arxiv.org/pdf/1811.11206.pdf)
* [Learning Rate Adaptation for Federated and Differentially Private Learning](https://arxiv.org/pdf/1809.03832.pdf)
* [Communication-Efficient Robust Federated Learning Over Heterogeneous Datasets](https://arxiv.org/pdf/2006.09992.pdf)
* [Don’t Use Large Mini-Batches, Use Local SGD](https://arxiv.org/pdf/1808.07217.pdf)
* [Overlap Local-SGD: An Algorithmic Approach to Hide Communication Delays in Distributed SGD](https://arxiv.org/pdf/2002.09539.pdf)
* [Local SGD With a Communication Overhead Depending Only on the Number of Workers](https://arxiv.org/pdf/2006.02582.pdf)
* [Federated Accelerated Stochastic Gradient Descent](https://arxiv.org/pdf/2006.08950.pdf)
* [Tighter Theory for Local SGD on Identical and Heterogeneous Data](https://arxiv.org/pdf/1909.04746.pdf)
* [STL-SGD: Speeding Up Local SGD with Stagewise Communication Period](https://arxiv.org/pdf/2006.06377.pdf)
* [Cooperative SGD: A unified Framework for the Design and Analysis of Communication-Efficient SGD Algorithms](https://arxiv.org/pdf/1808.07576.pdf)
* [Understanding Unintended Memorization in Federated Learning](http://arxiv.org/pdf/2006.07490.pdf)
* [eSGD: Communication Efficient Distributed Deep Learning on the Edge](https://www.usenix.org/conference/hotedge18/presentation/tao) (USENIX 2018 Workshop)
* [CMFL: Mitigating Communication Overhead for Federated Learning](http://home.cse.ust.hk/~lwangbm/CMFL.pdf)

### 6.1 Compression
* [Expanding the Reach of Federated Learning by Reducing Client Resource Requirements](https://arxiv.org/abs/1812.07210)
* [Federated Learning: Strategies for Improving Communication Efficiency](https://arxiv.org/abs/1610.05492) （NIPS2016 Workshop) [Google]
* [Natural Compression for Distributed Deep Learning](https://arxiv.org/abs/1905.10988)
* [FedPAQ: A Communication-Efficient Federated Learning Method with Periodic Averaging and Quantization](https://arxiv.org/abs/1909.13014)
* [ATOMO: Communication-efficient Learning via Atomic Sparsification](https://arxiv.org/abs/1806.04090)(NIPS 2018）
* [vqSGD: Vector Quantized Stochastic Gradient Descent](https://arxiv.org/abs/1911.07971)
* [QSGD: Communication-efficient SGD via gradient quantization and encoding](https://arxiv.org/abs/1610.02132) （NIPS 2017)
* [Federated Optimization: Distributed Machine Learning for On-Device Intelligence](https://arxiv.org/abs/1610.02527) [Google]
* [Distributed Mean Estimation with Limited Communication](https://arxiv.org/abs/1611.00429) (ICML 2017)
* [Randomized Distributed Mean Estimation: Accuracy vs Communication](https://arxiv.org/abs/1611.07555)
* [Error Feedback Fixes SignSGD and other Gradient Compression Schemes](https://arxiv.org/abs/1901.09847) (ICML 2019）
* [ZipML: Training Linear Models with End-to-End Low Precision, and a Little Bit of Deep Learning](http://proceedings.mlr.press/v70/zhang17e.html) (ICML 2017)

## Part 7: Personalized Federated Learning
### 7.1 Meta Learning
* [Federated Meta-Learning with Fast Convergence and Efficient Communication](https://arxiv.org/abs/1802.07876)
* [Federated Meta-Learning for Recommendation](https://www.semanticscholar.org/paper/Federated-Meta-Learning-for-Recommendation-Chen-Dong/8e21d353ba283bee8fd18285558e5e8df39d46e8#paper-header)
* [Adaptive Gradient-Based Meta-Learning Methods](https://arxiv.org/abs/1906.02717)

### 7.2 Multi-task Learning
* [MOCHA: Federated Multi-Task Learning](https://arxiv.org/abs/1705.10467) (NIPS 2017)
* [Variational Federated Multi-Task Learning](https://arxiv.org/abs/1906.06268)
* [Federated Kernelized Multi-Task Learning](https://mlsys.org/Conferences/2019/doc/2018/30.pdf)
* [Clustered Federated Learning: Model-Agnostic Distributed Multi-Task Optimization under Privacy Constraints](https://arxiv.org/abs/1910.01991) （NIPS 2019 Workshop)
* [Local Stochastic Approximation: A Unified View of Federated Learning and Distributed Multi-Task Reinforcement Learning Algorithms](https://arxiv.org/pdf/2006.13460.pdf)

### 7.3 Hierarchical FL
* [Client-Edge-Cloud Hierarchical Federated Learning](https://arxiv.org/pdf/1905.06641.pdf)
* [(FL startup: Tongdun, HangZhou, China) Knowledge Federation: A Unified and Hierarchical Privacy-Preserving AI Framework.](https://arxiv.org/pdf/2002.01647.pdf)
* [HFEL: Joint Edge Association and Resource Allocation for Cost-Efficient Hierarchical Federated Edge Learning](https://arxiv.org/pdf/2002.11343.pdf)
* [Hierarchical Federated Learning Across Heterogeneous Cellular Networks](https://arxiv.org/pdf/1909.02362.pdf)
* [Enhancing Privacy via Hierarchical Federated Learning](https://arxiv.org/pdf/2004.11361.pdf)
* [Federated learning with hierarchical clustering of local updates to improve training on non-IID data.](https://arxiv.org/pdf/2004.11791.pdf)
* [Federated Hierarchical Hybrid Networks for Clickbait Detection](https://arxiv.org/pdf/1906.00638.pdf)

### 7.4 Transfer Learning
* [Secure Federated Transfer Learning. IEEE Intelligent Systems 2018.](https://arxiv.org/pdf/1812.03337.pdf)
* [Secure and Efficient Federated Transfer Learning](https://arxiv.org/pdf/1910.13271.pdf)
* [Wireless Federated Distillation for Distributed Edge Learning with Heterogeneous Data](https://arxiv.org/pdf/1907.02745.pdf)
* [Proxy Experience Replay: Federated Distillation for Distributed Reinforcement Learning.](https://arxiv.org/pdf/2005.06105.pdf)
* [Cooperative Learning via Federated Distillation over Fading Channels](https://arxiv.org/pdf/2002.01337.pdf)
* [(*) Cronus: Robust and Heterogeneous Collaborative Learning with Black-Box Knowledge Transfer](https://arxiv.org/pdf/1912.11279.pdf)
* [Federated Reinforcement Distillation with Proxy Experience Memory](https://arxiv.org/pdf/1907.06536.pdf)
* [Federated Continual Learning with Weighted Inter-client Transfer](https://openreview.net/forum?id=xWr8qQCJU3m) (ICML 2021)

## Part 8 Decentralization & Incentive Mechanism

### 8.1 Decentralized
* [Communication Compression for Decentralized Training](https://arxiv.org/abs/1803.06443) （NIPS 2018）
* [𝙳𝚎𝚎𝚙𝚂𝚚𝚞𝚎𝚎𝚣𝚎: Decentralization Meets Error-Compensated Compression](https://arxiv.org/abs/1907.07346)
* [Central Server Free Federated Learning over Single-sided Trust Social Networks](https://arxiv.org/pdf/1910.04956.pdf)
* [Can Decentralized Algorithms Outperform Centralized Algorithms? A Case Study for Decentralized Parallel Stochastic Gradient Descent](https://arxiv.org/pdf/1705.09056.pdf)
* [Multi-consensus Decentralized Accelerated Gradient Descent](https://arxiv.org/pdf/2005.00797.pdf)
* [Decentralized Bayesian Learning over Graphs.](https://arxiv.org/pdf/1905.10466.pdf)
* [BrainTorrent: A Peer-to-Peer Environment for Decentralized Federated Learning](https://arxiv.org/pdf/1905.06731.pdf)
* [Biscotti: A Ledger for Private and Secure Peer-to-Peer Machine Learning](https://arxiv.org/pdf/1811.09904.pdf)
* [Matcha: Speeding Up Decentralized SGD via Matching Decomposition Sampling](https://arxiv.org/pdf/1905.09435.pdf)

### 8.2 Incentive Mechanism
* [Incentive Mechanism for Reliable Federated Learning: A Joint Optimization Approach to Combining Reputation and Contract Theory](https://ieeexplore.ieee.org/document/8832210)
* [Motivating Workers in Federated Learning: A Stackelberg Game Perspective](https://arxiv.org/abs/1908.03092)
* [Incentive Design for Efficient Federated Learning in Mobile Networks: A Contract Theory Approach](https://arxiv.org/abs/1905.07479)
* [Fair Resource Allocation in Federated Learning](https://arxiv.org/pdf/1905.10497v1.pdf)
* [FMore: An Incentive Scheme of Multi-dimensional Auction for Federated Learning in MEC.](https://arxiv.org/pdf/2002.09699.pdf)(ICDCS 2020)
* [Toward an Automated Auction Framework for Wireless Federated Learning Services Market](https://arxiv.org/pdf/1912.06370.pdf)
* [Federated Learning for Edge Networks: Resource Optimization and Incentive Mechanism](https://arxiv.org/pdf/1911.05642.pdf)
* [A Learning-based Incentive Mechanism forFederated Learning](https://www.u-aizu.ac.jp/~pengli/files/fl_incentive_iot.pdf)
* [A Crowdsourcing Framework for On-Device Federated Learning](https://arxiv.org/pdf/1911.01046.pdf)
* [Rewarding High-Quality Data via Influence Functions](https://arxiv.org/abs/1908.11598)
* [Joint Service Pricing and Cooperative Relay Communication for Federated Learning](https://arxiv.org/abs/1811.12082)
* [Measure Contribution of Participants in Federated Learning](https://arxiv.org/abs/1909.08525)
* [DeepChain: Auditable and Privacy-Preserving Deep Learning with Blockchain-based Incentive](https://eprint.iacr.org/2018/679.pdf)

  
## Part 9: Vertical Federated Learning

* [A Quasi-Newton Method Based Vertical Federated Learning Framework for Logistic Regression](https://arxiv.org/abs/1912.00513) （NIPS 2019 Workshop)
* [SecureBoost: A Lossless Federated Learning Framework](https://arxiv.org/pdf/1901.08755.pdf)
* [Parallel Distributed Logistic Regression for Vertical Federated Learning without Third-Party Coordinator](https://arxiv.org/pdf/1911.09824.pdf)
* [Private federated learning on vertically partitioned data via entity resolution and additively homomorphic encryption](https://arxiv.org/pdf/1711.10677.pdf)
* [Entity Resolution and Federated Learning get a Federated Resolution.](https://arxiv.org/pdf/1803.04035.pdf)
* [Multi-Participant Multi-Class Vertical Federated Learning](https://arxiv.org/pdf/2001.11154.pdf)
* [A Communication-Efficient Collaborative Learning Framework for Distributed Features](https://arxiv.org/pdf/1912.11187.pdf)
* [Asymmetrical Vertical Federated Learning](https://arxiv.org/pdf/2004.07427.pdf)
* [VAFL: a Method of Vertical Asynchronous Federated Learning](https://arxiv.org/abs/2007.06081) (ICML workshop on FL, 2020)
* [SplitFed: When Federated Learning Meets Split Learning](https://arxiv.org/abs/2004.12088v2)
* [Privacy Enhanced Multimodal Neural Representations for Emotion Recognition](https://arxiv.org/abs/1910.13212)
* [PrivyNet: A Flexible Framework for Privacy-Preserving Deep Neural Network Training](https://arxiv.org/abs/1709.06161)
* [One Pixel Image and RF Signal Based Split Learning for mmWave Received Power Prediction](https://arxiv.org/abs/1911.01682)
* [Stochastic Distributed Optimization for Machine Learning from Decentralized Features](https://arxiv.org/abs/1812.06415)

### Part 10: Wireless Communication and Cloud Computing

* [Mix2FLD: Downlink Federated Learning After Uplink Federated Distillation With Two-Way Mixup](https://arxiv.org/pdf/2006.09801.pdf)
* [Wireless Communications for Collaborative Federated Learning in the Internet of Things](https://arxiv.org/pdf/2006.02499.pdf)
* [Democratizing the Edge: A Pervasive Edge Computing Framework](https://arxiv.org/pdf/2007.00641.pdf)
* [UVeQFed: Universal Vector Quantization for Federated Learning](https://arxiv.org/pdf/2006.03262.pdf)
* [Federated Deep Learning Framework For Hybrid Beamforming in mm-Wave Massive MIMO](https://arxiv.org/pdf/2005.09969.pdf)
* [Efficient Federated Learning over Multiple Access Channel with Differential Privacy Constraints](https://arxiv.org/pdf/2005.07776.pdf)
* [A Secure Federated Learning Framework for 5G Networks](https://arxiv.org/pdf/2005.05752.pdf)
* [Federated Learning and Wireless Communications](https://arxiv.org/pdf/2005.05265.pdf)
* [Lightwave Power Transfer for Federated Learning-based Wireless Networks](https://arxiv.org/pdf/2005.03977.pdf)
* [Towards Ubiquitous AI in 6G with Federated Learning](https://arxiv.org/pdf/2004.13563.pdf)
* [Optimizing Over-the-Air Computation in IRS-Aided C-RAN Systems](https://arxiv.org/pdf/2004.09168.pdf)
* [Network-Aware Optimization of Distributed Learning for Fog Computing](https://arxiv.org/pdf/2004.08488.pdf)
* [On the Design of Communication Efficient Federated Learning over Wireless Networks](https://arxiv.org/pdf/2004.07351.pdf)
* [Federated Machine Learning for Intelligent IoT via Reconfigurable Intelligent Surface](https://arxiv.org/pdf/2004.05843.pdf)
* [Client Selection and Bandwidth Allocation in Wireless Federated Learning Networks: A Long-Term Perspective](https://arxiv.org/pdf/2004.04314.pdf)
* [Resource Management for Blockchain-enabled Federated Learning: A Deep Reinforcement Learning Approach](https://arxiv.org/pdf/2004.04104.pdf)
* [A Blockchain-based Decentralized Federated Learning Framework with Committee Consensus](https://arxiv.org/pdf/2004.00773.pdf)
* [Scheduling for Cellular Federated Edge Learning with Importance and Channel.](https://arxiv.org/pdf/2004.00490.pdf)
* [Differentially Private Federated Learning for Resource-Constrained Internet of Things.](https://arxiv.org/pdf/2003.12705.pdf)
* [Federated Learning for Task and Resource Allocation in Wireless High Altitude Balloon Networks.](https://arxiv.org/pdf/2003.09375.pdf)
* [Gradient Estimation for Federated Learning over Massive MIMO Communication Systems](https://arxiv.org/pdf/2003.08059.pdf)
* [Adaptive Federated Learning With Gradient Compression in Uplink NOMA](https://arxiv.org/pdf/2003.01344.pdf)
* [Performance Analysis and Optimization in Privacy-Preserving Federated Learning](https://arxiv.org/pdf/2003.00229.pdf)
* [Energy-Efficient Federated Edge Learning with Joint Communication and Computation Design](https://arxiv.org/pdf/2003.00199.pdf)
* [Federated Over-the-Air Subspace Learning and Tracking from Incomplete Data](https://arxiv.org/pdf/2002.12873.pdf)
* [Decentralized Federated Learning via SGD over Wireless D2D Networks](https://arxiv.org/pdf/2002.12507.pdf)
* [Federated Learning in the Sky: Joint Power Allocation and Scheduling with UAV Swarms](https://arxiv.org/pdf/2002.08196.pdf)
* [Wireless Federated Learning with Local Differential Privacy](https://arxiv.org/pdf/2002.05151.pdf)
* [Federated Learning under Channel Uncertainty: Joint Client Scheduling and Resource Allocation.](https://arxiv.org/pdf/2002.01337.pdf)
* [Learning from Peers at the Wireless Edge](https://arxiv.org/pdf/2001.11567.pdf)
* [Convergence of Update Aware Device Scheduling for Federated Learning at the Wireless Edge](https://arxiv.org/pdf/2001.10402.pdf)
* [Communication Efficient Federated Learning over Multiple Access Channels](https://arxiv.org/pdf/2001.08737.pdf)
* [Convergence Time Optimization for Federated Learning over Wireless Networks](https://arxiv.org/pdf/2001.07845.pdf)
* [One-Bit Over-the-Air Aggregation for Communication-Efficient Federated Edge Learning: Design and Convergence Analysis](https://arxiv.org/pdf/2001.05713.pdf)
* [Federated Learning with Cooperating Devices: A Consensus Approach for Massive IoT Networks.](https://arxiv.org/pdf/1912.13163.pdf)(IEEE Internet of Things Journal. 2020)
* [Asynchronous Federated Learning with Differential Privacy for Edge Intelligence](https://arxiv.org/pdf/1912.07902.pdf)
* [Federated learning with multichannel ALOHA](https://arxiv.org/pdf/1912.06273.pdf)
* [Federated Learning with Autotuned Communication-Efficient Secure Aggregation](https://arxiv.org/pdf/1912.00131.pdf)
* [Bandwidth Slicing to Boost Federated Learning in Edge Computing](https://arxiv.org/pdf/1911.07615.pdf)
* [Energy Efficient Federated Learning Over Wireless Communication Networks](https://arxiv.org/pdf/1911.02417.pdf)
* [Device Scheduling with Fast Convergence for Wireless Federated Learning](https://arxiv.org/pdf/1911.00856.pdf)
* [Energy-Aware Analog Aggregation for Federated Learning with Redundant Data](https://arxiv.org/pdf/1911.00188.pdf)
* [Age-Based Scheduling Policy for Federated Learning in Mobile Edge Networks](https://arxiv.org/pdf/1910.14648.pdf)
* [Federated Learning over Wireless Networks: Convergence Analysis and Resource Allocation](https://arxiv.org/pdf/1910.13067.pdf)
* [Federated Learning over Wireless Networks: Optimization Model Design and Analysis](http://networking.khu.ac.kr/layouts/net/publications/data/2019\)Federated%20Learning%20over%20Wireless%20Network.pdf)
* [Resource Allocation in Mobility-Aware Federated Learning Networks: A Deep Reinforcement Learning Approach](https://arxiv.org/pdf/1910.09172.pdf)
* [Reliable Federated Learning for Mobile Networks](https://arxiv.org/pdf/1910.06837.pdf)
* [Cell-Free Massive MIMO for Wireless Federated Learning](https://arxiv.org/pdf/1909.12567.pdf)
* [A Joint Learning and Communications Framework for Federated Learning over Wireless Networks](https://arxiv.org/pdf/1909.07972.pdf)
* [On Safeguarding Privacy and Security in the Framework of Federated Learning](https://arxiv.org/pdf/1909.06512.pdf)
* [Scheduling Policies for Federated Learning in Wireless Networks](https://arxiv.org/pdf/1908.06287.pdf)
* [Federated Learning with Additional Mechanisms on Clients to Reduce Communication Costs](https://arxiv.org/pdf/1908.05891.pdf)
* [Energy-Efficient Radio Resource Allocation for Federated Edge Learning](https://arxiv.org/pdf/1907.06040.pdf)
* [Mobile Edge Computing, Blockchain and Reputation-based Crowdsourcing IoT Federated Learning: A Secure, Decentralized and Privacy-preserving System](https://arxiv.org/pdf/1906.10893.pdf)
* [Active Learning Solution on Distributed Edge Computing](https://arxiv.org/pdf/1906.10718.pdf)
* [Fast Uplink Grant for NOMA: a Federated Learning based Approach](https://arxiv.org/pdf/1905.04519.pdf)
* [Machine Learning at the Wireless Edge: Distributed Stochastic Gradient Descent Over-the-Air](https://arxiv.org/pdf/1901.00844.pdf)
* [Broadband Analog Aggregation for Low-Latency Federated Edge Learning](https://arxiv.org/pdf/1812.11494.pdf)
* [Federated Echo State Learning for Minimizing Breaks in Presence in Wireless Virtual Reality Networks](https://arxiv.org/pdf/1812.01202.pdf)
* [Joint Service Pricing and Cooperative Relay Communication for Federated Learning](https://arxiv.org/pdf/1811.12082.pdf)
* [In-Edge AI: Intelligentizing Mobile Edge Computing, Caching and Communication by Federated Learning](https://arxiv.org/pdf/1809.07857.pdf)
* [Asynchronous Task Allocation for Federated and Parallelized Mobile Edge Learning](https://arxiv.org/pdf/1905.01656.pdf)
* [Ask to upload some data from client to server Efficient Training Management for Mobile Crowd-Machine Learning: A Deep Reinforcement Learning Approach](https://arxiv.org/abs/1812.03633)
* [Low-latency Broadband Analog Aggregation For Federated Edge Learning](https://arxiv.org/abs/1812.11494)
* [Federated Learning over Wireless Fading Channels](https://arxiv.org/pdf/1907.09769.pdf)
* [Federated Learning via Over-the-Air Computation](https://arxiv.org/abs/1812.11750)

## Part 11: Federated with Deep learning

### 11.1 Neural Architecture Search(NAS)
* [FedNAS: Federated Deep Learning via Neural Architecture Search.](https://arxiv.org/pdf/2004.08546.pdf)(CVPR 2020)
* [Real-time Federated Evolutionary Neural Architecture Search.](https://arxiv.org/pdf/2003.02793.pdf)
* [Federated Neural Architecture Search.](https://arxiv.org/pdf/2002.06352.pdf)
* [Differentially-private Federated Neural Architecture Search.](https://arxiv.org/pdf/2006.10559.pdf)

### 11.2 Graph Neural Network(GNN)
* [SGNN: A Graph Neural Network Based Federated Learning Approach by Hiding Structure](https://ieeexplore.ieee.org/document/9005983) (Big Data)
* [FedGNN: Federated Graph Neural Network for Privacy-Preserving Recommendation](https://arxiv.org/abs/2102.04925)
* [GraphFederator: Federated Visual Analysis for Multi-party Graphs.](https://arxiv.org/abs/2008.11989)
* [FedE: Embedding Knowledge Graphs in Federated Setting](https://arxiv.org/abs/2010.12882)
* [ASFGNN: Automated Separated-Federated Graph Neural Network](https://arxiv.org/abs/2011.03248)
* [GraphFL: A Federated Learning Framework for Semi-Supervised Node Classification on Graphs](https://arxiv.org/abs/2012.04187)
* [Peer-to-peer Federated Learning on Graphs](https://arxiv.org/abs/1901.11173)
* [Towards Federated Graph Learning for Collaborative Financial Crimes Detection](https://arxiv.org/abs/1909.12946)

## Part 12: FL system & Library & Courses
### 12.1 System
* [Towards Federated Learning at Scale: System Design](https://arxiv.org/abs/1902.01046) **[Must Read]**
* [Demonstration of Federated Learning in a Resource-Constrained Networked Environment](https://ieeexplore.ieee.org/document/8784064)
* [Applied Federated Learning: Improving Google Keyboard Query Suggestions](https://arxiv.org/abs/1812.02903)
* [Federated Learning and Differential Privacy: Software tools analysis, the Sherpa.ai FL framework and methodological guidelines for preserving data privacy](https://arxiv.org/abs/2007.00914)
* [FedML: A Research Library and Benchmark for Federated Machine Learning](https://arxiv.org/pdf/2007.13518.pdf)
* [FLeet: Online Federated Learning via Staleness Awareness and Performance Prediction.](https://arxiv.org/pdf/2006.07273.pdf)
* [Heterogeneity-Aware Federated Learning](https://arxiv.org/pdf/2006.06983.pdf)
* [Decentralised Learning from Independent Multi-Domain Labels for Person Re-Identification](https://arxiv.org/pdf/2006.04150.pdf)
* [[startup] Industrial Federated Learning -- Requirements and System Design](https://arxiv.org/pdf/2005.06850.pdf)
* [(*) TiFL: A Tier-based Federated Learning System.](https://arxiv.org/pdf/2001.09249.pdf)(HPDC 2020)
* [Adaptive Gradient Sparsification for Efficient Federated Learning: An Online Learning Approach](https://arxiv.org/pdf/2001.04756.pdf)(ICDCS 2020)
* [Quantifying the Performance of Federated Transfer Learning](https://arxiv.org/pdf/1912.12795.pdf)
* [ELFISH: Resource-Aware Federated Learning on Heterogeneous Edge Devices](https://arxiv.org/pdf/1912.01684.pdf)
* [Privacy is What We Care About: Experimental Investigation of Federated Learning on Edge Devices](https://arxiv.org/pdf/1911.04559.pdf)
* [Substra: a framework for privacy-preserving, traceable and collaborative Machine Learning](https://arxiv.org/pdf/1910.11567.pdf)
* [BAFFLE : Blockchain Based Aggregator Free Federated Learning](https://arxiv.org/pdf/1909.07452.pdf)
* [Functional Federated Learning in Erlang (ffl-erl)](https://arxiv.org/pdf/1808.08143.pdf)
* [HierTrain: Fast Hierarchical Edge AI Learning With Hybrid Parallelism in Mobile-Edge-Cloud Computing](https://arxiv.org/pdf/2003.09876.pdf)

### 12.2 Courses

* [Applied Cryptography](https://www.udacity.com/course/applied-cryptography--cs387)
* [A Brief Introduction to Differential Privacy](https://medium.com/georgian-impact-blog/a-brief-introduction-to-differential-privacy-eacf8722283b)
* [Deep Learning with Differential Privacy.](http://doi.acm.org/10.1145/2976749.2978318)
* [Building Safe A.I.](http://iamtrask.github.io/2017/03/17/safe-ai/)
  * A Tutorial for Encrypted Deep Learning
  * Use Homomorphic Encryption (HE)

* [Private Deep Learning with MPC](https://mortendahl.github.io/2017/04/17/private-deep-learning-with-mpc/)
  * A Simple Tutorial from Scratch
  * Use Multiparty Compuation (MPC)

* [Private Image Analysis with MPC](https://mortendahl.github.io/2017/09/19/private-image-analysis-with-mpc/)
  * Training CNNs on Sensitive Data
  * Use SPDZ as MPC protocol

### 13.2 Secret Sharing
* [Simple Introduction to Sharmir's Secret Sharing and Lagrange Interpolation](https://www.youtube.com/watch?v=kkMps3X_tEE)
* [Secret Sharing, Part 1](https://mortendahl.github.io/2017/06/04/secret-sharing-part1/): Shamir's Secret Sharing & Packed Variant
* [Secret Sharing, Part 2](https://mortendahl.github.io/2017/06/24/secret-sharing-part2/): Improve efficiency
* [Secret Sharing, Part 3](https://mortendahl.github.io/2017/08/13/secret-sharing-part3/)


## Part 13: Secure Multi-party Computation(MPC)
### 13.1 Differential Privacy
* [Learning Differentially Private Recurrent Language Models](https://arxiv.org/abs/1710.06963)
* [Federated Learning with Bayesian Differential Privacy](https://arxiv.org/abs/1911.10071) （NIPS 2019 Workshop)
* [Private Federated Learning with Domain Adaptation](https://arxiv.org/abs/1912.06733) （NIPS 2019 Workshop)
* [cpSGD: Communication-efficient and differentially-private distributed SGD](https://arxiv.org/abs/1805.10559)
* [Practical Secure Aggregation for Federated Learning on User-Held Data.](https://arxiv.org/pdf/1611.04482.pdf)（NIPS 2016 Workshop)
* [Differentially Private Federated Learning: A Client Level Perspective.](https://arxiv.org/pdf/1712.07557.pdf)（NIPS 2017 Workshop)
* [Exploiting Unintended Feature Leakage in Collaborative Learning.](https://arxiv.org/pdf/1805.04049.pdf)（S&P 2019）
* [A Hybrid Approach to Privacy-Preserving Federated Learning.](https://arxiv.org/pdf/1812.03224.pdf)（AISec 2019）
* [A generic framework for privacy preserving deep learning.](https://arxiv.org/pdf/1811.04017.pdf)（PPML 2018）
* [Federated Generative Privacy.](https://arxiv.org/pdf/1910.08385.pdf)（IJCAI 2019 FL Workshop)
* [Enhancing the Privacy of Federated Learning with Sketching.](https://arxiv.org/pdf/1911.01812.pdf)
* [https://aisec.cc/](https://arxiv.org/pdf/1912.05897.pdf)
* [Anonymizing Data for Privacy-Preserving Federated Learning.](https://arxiv.org/pdf/2002.09096.pdf)
* [Practical and Bilateral Privacy-preserving Federated Learning.](https://arxiv.org/pdf/2002.09843.pdf)
* [Decentralized Policy-Based Private Analytics.](https://arxiv.org/pdf/2003.06612.pdf)
* [FedSel: Federated SGD under Local Differential Privacy with Top-k Dimension Selection.](https://arxiv.org/pdf/2003.10637.pdf)（DASFAA 2020)
* [Learn to Forget: User-Level Memorization Elimination in Federated Learning.](https://arxiv.org/pdf/2003.10933.pdf)
* [LDP-Fed: Federated Learning with Local Differential Privacy.](https://arxiv.org/pdf/2006.03637.pdf)（EdgeSys 2020)
* [PrivFL: Practical Privacy-preserving Federated Regressions on High-dimensional Data over Mobile Networks.](https://arxiv.org/pdf/2004.02264.pdf)
* [Local Differential Privacy based Federated Learning for Internet of Things.](https://arxiv.org/pdf/2004.08856.pdf)
* [Differentially Private AirComp Federated Learning with Power Adaptation Harnessing Receiver Noise.](https://arxiv.org/pdf/2004.06337.pdf)
* [Decentralized Differentially Private Segmentation with PATE.](https://arxiv.org/pdf/2004.06567.pdf)（MICCAI 2020 Under Review)
* [Privacy Preserving Distributed Machine Learning with Federated Learning.](https://arxiv.org/pdf/2004.12108.pdf)
* [Exploring Private Federated Learning with Laplacian Smoothing.](https://arxiv.org/pdf/2005.00218.pdf)
* [Information-Theoretic Bounds on the Generalization Error and Privacy Leakage in Federated Learning.](https://arxiv.org/pdf/2005.02503.pdf)
* [Efficient Privacy Preserving Edge Computing Framework for Image Classification.](https://arxiv.org/pdf/2005.04563.pdf)
* [A Distributed Trust Framework for Privacy-Preserving Machine Learning.](https://arxiv.org/pdf/2006.02456.pdf)
* [Secure Byzantine-Robust Machine Learning.](https://arxiv.org/pdf/2006.04747.pdf)
* [ARIANN: Low-Interaction Privacy-Preserving Deep Learning via Function Secret Sharing.](https://arxiv.org/pdf/2006.04593.pdf)
* [Privacy For Free: Wireless Federated Learning Via Uncoded Transmission With Adaptive Power Control.](https://arxiv.org/pdf/2006.05459.pdf)
* [(*) Distributed Differentially Private Averaging with Improved Utility and Robustness to Malicious Parties.](https://arxiv.org/pdf/2006.07218.pdf)
* [GS-WGAN: A Gradient-Sanitized Approach for Learning Differentially Private Generators.](https://arxiv.org/pdf/2006.08848.pdf)
* [Federated Learning with Differential Privacy:Algorithms and Performance Analysis](https://arxiv.org/pdf/1911.00222.pdf)

### 13.2 Secret Sharing
* [Simple Introduction to Sharmir's Secret Sharing and Lagrange Interpolation](https://www.youtube.com/watch?v=kkMps3X_tEE)
* [Secret Sharing, Part 1](https://mortendahl.github.io/2017/06/04/secret-sharing-part1/): Shamir's Secret Sharing & Packed Variant
* [Secret Sharing, Part 2](https://mortendahl.github.io/2017/06/24/secret-sharing-part2/): Improve efficiency
* [Secret Sharing, Part 3](https://mortendahl.github.io/2017/08/13/secret-sharing-part3/)


## Part 14: Applications

* [Federated Learning Approach for Mobile Packet Classification](https://arxiv.org/abs/1907.13113)
* [Federated Learning for Ranking Browser History Suggestions](https://arxiv.org/abs/1911.11807) (NIPS 2019 Workshop)

### 14.1 Healthcare
* [HHHFL: Hierarchical Heterogeneous Horizontal Federated Learning for Electroencephalography](https://arxiv.org/abs/1909.05784) (NIPS 2019 Workshop)
* [Learn Electronic Health Records by Fully Decentralized Federated Learning](https://arxiv.org/abs/1912.01792) (NIPS 2019 Workshop)
* [Patient Clustering Improves Efficiency of Federated Machine Learning to predict mortality and hospital stay time using distributed Electronic Medical Records](https://arxiv.org/ftp/arxiv/papers/1903/1903.09296.pdf) [[News]](https://venturebeat.com/2019/03/25/federated-learning-technique-predicts-hospital-stay-and-patient-mortality/)
* [Federated learning of predictive models from federated Electronic Health Records.](https://www.ncbi.nlm.nih.gov/pubmed/29500022)
* [FedHealth: A Federated Transfer Learning Framework for Wearable Healthcare](https://arxiv.org/pdf/1907.09173.pdf)
* [Multi-Institutional Deep Learning Modeling Without Sharing Patient Data: A Feasibility Study on Brain Tumor Segmentation](https://arxiv.org/pdf/1810.04304.pdf)
* [NVIDIA Clara Federated Learning to Deliver AI to Hospitals While Protecting Patient Data](https://blogs.nvidia.com/blog/2019/12/01/clara-federated-learning/)
* [What is Federated Learning](https://blogs.nvidia.com/blog/2019/10/13/what-is-federated-learning/)
* [Split learning for health: Distributed deep learning without sharing raw patient data](https://arxiv.org/pdf/1812.00564)
* [Two-stage Federated Phenotyping and Patient Representation Learning](https://www.aclweb.org/anthology/W19-5030.pdf) (ACL 2019)
* [Federated Tensor Factorization for Computational Phenotyping](https://dl.acm.org/doi/10.1145/3097983.3098118) (SIGKDD 2017)
* [FedHealth- A Federated Transfer Learning Framework for Wearable Healthcare](https://arxiv.org/abs/1907.09173) (ICJAI 2019 workshop)
* [Multi-Institutional Deep Learning Modeling Without Sharing Patient Data- A Feasibility Study on Brain Tumor Segmentation](https://arxiv.org/abs/1810.04304) (MICCAI'18 Workshop)
* [Federated Patient Hashing](https://aaai.org/ojs/index.php/AAAI/article/view/6121) (AAAI 2020)
  
### 14.2 Natual Language Processing

Google

* [Federated Learning for Mobile Keyboard Prediction](https://arxiv.org/abs/1811.03604)
* [Applied Federated Learning: Improving Google Keyboard Query Suggestions](https://arxiv.org/abs/1812.02903)
* [Federated Learning Of Out-Of-Vocabulary Words](https://arxiv.org/abs/1903.10635)
* [Federated Learning for Emoji Prediction in a Mobile Keyboard](https://arxiv.org/abs/1906.04329)

Snips

* [Federated Learning for Wake Keyword Spotting](https://arxiv.org/pdf/1810.05512.pdf) [[Blog]](https://medium.com/snips-ai/federated-learning-for-wake-word-detection-c8b8c5cdd2c5) [[Github]](https://github.com/snipsco/keyword-spotting-research-datasets)

### 14.3 Computer Vision

* [Performance Optimization for Federated Person Re-identification via Benchmark Analysis](https://arxiv.org/abs/2008.11560) (ACMMM 2020) [[Github]](https://github.com/cap-ntu/FedReID)
* [Real-World Image Datasets for Federated Learning](https://arxiv.org/abs/1910.11089)
* [FedVision- An Online Visual Object Detection Platform Powered by Federated Learning](https://arxiv.org/abs/2001.06202) (IAAI20)
* [Federated Learning for Vision-and-Language Grounding Problems](http://web.pkusz.edu.cn/adsp/files/2019/11/AAAI-FenglinL.1027.pdf) (AAAI20)

### 14.4 Recommendation

 * [Federated Collaborative Filtering for Privacy-Preserving Personalized Recommendation System](https://arxiv.org/abs/1901.09888)
 * [Federated Meta-Learning with Fast Convergence and Efficient Communication](https://arxiv.org/abs/1802.07876)

### 14.5 Industrial

* [Turbofan POC: Predictive Maintenance of Turbofan Engines using Federated Learning](https://github.com/matthiaslau/Turbofan-Federated-Learning-POC)
* [Turbofan Tycoon Simulation by Cloudera/FastForwardLabs](https://turbofan.fastforwardlabs.com/)
* [Firefox Search Bar](https://florian.github.io/federated-learning/)

## Part 15: Organizations and Companies
### 15.1 国内篇
##### 微众银行开源 [FATE](https://github.com/FederatedAI/FATE) 框架.
Qiang Yang, Tianjian Chen, Yang Liu, Yongxin Tong.
- [《Federated machine learning: Concept and applications》](https://dl.acm.org/doi/abs/10.1145/3298981)
- [《Secureboost: A lossless federated learning framework》](https://ieeexplore.ieee.org/abstract/document/9440789)


##### 字节跳动开源 [FedLearner](https://github.com/bytedance/fedlearner) 框架.
Jiankai Sun, Weihao Gao, Hongyi Zhang, Junyuan Xie.[《Label Leakage and Protection in Two-party Split learning》](https://arxiv.org/pdf/2102.08504.pdf)


##### 华控清交 PrivPy 多方计算平台
Yi Li, Wei Xu.[《PrivPy: General and Scalable Privacy-Preserving Data Mining》](https://dl.acm.org/doi/pdf/10.1145/3292500.3330920)

##### 同盾科技 同盾志邦知识联邦平台
Hongyu Li, Dan Meng, Hong Wang, Xiaolin Li.
- [《Knowledge Federation: A Unified and Hierarchical Privacy-Preserving AI Framework》](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9194566)
- [《FedMONN: Meta Operation Neural Network for Secure Federated Aggregation》](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9408024)

##### 百度 [MesaTEE](https://anquan.baidu.com/product/mesatee) 安全计算平台
Tongxin Li, Yu Ding, Yulong Zhang, Tao Wei.[《gbdt-rs: Fast and Trustworthy Gradient Boosting Decision Tree》](https://www.ieee-security.org/TC/SP2019/posters/hotcrp_sp19posters-final11.pdf)

##### 矩阵元 [Rosetta](https://github.com/LatticeX-Foundation/Rosetta) 隐私开源框架

##### 百度 [PaddlePaddle](https://github.com/PaddlePaddle/PaddleFL) 开源联邦学习框架

##### 蚂蚁区块链科技 [蚂蚁链摩斯安全计算平台](https://antchain.antgroup.com/products/morse)

##### 阿里云 [DataTrust](https://dp.alibaba.com/index) 隐私增强计算平台

##### 百度百度点石联邦学习平台

##### 富数科技 阿凡达安全计算平台



##### 香港理工大学

[《FedVision: An Online Visual Object Detection Platform Powered by Federated Learning》](https://ojs.aaai.org//index.php/AAAI/article/view/7021)

[《BatchCrypt: Efficient Homomorphic Encryption for Cross-Silo Federated Learning》](https://www.usenix.org/system/files/atc20-zhang-chengliang.pdf)

[《Abnormal Client Behavior Detection in Federated Learning》](https://arxiv.org/pdf/1910.09933.pdf)



##### 北京航空航天大学

[《Federated machine learning: Concept and applications》](https://dl.acm.org/doi/abs/10.1145/3298981)

[《Failure Prediction in Production Line Based on Federated Learning: An Empirical Study》](https://arxiv.org/pdf/2101.11715.pdf)



### 15.2 国际篇

Google 提出 Federated Learning.
H. Brendan McMahan. Daniel Ramage. Jakub Konečný. Kallista A. Bonawitz. Hubert Eichner.

[《Communication-efficient learning of deep networks from decentralized data》](https://arxiv.org/abs/1602.05629)

[《Federated Learning: Strategies for Improving Communication Efficiency》](https://arxiv.org/abs/1610.05492)

[《Advances and Open Problems in Federated Learning》](https://arxiv.org/pdf/1912.04977.pdf)

[《Towards Federated Learning at Scale: System Design》](https://arxiv.org/abs/1902.01046)

[《Differentially Private Learning with Adaptive Clipping》](https://arxiv.org/pdf/1905.03871.pdf)

......（更多联邦学习相关文章请自行搜索 Google Scholar）



#### Cornell University.

Antonio Marcedone.

[《Practical Secure Aggregation for Federated Learning on User-Held Data》](https://arxiv.org/pdf/1611.04482.pdf)

[《Practical Secure Aggregation for Privacy-Preserving Machine Learning》](https://academic.microsoft.com/paper/2949130532/citedby/search?q=Practical%20Secure%20Aggregation%20for%20Privacy%20Preserving%20Machine%20Learning.&qe=RId%253D2949130532&f=&orderBy=0)

Eugene Bagdasaryan, Andreas Veit, Yiqing Hua, Deborah Estrin, Vitaly Shmatikov.

[《How To Backdoor Federated Learning》](https://arxiv.org/pdf/1807.00459.pdf)

[《Differential privacy has disparate impact on model accuracy》](https://proceedings.neurips.cc/paper/2019/hash/fc0de4e0396fff257ea362983c2dda5a-Abstract.html)

Ziteng Sun.

[《Can you really backdoor federated learning?》](https://arxiv.org/abs/1911.07963)
