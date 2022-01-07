## 基于GAN和LSTM的压强场时空序列预测

-----------------------------------------------------------------------------------

#### 关于数据集

以二维圆柱绕流作为实际问题考虑压强场的时空序列预测，压强变化和流体流动由相应的Navier-Stokes方程驱动. 

流体流动数据（流速和压强）来自于实验测量得到的高保真数据集. 

**数据集来源**：[M. Raissi, P. ](https://www.sciencedirect.com/science/article/pii/S0021999118307125)[Perdikaris](https://www.sciencedirect.com/science/article/pii/S0021999118307125)[, G. E. ](https://www.sciencedirect.com/science/article/pii/S0021999118307125)[Karniadakis](https://www.sciencedirect.com/science/article/pii/S0021999118307125)[, Physics-informed neural networks: a deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378 (2019) 686–707.](https://www.sciencedirect.com/science/article/pii/S0021999118307125)

原始数据存在放`Data/cylinder_nektar_wake.mat`中，提取得到的压强场数据存放在`Data/Pressure.npy`

-----------------------------------------------------------------------------------

#### 关于其它文件说明

* **figure**
    * 包含问题说明示意图和一些模型流程示意图
* **model**
    * 保存训练得到的GAN模型的生成器和判别器
    * 包含三个文件夹MixedGAN_GC、MixedGAN_GP和MixedGAN_WD，分别对应三种不同的GAN模型训练策略：Gradient Clipping、Gradient Penalty和Wasserstein Loss，详细说明见Notebook源码文件. 
* **output**
    * 存放GAN模型在不同训练轮次后生成的压强场示例图，用于观察GAN模型的学习程度
    * 三个文件夹MixedGAN_GC、MixedGAN_GP和MixedGAN_WD，分别对应三种不同的GAN模型训练策略下生成结果
* **源码文件**
    * Notebook文件：基于GAN和LSTM的压强场时空序列预测.ipynb
    * Python源码：基于GAN和LSTM的压强场时空序列预测.py

-----------------------------------------------------------------------------------

#### 参考文献

[1] [M. Raissi, P. ](https://www.sciencedirect.com/science/article/pii/S0021999118307125)[Perdikaris](https://www.sciencedirect.com/science/article/pii/S0021999118307125)[, G. E. ](https://www.sciencedirect.com/science/article/pii/S0021999118307125)[Karniadakis](https://www.sciencedirect.com/science/article/pii/S0021999118307125)[, Physics-informed neural networks: a deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378 (2019) 686–707.](https://www.sciencedirect.com/science/article/pii/S0021999118307125)

[2] [Ian J. Goodfellow, Jean ](https://arxiv.org/abs/1406.2661)[Pouget](https://arxiv.org/abs/1406.2661)[-Abadie, Mehdi Mirza et al., Generative Adversarial Networks, ](https://arxiv.org/abs/1406.2661)[arXiv](https://arxiv.org/abs/1406.2661)[ preprint, arXiv:1406.2661, 2014.](https://arxiv.org/abs/1406.2661)

[3] [Martin ](https://arxiv.org/abs/1701.07875)[Arjovsky](https://arxiv.org/abs/1701.07875)[, ](https://arxiv.org/abs/1701.07875)[Soumith](https://arxiv.org/abs/1701.07875)[ ](https://arxiv.org/abs/1701.07875)[Chintala](https://arxiv.org/abs/1701.07875)[, Léon ](https://arxiv.org/abs/1701.07875)[Bottou](https://arxiv.org/abs/1701.07875)[, Wasserstein GAN, ](https://arxiv.org/abs/1701.07875)[arXiv](https://arxiv.org/abs/1701.07875)[ preprint, arXiv:1701.07875, 2017.](https://arxiv.org/abs/1701.07875)

[4] [Alec Radford, Luke Metz, ](https://arxiv.org/abs/1511.06434)[Soumith](https://arxiv.org/abs/1511.06434)[ ](https://arxiv.org/abs/1511.06434)[Chintala](https://arxiv.org/abs/1511.06434)[, Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks, ](https://arxiv.org/abs/1511.06434)[arXiv](https://arxiv.org/abs/1511.06434)[ preprint, arXiv:1511.06434, 2015.](https://arxiv.org/abs/1511.06434)

[5] [Z. Xu, J. Du, J. Wang, C. Jiang and Y. Ren, "Satellite Image Prediction Relying on GAN and LSTM Neural Networks," ICC 2019 - 2019 IEEE International Conference on Communications (ICC), 2019, pp. 1-6, ](https://ieeexplore.ieee.org/abstract/document/8761462/similar)[doi](https://ieeexplore.ieee.org/abstract/document/8761462/similar)[: 10.1109/ICC.2019.8761462.](https://ieeexplore.ieee.org/abstract/document/8761462/similar)

-----------------------------------------------------------------------------------
