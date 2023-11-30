When $a \ne 0$, there are two solutions to $(ax^2 + bx + c = 0)$ and they are 
$$ x = {-b \pm \sqrt{b^2-4ac} \over 2a} $$

The Cauchy-Schwarz Inequality

$$\left( \sum_{k=1}^n a_k b_k \right)^2 \leq \left( \sum_{k=1}^n a_k^2 \right) \left( \sum_{k=1}^n b_k^2 \right)$$
# 三个指标

## 1.MMD

$\text{MMD}(S_g, S_r) = \frac{1}{|S_r|} \sum_{Y \in S_r} \min_{X \in S_g} D(X, Y)$

Sr为参考点云集，Sg为生成点云集，其中函数D()

​		其中D()可以是一下两种之一:

$\text{CD}(X, Y) = \sum_{x \in X} \min_{y \in Y} \|x - y\|_2^2 + \sum_{y \in Y} \min_{x \in X} \|x - y\|_2^2$

$\text{EMD}(X, Y) = \min_{\phi:X \to Y} \sum_{x \in X} \|x - \phi(x)\|_2^2$

其中 X 和 Y 是两个具有相同点数的点云，φ 是它们之间的映射。请注意，大多数以前的方法在其训练目标中使用 CD 或 EMD，如果在相同的度量下进行评估，这往往会受到青睐。然而，我们的方法在训练期间不使用 CD 或 EMD

对于参考集中的每个点云，计算并平均生成集中与其最近邻居的距离,==越小越好==

## 2.COV

$\text{COV}(S_g, S_r) = \frac{\left| \{ \arg \min_{Y \in S_r} D(X, Y) \mid X \in S_g \} \right|}{|S_r|}$

​		覆盖度(COV)测量参考集中与生成集中至少一个点云匹配的点云的比例。对于生成集中的每个点云，其在参考集中最近的邻居被标记为匹配，==越大越好==

## 3.1-NNA

$1 - \text{NNA}(S_g, S_r) = \frac{\sum_{X \in S_g} II[N_X \in S_g] + \sum_{Y \in S_r} II[N_Y \in S_r]}{|S_g| + |S_r|}$

其中I[·]为指示函数。对于每个样本，1-NN分类器根据其最近样本的标签将其分类为来自Sr或Sg。如果Sg和Sr是从相同的分布中采样，那么给定足够数量的样本，这种分类器的准确率应该收敛到50%。==准确度越接近50%，Sg和Sr越相似==，因此模型在学习目标分布方面就越好。在我们的设置中，可以使用CD或EMD来计算最近的邻居。与JSD不同，1-NNA考虑形状分布之间的相似性，而不是边缘点分布之间的相似性。与COV和MMD不同，1-NNA直接衡量分布相似性，并考虑多样性和质量。



## 4.结果日志

日志参数示例：

```python
Mon Nov 20 12:40:57 2023 #运行日期
name:train_chair_stage2, # 训练模型昵称
lr:0.002, # 学习率
iter:700, # 迭代次数
epoch:29, # 迭代周期
batch_idx:4, # 批次索引
batch_size:128, # 批次大小
total_loss:0.815229058265686, # 总损失
eta:1 day, 21:11:58,
prior_loss:0.0, # 预测损失
kl_weight:0.0, # KL散度权重
log_p_part_0:-60355.62890625, # 第i部分的概率分布与参数
entropy_0:351.659912109375, # 第i部分的熵值
part_0_mean:0.004990874789655209, # 部分i的均值
part_0_logvar:-0.09053404629230499, # 部分i的对数方差
log_p_part_1:-60353.8984375,
entropy_1:351.9974060058594,
part_1_mean:-0.012598402798175812,
part_1_logvar:-0.08789718896150589,
log_p_part_2:-60353.12109375,
entropy_2:350.622314453125,
part_2_mean:0.012611385434865952,
part_2_logvar:-0.09864026308059692,
log_p_part_3:-60353.4296875,
entropy_3:354.4195251464844,
part_3_mean:-0.12426772713661194,
part_3_logvar:0.40280741453170776,
fit_loss:0.09926700592041016, # 拟合损失
mse_loss:0.7159620523452759 # 均方误差损失
```

## 5.评估结果

结果计算函数方法保存在python/difffacto/datasets/evaluation_utils.py中，替换npy可判断性能指标

# 模型

## 损失函数

$D_{\text{KL}}(\mathcal{N}(\mu_1, \sigma_1^2) \, || \, \mathcal{N}(\mu_2, \sigma_2^2)) = 0.5 \left( -1 + \log(\sigma_2^2) - \log(\sigma_1^2) + \frac{\sigma_1^2}{\sigma_2^2} + (\mu_2 - \mu_1)^2 \frac{1}{\sigma_2^2} \right)$

​	losses采用K-L散度损失

​	KL散度用于度量两个概率分布之间的差异。对于两个高斯分布𝒩(𝜇₁, 𝜎₁²)和𝒩(𝜇₂, 𝜎₂²)，tr表示迹运算，det表示行列式，*k*是分布的维度。在这个实现中，k被省略

## 模型修改（日期）

### 11-28

models/decomposers/transformer.py  修改mlp的输出 x = (x + x.mean(dim=1, keepdim=True)) * 0.5

​	参考[1]N. Hyeon-Woo, K. Yu-Ji, B. Heo, D. Han, S. Oh, and T.-H. Oh, “Scratching Visual Transformer’s Back with Uniform Attention,” Oct. 2022.

### 11-30

在模型优化器中注册Adan优化器，在config/train_chair_stage1.py切换优化器

==将11-28修改改回来进行的训练==

参考[1]X. Xie, P. Zhou, H. Li, Z. Lin, and S. Yan, “Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models,” Aug. 2022.

# 训练

# 评估

## 进程

selfdriving:

`tmux attach -t diffFacto_eval`评估

`tmux attach -t diffFacto_eval2`评估

`tmux attach -t diffFacto_eval3`评估

`tmux attach -t diffFacto_train1`训练——Adan优化器

`tmux attach -t diffFacto`训练——优化vit的mlp输出

## chair

### gen_fixed0000_chair_1250

#### 1.input_ref.npy & pred,npy

[lgan_mmd-CD] 0.07027258                                                                                                                 [lgan_cov-CD] 0.20312500                                                                                                                 [lgan_mmd_smp-CD] 0.06703432                                                                                                             [lgan_mmd-EMD] 0.34985146
[lgan_cov-EMD] 0.32812500
[lgan_mmd_smp-EMD] 0.32319206

{'lgan_mmd-CD': tensor(0.0703, device='cuda:0'), 'lgan_cov-CD': tensor(0.2031, device='cuda:0
'), 'lgan_mmd_smp-CD': tensor(0.0670, device='cuda:0'), 'lgan_mmd-EMD': tensor(0.3499, device
='cuda:0'), 'lgan_cov-EMD': tensor(0.3281, device='cuda:0'), 'lgan_mmd_smp-EMD': tensor(0.323
2, device='cuda:0'), '1-NN-CD-acc_t': tensor(0.8828, device='cuda:0'), '1-NN-CD-acc_f': tenso
r(1., device='cuda:0'), '1-NN-CD-acc': tensor(0.9414, device='cuda:0'), '1-NN-EMD-acc_t': ten
sor(0.9844, device='cuda:0'), '1-NN-EMD-acc_f': tensor(1., device='cuda:0'), '1-NN-EMD-acc':
tensor(0.9922, device='cuda:0')}

#### 2.input_ref.npy & 100_sample 0.npy

[lgan_mmd-CD] 0.18313739
[lgan_cov-CD] 0.23437500
[lgan_mmd_smp-CD] 0.15402350
[lgan_mmd-EMD] 0.54550886
[lgan_cov-EMD] 0.20312500
[lgan_mmd_smp-EMD] 0.53444016

{'lgan_mmd-CD': tensor(0.1831, device='cuda:0'), 'lgan_cov-CD': tensor(0.2344, device='cuda:0
'), 'lgan_mmd_smp-CD': tensor(0.1540, device='cuda:0'), 'lgan_mmd-EMD': tensor(0.5455, device
='cuda:0'), 'lgan_cov-EMD': tensor(0.2031, device='cuda:0'), 'lgan_mmd_smp-EMD': tensor(0.534
4, device='cuda:0'), '1-NN-CD-acc_t': tensor(0.9609, device='cuda:0'), '1-NN-CD-acc_f': tenso
r(1., device='cuda:0'), '1-NN-CD-acc': tensor(0.9805, device='cuda:0'), '1-NN-EMD-acc_t': ten
sor(1., device='cuda:0'), '1-NN-EMD-acc_f': tensor(1., device='cuda:0'), '1-NN-EMD-acc': tens
or(1., device='cuda:0')}

#### 3.input_ref.npy & sample prior 0.npy

[lgan_mmd-CD] 0.18178780                                                                     
[lgan_cov-CD] 0.21093750                                                                     
[lgan_mmd_smp-CD] 0.15364261                                                                 
[lgan_mmd-EMD] 0.54294533                                                                    
[lgan_cov-EMD] 0.19531250                                                                    
[lgan_mmd_smp-EMD] 0.53541660 

{'lgan_mmd-CD': tensor(0.1818, device='cuda:0'), 'lgan_cov-CD': tensor(0.2109, device='cuda:0
'), 'lgan_mmd_smp-CD': tensor(0.1536, device='cuda:0'), 'lgan_mmd-EMD': tensor(0.5429, device
='cuda:0'), 'lgan_cov-EMD': tensor(0.1953, device='cuda:0'), 'lgan_mmd_smp-EMD': tensor(0.535
4, device='cuda:0'), '1-NN-CD-acc_t': tensor(0.9609, device='cuda:0'), '1-NN-CD-acc_f': tenso
r(1., device='cuda:0'), '1-NN-CD-acc': tensor(0.9805, device='cuda:0'), '1-NN-EMD-acc_t': ten
sor(1., device='cuda:0'), '1-NN-EMD-acc_f': tensor(1., device='cuda:0'), '1-NN-EMD-acc': tens
or(1., device='cuda:0')}

## airplane

### gen_fixed0000_airplane_1000

#### 1.input_ref.npy & pred.npy(eval)

[lgan_mmd-CD] 0.03956761
[lgan_cov-CD] 0.44531250
[lgan_mmd_smp-CD] 0.02753396
[lgan_mmd-EMD] 0.27391928
[lgan_cov-EMD] 0.42187500
[lgan_mmd_smp-EMD] 0.24810757

#### 2..input_ref.npy & sample prior 0.npy(eval2)

[lgan_mmd-CD] 0.07048073
[lgan_cov-CD] 0.21875000
[lgan_mmd_smp-CD] 0.06686447
[lgan_mmd-EMD] 0.34944683
[lgan_cov-EMD] 0.32031250
[lgan_mmd_smp-EMD] 0.32432899

#### 3.input_ref.npy & 100_sample 0.npy(eval3)

[lgan_mmd-CD] 0.07027258
[lgan_cov-CD] 0.20312500
[lgan_mmd_smp-CD] 0.06703432
[lgan_mmd-EMD] 0.34982526
[lgan_cov-EMD] 0.32031250
[lgan_mmd_smp-EMD] 0.32313645



# 常用指令

## tmux后台

tmux new -s --name

tmux attach -t --name

## 环境+项目

conda activate diffFacto

cd /tmp/pycharm_project_696/

## 测试指标

python python/difffacto/datasets/evaluation_utils.py

==记得在evaluation_utils.py中，将两个计算的点云设置好==

## 训练

```python
CUDA_VISIBLE_DEVICES=[idx] python tools/run_net.py --config-file configs/train_chair_stage1.py  --task train --prefix chair_stage1
```

# 注意事项

在训练中默认会将模型训练从上次未完成的epoch开始，若想重头开始训练，需要将python/difffacto/config/config.py中约111行代码的self.name替换掉。

第109行self.name=修改为想生成的文件夹昵称
