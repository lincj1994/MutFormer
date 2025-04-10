# Tabular Transformer Training Strategies Demonstration

This repository contains Python code demonstrating and comparing three different training strategies for a tabular Transformer model on synthetic clinical/genomic data:

1.  **Direct Learning:** Training the model directly on the target supervised task.
2.  **Gradual Learning:** Pre-training the model on the target task's *training features* using a self-supervised objective (like feature masking) before fine-tuning on the target labels.
3.  **Transfer Learning:** Pre-training the model on a separate, larger, unlabeled dataset using a self-supervised objective before fine-tuning on the target task.

The goal is to predict patient `Response` based on clinical features (Age, Sex, Tumor Stage) and gene mutation features.

## Key Concepts Demonstrated

*   **Synthetic Data Generation:** Creates two datasets:
    *   `df_target`: A smaller dataset with labels (`Response`) for the main prediction task.
    *   `df_pretrain`: A larger dataset *without* labels, sharing similar features but potentially different distributions, used for transfer learning pre-training.
*   **Feature Engineering:** Defines numerical and categorical features based on the generated data.
*   **Data Splitting:** Splits the target dataset into training, validation, and test sets.
*   **Custom Transformer Pipeline:** Utilizes a (assumed) `CustomTransformerPipeline` class that encapsulates data preprocessing, model definition (a tabular Transformer), training loops for different learning modes, prediction, and evaluation.
    *   **Note:** The actual implementation of `CustomTransformerPipeline`, `UnifiedTabularTransformer`, `AdvancedTabularDataset`, helper modules (`MultiheadAttentionWithResidual`, etc.), and `mask_features` function are **not included** in this script but are **required** for it to run. This script focuses on *demonstrating their usage*.
*   **Learning Mode Comparison:** Executes and evaluates the model trained via Direct, Gradual, and Transfer Learning modes.

## Prerequisites

*   Python 3.13
*   pandas
*   numpy
*   scikit-learn
*   PyTorch (`torch`)
*   **Required Custom Modules:** The necessary custom Python modules/classes (`CustomTransformerPipeline`, `UnifiedTabularTransformer`, `AdvancedTabularDataset`, etc.) must be available in your Python environment.

## Code Structure
The script will execute the data generation, splitting, training for each mode (Direct, Gradual, Transfer), and finally print the performance comparison on the test set. Checkpoints for models might be saved in directories like `direct_learning_checkpoint/` and `transfer_learning_pretrain_checkpoint/` if specified and enabled within the `CustomTransformerPipeline`.

```python
# --------------------------------------------------------------------------
# 数据生成
# --------------------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# --- 通用参数 ---
n_genes = 10       # 基因突变特征数量
random_seed = 42   # 固定随机种子保证可复现

# --- 1. 生成用于 Transfer Learning 预训练的数据 (df_pretrain) ---
print("--- 正在生成预训练数据集 (df_pretrain)... ---")
n_samples_pretrain = 20000
np.random.seed(random_seed + 1) # 使用不同的种子以产生不同的数据

# 特征与目标数据相似，但没有标签 y
patient_ids_pretrain = [f'PreTrainPatient_{i+1}' for i in range(n_samples_pretrain)]
age_pretrain = np.random.normal(loc=58, scale=12, size=n_samples_pretrain).astype(int).clip(25, 90) # 稍有不同的分布
sex_pretrain = np.random.choice(['Male', 'Female'], size=n_samples_pretrain, p=[0.52, 0.48]) # 稍有不同的比例
tumor_stage_pretrain = np.random.choice(['Stage I', 'Stage II', 'Stage III', 'Stage IV'],
                                        size=n_samples_pretrain, p=[0.25, 0.35, 0.25, 0.15]) # 稍有不同的分期比例
gene_features_pretrain = {}
gene_names_pretrain = [f'Gene_{chr(65+i)}' for i in range(n_genes)] # 保持基因名称一致很重要
for i, gene_name in enumerate(gene_names_pretrain):
    mutation_rate = np.random.uniform(0.04, 0.35) # 稍有不同的突变率范围
    gene_features_pretrain[f'{gene_name}_Mutation'] = np.random.choice([0, 1], size=n_samples_pretrain, p=[1 - mutation_rate, mutation_rate])

# 组合成 DataFrame (没有 Response 列)
df_pretrain = pd.DataFrame({
    'PatientID': patient_ids_pretrain,
    'Age': age_pretrain,
    'Sex': sex_pretrain,
    'Tumor_Stage': tumor_stage_pretrain,
    **gene_features_pretrain
})
print(f"预训练数据集 (df_pretrain) 生成完毕，形状: {df_pretrain.shape}")
print(df_pretrain.head())

# --- 2. 生成用于目标任务的数据 (df_target) ---
# 使用你提供的代码
print("\n--- 正在生成目标任务数据集 (df_target)... ---")
n_samples_target = 2000
np.random.seed(random_seed) # 确保与你原始代码一致

# 特征
patient_ids_target = [f'TargetPatient_{i+1}' for i in range(n_samples_target)]
age_target = np.random.normal(loc=60, scale=10, size=n_samples_target).astype(int).clip(30, 85)
sex_target = np.random.choice(['Male', 'Female'], size=n_samples_target, p=[0.5, 0.5])
tumor_stage_target = np.random.choice(['Stage I', 'Stage II', 'Stage III', 'Stage IV'],
                                      size=n_samples_target, p=[0.2, 0.3, 0.3, 0.2])
gene_features_target = {}
gene_names_target = [f'Gene_{chr(65+i)}' for i in range(n_genes)] # 与预训练数据基因名一致
for i, gene_name in enumerate(gene_names_target):
    mutation_rate = np.random.uniform(0.05, 0.3)
    gene_features_target[f'{gene_name}_Mutation'] = np.random.choice([0, 1], size=n_samples_target, p=[1 - mutation_rate, mutation_rate])

# 标签 (疗效)
score_target = (
    -0.05 * (age_target - 60)
    - 2 * (tumor_stage_target == 'Stage IV').astype(int)
    - 1 * (tumor_stage_target == 'Stage III').astype(int)
    + 2.5 * gene_features_target['Gene_A_Mutation']
    - 2.0 * gene_features_target['Gene_C_Mutation']
    + np.random.normal(0, 1.5, n_samples_target)
)
prob_responder_target = 1 / (1 + np.exp(-score_target / 3))
response_target = (np.random.rand(n_samples_target) < prob_responder_target).astype(int)

# 组合成 DataFrame
df_target = pd.DataFrame({
    'PatientID': patient_ids_target,
    'Age': age_target,
    'Sex': sex_target,
    'Tumor_Stage': tumor_stage_target,
    **gene_features_target,
    'Response': response_target # 包含标签列
})
print(f"目标任务数据集 (df_target) 生成完毕，形状: {df_target.shape}")
print(df_target.head())

# --- 3. 定义特征类型列表 (基于目标数据集 df_target 的列名) ---
# 这些列表将在初始化 Pipeline 时传入
target = 'Response'
numerical_features = ['Age'] + [col for col in df_target.columns if 'Gene_' in col]
categorical_features = ['Sex', 'Tumor_Stage']
# 检查所有特征是否都在 df_target 中
assert all(f in df_target.columns for f in numerical_features + categorical_features)
# 检查所有特征是否都在 df_pretrain 中 (除了 PatientID 和 Response)
assert all(f in df_pretrain.columns for f in numerical_features + categorical_features)

# --- 4. 划分目标任务数据集 (df_target) ---
print("\n--- 正在划分目标任务数据集 (df_target) 为训练/验证/测试集... ---")
X_target = df_target.drop(columns=[target, 'PatientID'])
y_target = df_target[target]

# 第一次划分: 训练+验证集 (80%) vs 测试集 (20%) - 调整比例以获得更多训练数据
X_train_val_raw, X_test_raw, y_train_val, y_test = train_test_split(
    X_target, y_target, test_size=0.2, random_state=random_seed, stratify=y_target
)
# 第二次划分: 训练集 (占原始 80% * 80% = 64%) vs 验证集 (占原始 80% * 20% = 16%)
X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    X_train_val_raw, y_train_val, test_size=0.2, random_state=random_seed, stratify=y_train_val
)

print(f"目标任务数据集划分完成:")
print(f"  训练集 (X_train_raw): {X_train_raw.shape}, (y_train): {y_train.shape}")
print(f"  验证集 (X_val_raw): {X_val_raw.shape}, (y_val): {y_val.shape}")
print(f"  测试集 (X_test_raw): {X_test_raw.shape}, (y_test): {y_test.shape}")

# --- 5. 准备预训练数据集 (df_pretrain) ---
# 对于 transfer learning，我们只需要特征部分 X_pretrain_raw
# PatientID 通常不作为特征输入模型
X_pretrain_raw = df_pretrain.drop(columns=['PatientID'])
print(f"\n预训练特征集 (X_pretrain_raw): {X_pretrain_raw.shape}")
```

```python
# --------------------------------------------------------------------------
# 参数设置
# --------------------------------------------------------------------------
# --- [确保之前的 Pipeline 代码已定义或导入] ---
# 包括:
# - 辅助模块 (MultiheadAttentionWithResidual, FeedForwardWithResidual, ...)
# - 特征掩码函数 (mask_features)
# - 数据集类 (AdvancedTabularDataset)
# - 统一模型类 (UnifiedTabularTransformer)
# - 封装类 (CustomTransformerPipeline)

# 假设以上类和函数已在此 Python 环境中定义
print("\n--- Transformer 相关类和函数已准备就绪 ---")

# --- [设置共享参数] ---
# 选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型架构超参数 (保持一致)
model_args = {
    "d_model": 64,
    "nhead": 4,        # 减少头数以加快演示速度
    "num_layers": 2,   # 减少层数以加快演示速度
    "dim_feedforward": 128,
    "dropout": 0.1,
    "use_cls_token": True
}

# 通用训练参数
batch_size = 64    # 适当调整
random_seed = 42



print("\n" + "="*40)
print("--- 演示: Direct Learning ---")
print("="*40)

# --- 初始化 Pipeline ---
pipeline_direct = CustomTransformerPipeline(
    # 传入特征列名
    numerical_features=numerical_features,
    categorical_features=categorical_features,
    # 指定学习模式
    learning_mode="direct",
    # 模型架构参数
    **model_args,
    # 监督学习参数
    supervised_epochs=2,   # 演示减少轮数
    supervised_lr=1e-4,
    supervised_weight_decay=1e-5,
    patience=5,             # 早停耐心值
    # 其他参数
    batch_size=batch_size,
    device=device,
    random_seed=random_seed,
    verbose=True,
    save_dir='direct_learning_checkpoint'
)

# --- 训练模型 ---
# 传入原始的训练集和验证集
pipeline_direct.fit(
    X_train_raw=X_train_raw,
    y_train=y_train.to_numpy(), # 确保传入 NumPy
    X_val_raw=X_val_raw,
    y_val=y_val.to_numpy(),   # 确保传入 NumPy
    # Direct learning 不需要 X_pretrain_raw
)

# --- 预测测试集 ---
print("\n--- 使用 Direct Learning 模型进行预测 ---")
# 传入原始测试集特征
direct_test_probs = pipeline_direct.predict_proba(X_test_raw)
direct_test_preds = pipeline_direct.predict(X_test_raw)

# --- 评估性能 (可选) ---
direct_test_auc = roc_auc_score(y_test, direct_test_probs[:, 1])
direct_test_acc = accuracy_score(y_test, direct_test_preds)
print(f"\nDirect Learning - 测试集性能:")
print(f"  AUC: {direct_test_auc:.4f}")
print(f"  Accuracy: {direct_test_acc:.4f}")

# --- 获取嵌入或注意力 (可选) ---
direct_embeddings = pipeline_direct.get_final_embeddings(X_test_raw[:10])
direct_attention = pipeline_direct.get_attention_weights(X_test_raw[:1])
```

```python
# --------------------------------------------------------------------------
# Direct learning
# --------------------------------------------------------------------------
print("\n" + "="*40)
print("--- 演示: Direct Learning ---")
print("="*40)

# --- 初始化 Pipeline ---
pipeline_direct = CustomTransformerPipeline(
    # 传入特征列名
    numerical_features=numerical_features,
    categorical_features=categorical_features,
    # 指定学习模式
    learning_mode="direct",
    # 模型架构参数
    **model_args,
    # 监督学习参数
    supervised_epochs=2,   # 演示减少轮数
    supervised_lr=1e-4,
    supervised_weight_decay=1e-5,
    patience=5,             # 早停耐心值
    # 其他参数
    batch_size=batch_size,
    device=device,
    random_seed=random_seed,
    verbose=True,
    save_dir='direct_learning_checkpoint'
)

# --- 训练模型 ---
# 传入原始的训练集和验证集
pipeline_direct.fit(
    X_train_raw=X_train_raw,
    y_train=y_train.to_numpy(), # 确保传入 NumPy
    X_val_raw=X_val_raw,
    y_val=y_val.to_numpy(),   # 确保传入 NumPy
    # Direct learning 不需要 X_pretrain_raw
)

# --- 预测测试集 ---
print("\n--- 使用 Direct Learning 模型进行预测 ---")
# 传入原始测试集特征
direct_test_probs = pipeline_direct.predict_proba(X_test_raw)
direct_test_preds = pipeline_direct.predict(X_test_raw)

# --- 评估性能 (可选) ---
direct_test_auc = roc_auc_score(y_test, direct_test_probs[:, 1])
direct_test_acc = accuracy_score(y_test, direct_test_preds)
print(f"\nDirect Learning - 测试集性能:")
print(f"  AUC: {direct_test_auc:.4f}")
print(f"  Accuracy: {direct_test_acc:.4f}")

# --- 获取嵌入或注意力 (可选) ---
direct_embeddings = pipeline_direct.get_final_embeddings(X_test_raw[:10])
direct_attention = pipeline_direct.get_attention_weights(X_test_raw[:1])
```

```python
# --------------------------------------------------------------------------
# Gradual learning
# --------------------------------------------------------------------------
print("\n" + "="*40)
print("--- 演示: Gradual Learning ---")
print("="*40)

# --- 初始化 Pipeline ---
pipeline_gradual = CustomTransformerPipeline(
    # 特征列名
    numerical_features=numerical_features,
    categorical_features=categorical_features,
    # 指定学习模式
    learning_mode="gradual",
    # 模型架构参数
    **model_args,
    # 预训练参数 (在目标训练集 X 上进行)
    pretrain_epochs=3,       # 演示减少轮数
    pretrain_lr=5e-4,
    mask_probability=0.2,    # 可以调整掩码概率
    pretrain_weight_decay=1e-5,
    # 微调参数
    supervised_epochs=2,   # 演示减少轮数
    supervised_lr=5e-5,    # 微调时使用较低的学习率
    supervised_weight_decay=1e-5,
    patience=5,
    # 其他参数
    batch_size=batch_size,
    device=device,
    random_seed=random_seed,
    verbose=True
)

# --- 训练模型 ---
# 传入原始的训练集和验证集
# `fit` 方法内部会先用 X_train_raw 进行预训练，然后用 (X_train_raw, y_train) 微调
pipeline_gradual.fit(
    X_train_raw=X_train_raw,
    y_train=y_train.to_numpy(),
    X_val_raw=X_val_raw,
    y_val=y_val.to_numpy(),
    # Gradual learning 不需要 X_pretrain_raw
)

# --- 预测测试集 ---
print("\n--- 使用 Gradual Learning 模型进行预测 ---")
gradual_test_probs = pipeline_gradual.predict_proba(X_test_raw)
gradual_test_preds = pipeline_gradual.predict(X_test_raw)

# --- 评估性能 (可选) ---
gradual_test_auc = roc_auc_score(y_test, gradual_test_probs[:, 1])
gradual_test_acc = accuracy_score(y_test, gradual_test_preds)
print(f"\nGradual Learning - 测试集性能:")
print(f"  AUC: {gradual_test_auc:.4f}")
print(f"  Accuracy: {gradual_test_acc:.4f}")
```

```python
# --------------------------------------------------------------------------
# Transfer learning
# --------------------------------------------------------------------------
print("\n" + "="*40)
print("--- 演示: Transfer Learning (Internal Pre-training) ---")
print("="*40)

# --- 初始化 Pipeline ---
pipeline_transfer = CustomTransformerPipeline(
    # 特征列名
    numerical_features=numerical_features,
    categorical_features=categorical_features,
    # 指定学习模式
    learning_mode="transfer_internal_pretrain",
    # 模型架构参数
    **model_args,
    # 预训练参数 (在额外的 X_pretrain_raw 上进行)
    pretrain_epochs=5,       # 演示轮数，大数据集可能需要更多
    pretrain_lr=5e-4,
    mask_probability=0.2,
    pretrain_weight_decay=1e-5,
    # 微调参数
    supervised_epochs=2,   # 演示轮数
    supervised_lr=5e-5,    # 微调低学习率
    supervised_weight_decay=1e-5,
    patience=5,
    # 其他参数
    batch_size=batch_size,
    device=device,
    random_seed=random_seed,
    verbose=True,
    save_dir='transfer_learning_pretrain_checkpoint'
)

# --- 训练模型 ---
# **关键:** 传入 X_pretrain_raw 用于预训练
pipeline_transfer.fit(
    X_train_raw=X_train_raw,         # 目标任务训练数据
    y_train=y_train.to_numpy(),
    X_val_raw=X_val_raw,             # 目标任务验证数据
    y_val=y_val.to_numpy(),
    X_pretrain_raw=X_pretrain_raw    # 用于预训练的额外数据
)

# --- 预测测试集 ---
print("\n--- 使用 Transfer Learning 模型进行预测 ---")
transfer_test_probs = pipeline_transfer.predict_proba(X_test_raw)
transfer_test_preds = pipeline_transfer.predict(X_test_raw)

# --- 评估性能 (可选) ---
transfer_test_auc = roc_auc_score(y_test, transfer_test_probs[:, 1])
transfer_test_acc = accuracy_score(y_test, transfer_test_preds)
print(f"\nTransfer Learning - 测试集性能:")
print(f"  AUC: {transfer_test_auc:.4f}")
print(f"  Accuracy: {transfer_test_acc:.4f}")

# --- 对比结果 ---
print("\n--- 性能对比 ---")
print(f"Direct Learning AUC: {direct_test_auc:.4f}, Accuracy: {direct_test_acc:.4f}")
print(f"Gradual Learning AUC: {gradual_test_auc:.4f}, Accuracy: {gradual_test_acc:.4f}")
print(f"Transfer Learning AUC: {transfer_test_auc:.4f}, Accuracy: {transfer_test_acc:.4f}")
```
