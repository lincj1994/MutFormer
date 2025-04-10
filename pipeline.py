import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

import copy # 用于保存最佳模型
import os
import tempfile
import warnings
from pathlib import Path
from typing import Literal, Dict, Optional, Union, List, Any, Tuple, Self


# --------------------------------------------------------------------------
# 特征掩码函数
# --------------------------------------------------------------------------
def mask_features(x_num_batch, x_cat_batch, num_numerical, cat_feature_names,
                  mask_probability=0.15, mask_token_value=-1e9):
    """
    对批次中的数值和分类特征进行随机掩码。

    Args:
        x_num_batch (Tensor): 数值特征批次, Shape (B, N_num).
        x_cat_batch (Tensor): 分类特征批次, Shape (B, N_cat).
        num_numerical (int): 数值特征的数量。
        cat_feature_names (list): 分类特征的名称列表。
        mask_probability (float): 每个特征被掩盖的概率。
        mask_token_value (float/int): 用于替换被掩盖特征的值 (用于数值和分类)。
                                      注意：实际模型是用 MASK embedding 替换，
                                      这里返回的 x_masked_* 只是标记了位置。

    Returns:
        tuple: 包含:
            - x_num_masked (Tensor): 被掩盖标记的数值特征。
            - x_cat_masked (Tensor): 被掩盖标记的分类特征。
            - masked_indices_num (Tensor[bool]): 数值特征的掩码指示。
            - masked_indices_cat (Tensor[bool]): 分类特征的掩码指示。
            - original_num (Tensor): 原始数值特征 (用于计算损失)。
            - original_cat (Tensor): 原始分类特征 (用于计算损失)。
    """
    device = x_num_batch.device
    batch_size_num, n_num = x_num_batch.shape
    batch_size_cat, n_cat = x_cat_batch.shape
    assert batch_size_num == batch_size_cat

    # 创建掩码指示张量
    masked_indices_num = torch.zeros_like(x_num_batch, dtype=torch.bool, device=device)
    masked_indices_cat = torch.zeros_like(x_cat_batch, dtype=torch.bool, device=device)

    # 复制原始值用于计算损失
    original_num = x_num_batch.clone()
    original_cat = x_cat_batch.clone()

    # 复制用于创建掩码版本
    x_num_masked = x_num_batch.clone()
    x_cat_masked = x_cat_batch.clone()

    # 对每个样本独立进行掩码
    for i in range(batch_size_num):
        # --- 掩盖数值特征 ---
        # 决定掩盖哪些数值特征
        num_mask = torch.rand(n_num, device=device) < mask_probability
        masked_indices_num[i, num_mask] = True
        # 应用掩码 (实际替换发生在模型内部用 MASK embedding)
        # 这里只是为了返回一个标记过的输入，虽然模型不直接用这个 value
        x_num_masked[i, num_mask] = mask_token_value

        # --- 掩盖分类特征 ---
        cat_mask = torch.rand(n_cat, device=device) < mask_probability
        masked_indices_cat[i, cat_mask] = True
        # 应用掩码
        x_cat_masked[i, cat_mask] = mask_token_value # 用特殊值标记

    return x_num_masked, x_cat_masked, masked_indices_num, masked_indices_cat, original_num, original_cat


# --------------------------------------------------------------------------
# PyTorch 数据集类
# 需要将NumPy数据包装成PyTorch认识的Dataset格式
# --------------------------------------------------------------------------

class AdvancedTabularDataset(Dataset):
    """
    自定义的PyTorch数据集类，用于处理分离的数值和分类特征。
    PyTorch的Dataset类需要实现 __init__, __len__, __getitem__ 三个方法。
    """
    def __init__(self, features_num, features_cat, labels):
        """
        初始化函数，加载数据。
        Args:
            features_num (np.array): 数值特征数组。
            features_cat (np.array): 分类特征数组 (整数索引)。
            labels (np.array): 目标标签数组。
        """
        # 将 NumPy 数组转换为 PyTorch张量 (Tensor)
        self.features_num = torch.tensor(features_num, dtype=torch.float32)
        self.features_cat = torch.tensor(features_cat, dtype=torch.int64) # 分类索引是整数
        self.labels = torch.tensor(labels, dtype=torch.int64)             # 标签也是整数

    def __len__(self):
        """返回数据集中的样本总数。"""
        return len(self.labels)

    def __getitem__(self, idx):
        """
        根据给定的索引 idx，返回对应的单个样本数据。
        DataLoader 会调用这个方法来获取每个批次的数据。
        Args:
            idx (int): 样本索引。
        Returns:
            tuple: 包含单个样本的数值特征、分类特征和标签的元组。
        """
        return self.features_num[idx], self.features_cat[idx], self.labels[idx]

print("\n--- 定义 PyTorch 数据集类完成 ---")


# --------------------------------------------------------------------------
# 封装的 Pipeline 类 (集成预处理和内部 Transfer Learning)
# --------------------------------------------------------------------------
class CustomTransformerPipeline:
    """
    封装了数据预处理、训练（Direct, Gradual, Transfer）和预测的流水线。
    """
    def __init__(
        self,
        # --- 特征定义 ---
        numerical_features: List[str],          # 数值特征列名列表
        categorical_features: List[str],        # 分类特征列名列表
        # --- 模型架构参数 ---
        d_model: int = 64, nhead: int = 8, num_layers: int = 3,
        dim_feedforward: int = 128, dropout: float = 0.1, use_cls_token: bool = True,
        # --- 学习模式 ---
        learning_mode: Literal["direct", "gradual", "transfer_internal_pretrain"] = "direct",
        # --- 预训练参数 ---
        pretrain_epochs: int = 100, pretrain_lr: float = 5e-4,
        mask_probability: float = 0.15, pretrain_weight_decay: float = 1e-5,
        # --- 监督学习/微调参数 ---
        supervised_epochs: int = 50, supervised_lr: float = 1e-5,
        supervised_weight_decay: float = 1e-5, patience: int = 15,
        # --- 其他 ---
        batch_size: int = 64, device: Union[str, torch.device] = "auto",
        random_seed: int = 42, verbose: bool = True,
        save_epoch_freq: int = 1,
        save_dir: str = "gradual_learning_pretrain_checkpoints"
        ):
        """
        初始化 Pipeline。

        Args:
            numerical_features: 需要进行数值处理的原始列名列表。
            categorical_features: 需要进行分类处理的原始列名列表。
            d_model, nhead, ... : 模型架构超参数。
            learning_mode: 'direct', 'gradual', 或 'transfer_internal_pretrain'。
            pretrain_*: 预训练阶段的超参数。
            supervised_*: 监督学习/微调阶段的超参数。
            patience: 早停法的耐心轮数。
            batch_size: 批次大小。
            device: 计算设备。
            random_seed: 随机种子。
            verbose: 是否打印过程信息。
            save_epoch_freq: 每隔多少个监督/微调 epoch 保存一次模型权重。设为 0 或 None 则不保存中间权重。
            save_dir: 用于保存 epoch 权重和最佳模型的目录名称。
        """
        # --- 保存配置 ---
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.model_params = {
            "d_model": d_model, "nhead": nhead, "num_layers": num_layers,
            "dim_feedforward": dim_feedforward, "dropout": dropout,
            "use_cls_token": use_cls_token
        }
        if learning_mode not in ["direct", "gradual", "transfer_internal_pretrain"]:
            raise ValueError("learning_mode 必须是 'direct', 'gradual', 或 'transfer_internal_pretrain'")
        self.learning_mode = learning_mode
        self.pretrain_params = {
            "epochs": pretrain_epochs, "lr": pretrain_lr,
            "mask_probability": mask_probability, "weight_decay": pretrain_weight_decay
        }
        self.supervised_params = {
            "epochs": supervised_epochs, "lr": supervised_lr,
            "weight_decay": supervised_weight_decay, "patience": patience
        }
        self.batch_size = batch_size
        self.verbose = verbose
        self.random_seed = random_seed
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        # 自动选择设备
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # --- 初始化内部状态 ---
        self.model_: Optional[UnifiedTabularTransformer] = None # 存储最终模型
        self.preprocessor_: Optional[ColumnTransformer] = None # 存储拟合的预处理器
        self.preprocessor_info_: Optional[Dict[str, Any]] = None # 存储预处理元信息
        # 临时文件用于 Gradual/Transfer 模式下的预训练权重传递
        self._temp_pretrain_path = os.path.join(tempfile.gettempdir(), f"temp_pretrained_{random_seed}.pth")

        self.save_epoch_freq = save_epoch_freq
        self.save_dir = Path(save_dir)

        if self.verbose:
            print("--- CustomTransformerPipeline 初始化 ---")
            print(f"  数值特征: {self.numerical_features}")
            print(f"  分类特征: {self.categorical_features}")
            print(f"  学习模式: {self.learning_mode}")
            print(f"  设备: {self.device}")
            print(f"  模型参数: {self.model_params}")
            if self.learning_mode != "direct":
                print(f"  预训练参数: {self.pretrain_params}")
            print(f"  监督/微调参数: {self.supervised_params}")
            print("-" * 40)

    def _build_preprocessor(self) -> ColumnTransformer:
        """构建 scikit-learn 预处理管道"""
        if self.verbose: print("  构建预处理器管道...")

        # 数值特征处理：填充缺失值 -> 稳健缩放
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', RobustScaler()) # 对离群值更鲁棒
        ])

        # 分类特征处理：顺序编码 -> 填充（处理未知值导致的NaN）
        categorical_transformer = Pipeline(steps=[
            ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-99)), # 未知映射为-99
            ('imputer', SimpleImputer(strategy='most_frequent')) # 填充-99或其他意外NaN
        ])

        # 组合预处理器：
        # 1. 对数值列：先生成NaN指示器，然后与填充+缩放后的原列合并
        # 2. 对分类列：应用分类处理管道
        preprocessor = ColumnTransformer(
            transformers=[
                ('numerical_processing', Pipeline([
                    ('feature_union', FeatureUnion([
                        # ('A') 缺失指示器: 只对实际有缺失的列生成 0/1 特征
                        ('missing_indicator', MissingIndicator(error_on_new=False, features='missing-only')),
                        # ('B') 填充+缩放: 应用于所有指定的数值列
                        ('imputer_scaler', numerical_transformer)
                    ]))
                 ]), self.numerical_features), # 应用于原始数值列列表

                ('categorical_processing', categorical_transformer, self.categorical_features) # 应用于原始分类列列表
            ],
            remainder='drop' # 丢弃未在 numerical_features 或 categorical_features 中指定的列
        )
        return preprocessor

    def _preprocess_data(self,
                         X_raw: Union[pd.DataFrame, np.ndarray],
                         fit_preprocessor: bool = False
                         ) -> Tuple[np.ndarray, np.ndarray, Optional[Dict[str, Any]]]:
        """
        应用（或拟合并应用）预处理步骤。

        Args:
            X_raw: 原始输入特征数据 (DataFrame 或 NumPy)。
            fit_preprocessor: 是否需要拟合预处理器（只对训练数据做一次）。

        Returns:
            Tuple: (处理后的数值特征, 处理后的分类特征, 元信息字典 or None)。
                   元信息字典仅在 fit_preprocessor=True 时返回。
        """
        # 确保输入是 DataFrame
        if not isinstance(X_raw, pd.DataFrame):
            if isinstance(X_raw, np.ndarray):
                # 尝试根据初始化时提供的列名创建 DataFrame
                all_feature_names = self.numerical_features + self.categorical_features
                if X_raw.shape[1] == len(all_feature_names):
                    X_df = pd.DataFrame(X_raw, columns=all_feature_names)
                else:
                    raise ValueError(f"NumPy 输入的列数 ({X_raw.shape[1]}) 与指定的特征列表长度 ({len(all_feature_names)}) 不匹配。")
            else:
                raise TypeError("输入数据 X 必须是 pandas DataFrame 或 NumPy array。")
        else:
            # 确保 DataFrame 包含所有需要的列
            required_cols = set(self.numerical_features + self.categorical_features)
            missing_cols = required_cols - set(X_raw.columns)
            if missing_cols:
                raise ValueError(f"输入 DataFrame 缺少以下列: {missing_cols}")
            # 按指定顺序选择列，防止顺序问题
            X_df = X_raw[self.numerical_features + self.categorical_features]


        # --- 1. 构建或获取预处理器 ---
        if fit_preprocessor or self.preprocessor_ is None:
            # 如果需要拟合，或者预处理器还不存在
            if self.verbose and fit_preprocessor: print("  拟合预处理器...")
            self.preprocessor_ = self._build_preprocessor()
            # 在 DataFrame 上拟合，ColumnTransformer 会自动按列名选择
            self.preprocessor_.fit(X_df)
            if self.verbose and fit_preprocessor: print("  预处理器拟合完成。")
        elif not fit_preprocessor and self.preprocessor_ is None:
            # 如果不需要拟合，但预处理器不存在（说明没调用过 fit），则报错
            raise RuntimeError("预处理器尚未拟合，请先使用训练数据调用 fit。")

        # --- 2. 转换数据 ---
        if self.verbose: print(f"  转换数据 shape: {X_df.shape}...")
        X_processed: np.ndarray = self.preprocessor_.transform(X_df)
        if self.verbose: print(f"  转换后数据 shape: {X_processed.shape}")


        # --- 3. 提取元信息 (仅在 fit 时进行一次) ---
        metadata: Optional[Dict[str, Any]] = None
        if fit_preprocessor:
            if self.verbose: print("  提取预处理后的元信息...")
            try:
                # 尝试获取处理后的特征名称
                processed_feature_names: List[str] = self.preprocessor_.get_feature_names_out()

                # --- 确定各部分特征数量和索引 ---
                # 数值部分包含指示器和原数值列的处理结果
                num_pipeline = self.preprocessor_.named_transformers_['numerical_processing']
                feature_union = num_pipeline.named_steps['feature_union']

                # 手动遍历 transformer_list 来获取子输出
                indicator_names = []
                imputer_scaler_names = []
                for name, transformer in feature_union.transformer_list:
                    # 如果是 MissingIndicator
                    if name == 'missing_indicator':
                        indicator_names = transformer.get_feature_names_out(self.numerical_features)
                # 如果是 imputer+scaler pipeline
                    elif name == 'imputer_scaler':
                    # pipeline 最后一步 scaler 支持 get_feature_names_out
                        last = transformer.named_steps.get('scaler', transformer)
                        imputer_scaler_names = last.get_feature_names_out(self.numerical_features)

                num_indicator = len(indicator_names)
                num_scaled = len(imputer_scaler_names)

                # 分类处理管道
                cat_pipeline = self.preprocessor_.named_transformers_['categorical_processing']
                # OrdinalEncoder 本身不支持 get_feature_names_out，手动用原名
                cat_encoder_names = self.categorical_features.copy()
                num_cat = len(cat_encoder_names)

                # 验证
                total = num_indicator + num_scaled + num_cat
                if total != X_processed.shape[1]:
                    warnings.warn(
                        f"计算特征数 ({total}) != 实际输出 ({X_processed.shape[1]})",
                        RuntimeWarning, stacklevel=2
                    )

                # 构造索引
                model_numerical_indices = list(range(num_indicator + num_scaled))
                model_categorical_indices = list(range(num_indicator + num_scaled,
                                                    num_indicator + num_scaled + num_cat))

                # 取类别大小
                ordinal = cat_pipeline.named_steps['ordinal']
                category_sizes = {
                    feat: len(cats)
                    for feat, cats in zip(self.categorical_features, ordinal.categories_)
                }

                # # 获取 MissingIndicator 产生的特征名
                # indicator_names = feature_union.named_transformers_['missing_indicator'].get_feature_names_out()
                # num_indicator_features = len(indicator_names)
                # # 获取 Imputer+Scaler 产生的特征名 (通常是原始数值列名)
                # imputer_scaler_names = feature_union.named_transformers_['imputer_scaler'].get_feature_names_out()
                # num_numerical_scaled = len(imputer_scaler_names) # 应等于 len(self.numerical_features)

                # # 分类部分由 OrdinalEncoder 处理，列数不变
                # cat_pipeline = self.preprocessor_.named_transformers_['categorical_processing']
                # # 获取 OrdinalEncoder 处理后的特征名 (通常是原始分类列名)
                # cat_encoder_names = cat_pipeline.get_feature_names_out()
                # num_categorical_encoded = len(cat_encoder_names) # 应等于 len(self.categorical_features)

                # # 验证总数是否匹配
                # total_processed_features = num_indicator_features + num_numerical_scaled + num_categorical_encoded
                # if total_processed_features != X_processed.shape[1]:
                #      warnings.warn(f"计算得到的处理后特征数 ({total_processed_features}) 与实际输出 ({X_processed.shape[1]}) 不符，索引可能错误！", RuntimeWarning, stacklevel=2)

                # # --- 计算模型输入的数值和分类特征索引 ---
                # # 数值输入 = 指示器 + 填充缩放后的原数值列
                # model_numerical_indices = list(range(num_indicator_features + num_numerical_scaled))
                # # 分类输入 = 顺序编码后的原分类列
                # model_categorical_indices = list(range(num_indicator_features + num_numerical_scaled, total_processed_features))

                # # --- 获取类别大小 ---
                # category_sizes: Dict[str, int] = {}
                # ordinal_encoder = cat_pipeline.named_steps['ordinal']
                # for i, feature_name in enumerate(self.categorical_features):
                #     # ordinal_encoder.categories_ 是一个列表，第 i 个元素是第 i 个分类特征的所有类别
                #     num_known_categories = len(ordinal_encoder.categories_[i])
                #     category_sizes[feature_name] = num_known_categories

                # --- 存储元信息 ---
                metadata = {
                    'num_numerical': len(model_numerical_indices), # 模型接收的数值输入总数
                    'category_sizes': category_sizes,               # 分类特征类别数
                    'model_numerical_indices': model_numerical_indices, # 数值输入索引
                    'model_categorical_indices': model_categorical_indices, # 分类输入索引
                    'processed_feature_names': processed_feature_names  # 处理后所有特征名
                }
                self.preprocessor_info_ = metadata # 存储在实例中
                if self.verbose:
                    print("  元信息提取完成:")
                    print(f"    模型数值输入数: {metadata['num_numerical']}")
                    print(f"    分类类别数: {metadata['category_sizes']}")
                    print(f"    处理后特征名 (部分): {metadata['processed_feature_names'][:5]}...")

            except Exception as e:
                warnings.warn(f"自动提取预处理元信息失败: {e}。请确保 scikit-learn 版本支持 get_feature_names_out 或手动提供信息。", RuntimeWarning, stacklevel=2)
                self.preprocessor_info_ = None
                raise RuntimeError("无法自动确定处理后的特征信息！fit 失败。") from e

        # --- 4. 分离特征 (使用存储或刚计算的索引) ---
        if self.preprocessor_info_ is None:
             raise RuntimeError("无法分离特征，缺少预处理器信息。")

        num_indices = self.preprocessor_info_['model_numerical_indices']
        cat_indices = self.preprocessor_info_['model_categorical_indices']

        # 从 X_processed 中按索引提取
        X_num = X_processed[:, num_indices].astype(np.float32)
        X_cat = X_processed[:, cat_indices].astype(np.int64)

        # 处理 OrdinalEncoder 可能产生的未知值 (-99 -> 0)
        # np.maximum 比 inplace 操作更安全
        X_cat = np.maximum(X_cat, 0)

        return X_num, X_cat, metadata

    def _create_model(self, num_classes=2) -> UnifiedTabularTransformer:
        """使用存储的 preprocessor_info_ 创建模型实例"""
        if self.preprocessor_info_ is None:
            raise RuntimeError("模型需要先通过 fit 获取数据信息才能创建。")
        if self.verbose: print("  创建模型实例...")
        model = UnifiedTabularTransformer(
            num_numerical_features=self.preprocessor_info_['num_numerical'],
            category_sizes_dict=self.preprocessor_info_['category_sizes'],
            num_classes=num_classes,
            **self.model_params # 传入架构参数
        )
        # 将模型 verbose 属性与 pipeline 同步
        model.verbose = self.verbose # type: ignore[attr-defined] # 动态添加属性
        model.to(self.device)
        if self.verbose: print("  模型实例创建完成。")
        return model

    def fit(self,
            X_train_raw: Union[pd.DataFrame, np.ndarray],
            y_train: np.ndarray,
            X_val_raw: Optional[Union[pd.DataFrame, np.ndarray]] = None,
            y_val: Optional[np.ndarray] = None,
            X_pretrain_raw: Optional[Union[pd.DataFrame, np.ndarray]] = None,
            n_classes: Optional[int] = None
            ) -> Self:
        """
        训练模型，包含内部预处理。

        Args:
            X_train_raw: 原始训练集特征 (DataFrame 或 NumPy)。
            y_train: 训练集标签 (NumPy)。
            X_val_raw, y_val (optional): 原始验证集数据。如果未提供，将从训练集自动划分。
            X_pretrain_raw (optional): 用于 'transfer_internal_pretrain' 模式的额外原始预训练特征数据。
            n_classes (int, optional): 目标类别数量。如果 y_train 包含所有类别，可以自动推断。
        """
        # --- 参数检查 ---
        if self.save_epoch_freq is not None and self.save_epoch_freq > 0:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            if self.verbose: print(f"  模型权重将保存在: {self.save_dir}")

        if self.learning_mode == "transfer_internal_pretrain" and X_pretrain_raw is None:
            raise ValueError("当 learning_mode='transfer_internal_pretrain' 时，必须提供 X_pretrain_raw。")
        if (X_val_raw is None) != (y_val is None):
            raise ValueError("X_val_raw 和 y_val 必须同时提供或同时不提供。")

        # --- 准备标签和类别数 ---
        y_train = np.asarray(y_train)
        if n_classes is None:
            unique_classes = np.unique(y_train)
            n_classes = len(unique_classes)
            if self.verbose: print(f"自动推断目标类别数量: {n_classes}")
            # 检查类别是否从 0 开始连续，如果不是，可能需要 LabelEncoder
            if not np.all(unique_classes == np.arange(n_classes)):
                 warnings.warn("类别标签不是从 0 开始的连续整数，建议先使用 LabelEncoder 处理。", UserWarning, stacklevel=2)
        self.n_classes_ = n_classes

        # --- 1. 拟合预处理器并处理训练/验证数据 ---
        if self.verbose: print("\n--- 阶段: 数据预处理 ---")
        # 拟合预处理器，处理训练数据，获取元信息
        X_num_train, X_cat_train, _ = self._preprocess_data(X_train_raw, fit_preprocessor=True)
        if self.preprocessor_info_ is None: # 再次检查，确保元信息已获取
            raise RuntimeError("预处理或元信息提取在 fit 中失败！")

        # 处理验证数据
        if X_val_raw is not None and y_val is not None:
            y_val = np.asarray(y_val)
             # 使用已拟合的预处理器转换验证集
            X_num_val, X_cat_val, _ = self._preprocess_data(X_val_raw, fit_preprocessor=False)
        else:
            # 如果没有提供验证集，从处理后的训练数据中划分
            if self.verbose: print("  未提供验证集，从处理后的训练集中划分 20%...")
            indices = np.arange(X_num_train.shape[0])
            train_indices, val_indices = train_test_split(
                indices, test_size=0.2, random_state=self.random_seed, stratify=y_train
            )
            # 使用索引划分处理后的数据
            X_num_val, X_cat_val, y_val = X_num_train[val_indices], X_cat_train[val_indices], y_train[val_indices]
            X_num_train, X_cat_train, y_train = X_num_train[train_indices], X_cat_train[train_indices], y_train[train_indices]
            if self.verbose: print(f"  划分后: 训练集 {len(train_indices)} 样本, 验证集 {len(val_indices)} 样本")

        # --- 2. 准备数据加载器 ---
        if self.verbose: print("  准备数据加载器...")
        train_dataset = AdvancedTabularDataset(X_num_train, X_cat_train, y_train)
        val_dataset = AdvancedTabularDataset(X_num_val, X_cat_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # --- 3. 根据模式执行预训练 (如果需要) ---
        initial_weights_path: Optional[str] = None
        if self.learning_mode == "gradual" or self.learning_mode == "transfer_internal_pretrain":
            mode_desc = "Gradual Learning - Pretraining" if self.learning_mode=="gradual" else "Transfer Learning - Pretraining"
            if self.verbose: 
                print(f"\n--- 阶段: {mode_desc} ({self.pretrain_params['epochs']} epochs) ---")
            if self.learning_mode == "gradual": 
                pretrain_X_dataset = TensorDataset(torch.tensor(X_num_train, dtype=torch.float32), torch.tensor(X_cat_train, dtype=torch.int64))
                pretrain_loader = DataLoader(pretrain_X_dataset, batch_size=self.batch_size, shuffle=True)
            else: 
                X_num_pretrain, X_cat_pretrain, _ = self._preprocess_data(X_pretrain_raw, fit_preprocessor=False)
                pretrain_X_dataset = TensorDataset(torch.tensor(X_num_pretrain, dtype=torch.float32), torch.tensor(X_cat_pretrain, dtype=torch.int64))
                pretrain_loader = DataLoader(pretrain_X_dataset, batch_size=self.batch_size, shuffle=True)
            initial_weights_path = self._pretrain(pretrain_loader) # _pretrain 返回临时路径

        if self.learning_mode == "direct": 
            mode_desc = "Direct Learning"
        elif initial_weights_path: 
            mode_desc = "Gradual/Transfer - 监督微调"
        else: 
            mode_desc = "监督训练 (预训练失败?)"
        if self.verbose:
            print(f"\n--- 阶段: {mode_desc} ({self.supervised_params['epochs']} epochs max) ---")


        # --- 4. 执行监督训练或微调 ---
        if self.learning_mode == "direct":
            if self.verbose: print(f"\n--- 阶段: Direct Learning ({self.supervised_params['epochs']} epochs max) ---")
        elif initial_weights_path: # Gradual 或 Transfer
             mode_name = "Gradual Learning - 监督微调" if self.learning_mode == "gradual" else "Transfer Learning - 监督微调"
             if self.verbose: print(f"\n--- 阶段: {mode_name} ({self.supervised_params['epochs']} epochs max) ---")

        # 核心训练/微调步骤
        self.model_ = self._fit_supervised(train_loader, val_loader, 
                                           initial_weights_path=initial_weights_path,
                                           save_epoch_freq=self.save_epoch_freq,
                                           save_dir=self.save_dir)

        # --- 5. 清理临时文件 ---
        if initial_weights_path == self._temp_pretrain_path and os.path.exists(self._temp_pretrain_path):
            try: os.remove(self._temp_pretrain_path)
            except OSError as e: warnings.warn(f"删除临时文件失败: {e}", RuntimeWarning, stacklevel=2)
            if self.verbose: print(f"  已尝试删除临时预训练权重文件: {self._temp_pretrain_path}")

        if self.verbose: print("\n--- 模型训练流程完成 ---")
        
        if best_model_path := getattr(self, '_best_model_path', None): # 获取最佳模型路径属性（在_fit_supervised中设置）
            if self.verbose: print(f"验证集上性能最佳的模型已保存在: {best_model_path}")
        elif self.save_epoch_freq is not None and self.save_epoch_freq > 0:
             if self.verbose: print(f"每 {self.save_epoch_freq} epoch 的模型权重已保存在目录: {self.save_dir}")
        
        return self


    def _pretrain(self, pretrain_loader: DataLoader) -> str:
        """
        内部函数：执行自监督预训练。

        Args:
            pretrain_loader: 包含处理后的预训练特征数据的 DataLoader (仅含 X_num, X_cat)。

        Returns:
            保存预训练权重的临时文件路径。
        """
        # 1. 创建模型并切换到预训练头
        model = self._create_model(num_classes=self.n_classes_)
        model.add_pretraining_heads()
        model.to(self.device)

        # 2. 定义损失和优化器
        criterion_mse = nn.MSELoss(reduction='none')
        criterion_ce = nn.CrossEntropyLoss(reduction='none')
        optimizer = optim.AdamW(model.parameters(), 
                                lr=self.pretrain_params['lr'], 
                                weight_decay=self.pretrain_params['weight_decay'])


        if self.verbose: print(f"  开始自监督预训练，共 {self.pretrain_params['epochs']} epochs...")
        for epoch in range(self.pretrain_params['epochs']):
            model.train()
            total_epoch_loss = 0.0
            total_masked_tokens = 0

            # 3. 遍历预训练数据
            # 注意：loader 返回的是 (features_num, features_cat)
            for batch_idx, (features_num, features_cat) in enumerate(pretrain_loader):
                features_num = features_num.to(self.device)
                features_cat = features_cat.to(self.device)

                # a) 执行特征掩码
                _, _, masked_indices_num, masked_indices_cat, \
                original_num, original_cat = mask_features(
                    features_num, features_cat,
                    self.preprocessor_info_['num_numerical'],
                    list(self.preprocessor_info_['category_sizes'].keys()),
                    mask_probability=self.pretrain_params['mask_probability']
                )

                # 如果此批次没有特征被掩盖，则跳过
                num_masked_in_batch = masked_indices_num.sum().item() + masked_indices_cat.sum().item()
                if num_masked_in_batch == 0: 
                    continue

                # b) 梯度清零
                optimizer.zero_grad()

                # c) 模型前向传播
                # 输入掩码后的数据和掩码索引（虽然当前模型实现没用索引，但接口保留）
                predictions = model(features_num, features_cat, # 使用原始未标记的输入
                                    masked_indices_num, masked_indices_cat) # 传入掩码位置
                pred_num = predictions['pred_num']              # (B, N_num, 1)
                pred_cat_logits_dict = predictions['pred_cat'] # Dict{'name': (B, C_cat)}

                # d) 计算损失 (只在被掩盖的位置)
                pred_num = predictions['pred_num']
                pred_cat_logits_dict = predictions['pred_cat']
                #    数值损失 (MSE)
                loss_num_per_token = criterion_mse(pred_num.squeeze(-1), original_num) # (B, N_num)
                loss_num_masked = loss_num_per_token * masked_indices_num        # (B, N_num), 非掩码处为0
                total_loss_num_batch = loss_num_masked.sum()                     # 当前批次数值总损失

                #    分类损失 (CrossEntropy)
                total_loss_cat_batch = torch.tensor(0.0, device=self.device)
                num_masked_cat = masked_indices_cat.sum().item()
                for i, (name, logits) in enumerate(pred_cat_logits_dict.items()):
                    # logits: (B, C_cat), original_cat[:, i]: (B,)
                    loss_cat_i_per_token = criterion_ce(logits, original_cat[:, i]) # (B,)
                    mask_cat_i = masked_indices_cat[:, i]                     # (B,)
                    loss_cat_masked = loss_cat_i_per_token * mask_cat_i       # (B,), 非掩码处为0
                    total_loss_cat_batch += loss_cat_masked.sum()            # 累加当前批次分类总损失

                # e) 合并与平均损失
                # 当前批次总损失 = 数值总损失 + 分类总损失
                batch_total_loss = total_loss_num_batch + total_loss_cat_batch
                # 当前批次平均损失 = 总损失 / 被掩盖的 token 总数
                batch_average_loss = batch_total_loss / num_masked_in_batch

                # f) 反向传播与优化
                batch_average_loss.backward()
                optimizer.step()

                # 累加 epoch 统计数据
                total_epoch_loss += batch_total_loss.item() # 累加的是未平均的批次总损失
                total_masked_tokens += num_masked_in_batch   # 累加被掩盖的 token 总数

            # 计算整个 epoch 的平均损失
            epoch_average_loss = total_epoch_loss / total_masked_tokens if total_masked_tokens > 0 else 0.0

            # 打印训练信息 (例如，每 10 轮)
            if self.verbose and (epoch + 1) % 10 == 0:
                 print(f"  Pretrain Epoch [{epoch+1}/{self.pretrain_params['epochs']}] | Avg Masked Loss: {epoch_average_loss:.4f}")

        # 4. 保存最终预训练模型的 state_dict 到临时文件
        torch.save(model.state_dict(), self._temp_pretrain_path)
        if self.verbose: print(f"  内部预训练完成, 权重保存至临时文件: {self._temp_pretrain_path}")

        # 5. 返回临时文件路径
        return self._temp_pretrain_path


    def _fit_supervised(self, train_loader: DataLoader, val_loader: DataLoader,
                        initial_weights_path: Optional[Union[str, Path]] = None,
                        save_epoch_freq: Optional[int] = None,
                        save_dir: Optional[Path] = None
                        ) -> UnifiedTabularTransformer:
        """
        内部函数：执行监督学习或微调。

        Args:
            train_loader: 训练数据加载器 (含标签)。
            val_loader: 验证数据加载器 (含标签)。
            initial_weights_path: 预训练权重的路径 (如果是微调)。
            save_epoch_freq: 每隔多少个监督/微调 epoch 保存一次模型权重。设为 0 或 None 则不保存中间权重。
            save_dir: 用于保存 epoch 权重和最佳模型的目录名称。

        Returns:
            训练/微调后的最终模型。
        """
        # 1. 创建模型实例并确保是分类头
        model = self._create_model(num_classes=self.n_classes_)
        # 确保切换到分类头（即使是微调，也要确保移除了预训练头）
        model.add_classification_head(self.n_classes_, self.model_params['dropout'])
        model.to(self.device)

        # 2. 加载初始权重 (如果是微调)
        is_finetuning = initial_weights_path is not None
        if is_finetuning:
            if self.verbose: print(f"  加载初始权重从: {initial_weights_path}...")
            try:
                pretrained_dict = torch.load(initial_weights_path, map_location=self.device)
                model_dict = model.state_dict()
                # 过滤掉头部层和不匹配的层
                # 确保只加载 Transformer 主体和嵌入层的权重
                pretrained_dict_loaded = {
                    k: v for k, v in pretrained_dict.items() if k in model_dict and \
                    model_dict[k].shape == v.shape and \
                    'classifier' not in k and 'final_norm' not in k and \
                    'regression_head' not in k and 'classification_heads_cat' not in k
                }
                loaded_count = len(pretrained_dict_loaded)
                if self.verbose: print(f"    成功加载 {loaded_count} 个兼容的预训练参数层。")
                if not loaded_count and self.verbose:
                    warnings.warn("未加载任何兼容的预训练权重！请检查模型架构和权重文件。", RuntimeWarning, stacklevel=2)

                # 更新模型 state_dict 并加载
                model_dict.update(pretrained_dict_loaded)
                model.load_state_dict(model_dict)

            except Exception as e:
                 warnings.warn(f"加载权重时出错: {e}。将从头开始训练。", RuntimeWarning, stacklevel=2)
                 is_finetuning = False # 加载失败，则标记为从头训练

        # 3. 设置学习率和优化器
        # 微调时学习率降低 10 倍
        lr_factor = 0.1 if is_finetuning else 1.0
        learning_rate = self.supervised_params['lr'] * lr_factor
        if self.verbose: print(f"  使用学习率: {learning_rate}")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(),
                                lr=learning_rate,
                                weight_decay=self.supervised_params['weight_decay'])

        # 4. 初始化早停相关变量
        best_val_metric = -float('inf') # 使用 AUC 作为指标，越高越好
        best_model_state: Optional[Dict[str, torch.Tensor]] = None
        epochs_no_improve = 0
        patience = self.supervised_params['patience']
        mode_name = "微调" if is_finetuning else "训练"
        if self.verbose: print(f"  开始监督 {mode_name}...")

        # 5. 监督训练/微调循环
        for epoch in range(self.supervised_params['epochs']):
            # --- 训练 ---
            model.train()
            total_train_loss = 0.0
            total_train_samples = 0
            for batch_idx, (features_num, features_cat, labels) in enumerate(train_loader):
                # 数据移动到设备
                features_num = features_num.to(self.device)
                features_cat = features_cat.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                # 模型前向传播获取 logits
                outputs = model(features_num, features_cat)
                # outputs 可能已经是 logits，或者是一个包含 logits 的字典
                current_logits = outputs['logits'] if isinstance(outputs, dict) else outputs

                loss = criterion(current_logits, labels)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item() * labels.size(0) # 累加批次总损失
                total_train_samples += labels.size(0)

            epoch_train_loss = total_train_loss / total_train_samples if total_train_samples > 0 else 0.0

            # --- 验证 ---
            model.eval()
            total_val_loss = 0.0
            total_val_samples = 0
            all_val_labels_list: List[np.ndarray] = []
            all_val_probs_list: List[np.ndarray] = []
            with torch.no_grad():
                for features_num, features_cat, labels in val_loader:
                    features_num = features_num.to(self.device)
                    features_cat = features_cat.to(self.device)
                    labels = labels.to(self.device)

                    outputs = model(features_num, features_cat)
                    current_logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                    loss = criterion(current_logits, labels)

                    total_val_loss += loss.item() * labels.size(0)
                    total_val_samples += labels.size(0)
                    # 计算概率用于 AUC
                    probabilities = torch.softmax(current_logits, dim=1)
                    # 保存标签和正类概率
                    all_val_labels_list.append(labels.cpu().numpy())
                    all_val_probs_list.append(probabilities[:, 1].cpu().numpy()) # 假设 1 是正类

            epoch_val_loss = total_val_loss / total_val_samples if total_val_samples > 0 else 0.0
            # 合并所有批次的标签和概率
            all_val_labels = np.concatenate(all_val_labels_list) if all_val_labels_list else np.array([])
            all_val_probs = np.concatenate(all_val_probs_list) if all_val_probs_list else np.array([])
            # 计算验证集 AUC
            epoch_val_auc = 0.5 # 默认值
            if len(all_val_labels) > 0 and len(np.unique(all_val_labels)) > 1: # 确保有足够样本和至少两个类别
                try:
                    epoch_val_auc = roc_auc_score(all_val_labels, all_val_probs)
                except ValueError as e:
                    warnings.warn(f"计算 AUC 时出错: {e}", RuntimeWarning, stacklevel=2)
            elif len(all_val_labels) > 0:
                 warnings.warn("验证集中只存在一个类别，无法计算 AUC。", RuntimeWarning, stacklevel=2)

            # 打印信息
            if self.verbose:
                print(f"  Epoch [{epoch+1}/{self.supervised_params['epochs']}] | "
                      f"Train Loss: {epoch_train_loss:.4f} | "
                      f"Val Loss: {epoch_val_loss:.4f}, Val AUC: {epoch_val_auc:.4f}")
            
            if save_epoch_freq is not None and save_epoch_freq > 0 and save_dir is not None:
                if (epoch + 1) % save_epoch_freq == 0:
                    epoch_save_path = save_dir / f"epoch_{epoch+1}_model.pth"
                    torch.save(model.state_dict(), epoch_save_path)
                    if self.verbose: print(f"    已保存 Epoch {epoch+1} 的权重到: {epoch_save_path}")


            # --- 早停检查 ---
            current_val_metric = epoch_val_auc # 使用 AUC 作为早停指标
            if current_val_metric > best_val_metric:
                best_val_metric = current_val_metric
                best_model_state = copy.deepcopy(model.state_dict()) # 保存最佳模型状态
                epochs_no_improve = 0
                if save_dir is not None:
                     self._best_model_path = save_dir / "best_model.pth" # 记录路径
                     torch.save(best_model_state, self._best_model_path)
                     if self.verbose: print(f"    验证 AUC 提升至 {best_val_metric:.4f}，最佳模型已保存至: {self._best_model_path}")
                # ---
                elif self.verbose: print(f"    验证 AUC 提升至 {best_val_metric:.4f} (未指定 save_dir，最佳模型仅保存在内存中)。")

            else:
                epochs_no_improve += 1
                if self.verbose: print(f"    验证 AUC 未提升 {epochs_no_improve}/{patience} 轮。")

            if epochs_no_improve >= patience:
                if self.verbose: print(f"  早停触发于轮次 {epoch+1}")
                break

        # 6. 加载最佳模型状态
        if best_model_state:
            model.load_state_dict(best_model_state)
            if self.verbose: print(f"  加载验证集上最佳 AUC ({best_val_metric:.4f}) 对应的模型。")
        else:
            # 如果没有找到更好的状态（例如，训练轮数太少或patience太小）
            if self.verbose: print("  警告：未能找到更优的模型状态（基于验证AUC），使用最后一轮的模型。")

        return model

    # --- 预测和分析方法 ---
    def predict_proba(self, X_test_raw: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """进行概率预测，自动处理原始输入数据"""
        # 1. 检查模型和预处理器是否准备就绪
        if self.model_ is None:
            raise RuntimeError("模型尚未训练，请先调用 fit。")
        if self.preprocessor_ is None:
            raise RuntimeError("预处理器尚未拟合，请先调用 fit。")

        # 2. 预处理测试数据
        if self.verbose: print("  正在预处理测试数据...")
        X_num_test, X_cat_test, _ = self._preprocess_data(X_test_raw, fit_preprocessor=False)
        if self.verbose: print("  预处理完成。")

        # 3. 创建测试数据加载器
        test_dataset = AdvancedTabularDataset(X_num_test, X_cat_test, np.zeros(X_num_test.shape[0])) # 标签仅占位
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        # 4. 执行预测
        self.model_.eval() # 设置为评估模式
        all_probabilities_list: List[torch.Tensor] = []
        if self.verbose: print("  开始预测...")
        with torch.no_grad(): # 关闭梯度计算
            for features_num, features_cat, _ in test_loader:
                features_num = features_num.to(self.device)
                features_cat = features_cat.to(self.device)

                # 模型前向传播
                outputs = self.model_(features_num, features_cat)
                current_logits = outputs['logits'] if isinstance(outputs, dict) else outputs

                # 计算概率
                probabilities = torch.softmax(current_logits, dim=1)
                all_probabilities_list.append(probabilities.cpu()) # 移回 CPU

        # 5. 合并结果并返回 NumPy 数组
        all_probabilities = torch.cat(all_probabilities_list, dim=0).numpy()
        if self.verbose: print("  预测完成。")
        return all_probabilities

    def predict(self, X_test_raw: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """进行类别预测，自动处理原始输入数据"""
        # 获取概率预测
        probabilities = self.predict_proba(X_test_raw)
        # 返回概率最高的类别索引
        return np.argmax(probabilities, axis=1)

    def get_final_embeddings(self, X_raw: Union[pd.DataFrame, np.ndarray]) -> torch.Tensor:
         """获取最终特征嵌入，自动处理原始输入数据"""
         # 1. 检查状态
         if self.model_ is None: raise RuntimeError("模型未训练。")
         if self.preprocessor_ is None: raise RuntimeError("预处理器未拟合。")

         # 2. 预处理数据
         X_num, X_cat, _ = self._preprocess_data(X_raw, fit_preprocessor=False)

         # 3. 创建加载器
         dataset = AdvancedTabularDataset(X_num, X_cat, np.zeros(X_num.shape[0]))
         loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

         # 4. 获取嵌入
         self.model_.eval()
         all_embeddings_list: List[torch.Tensor] = []
         if self.verbose: print("  正在获取最终嵌入...")
         with torch.no_grad():
              for features_num, features_cat, _ in loader:
                   features_num = features_num.to(self.device)
                   features_cat = features_cat.to(self.device)
                   # 调用 forward 并请求嵌入
                   outputs = self.model_(features_num, features_cat, return_final_embeddings=True)
                   final_hidden = outputs['final_embeddings']
                   # 提取特征部分
                   embeds = final_hidden[:, self.model_.num_special_tokens:, :] if self.model_.use_cls_token else final_hidden
                   all_embeddings_list.append(embeds.cpu()) # 移回 CPU
         if self.verbose: print("  获取嵌入完成。")

         # 5. 合并返回
         return torch.cat(all_embeddings_list, dim=0)

    def get_attention_weights(self, X_raw: Union[pd.DataFrame, np.ndarray], layer_index: int = -1) -> Optional[List[torch.Tensor]]:
         """获取指定层的注意力权重，自动处理原始输入数据 (只处理第一个样本)"""
         # 1. 检查状态
         if self.model_ is None: raise RuntimeError("模型未训练。")
         if self.preprocessor_ is None: raise RuntimeError("预处理器未拟合。")

         # 2. 预处理数据 (只取第一个样本以方便处理)
         if isinstance(X_raw, pd.DataFrame):
              X_first_sample_raw = X_raw.iloc[:1]
         else:
              X_first_sample_raw = X_raw[:1]
         X_num, X_cat, _ = self._preprocess_data(X_first_sample_raw, fit_preprocessor=False)

         # 3. 获取权重
         self.model_.eval()
         attention_weights: Optional[List[torch.Tensor]] = None
         if self.verbose: print("  正在获取第一个样本的注意力权重...")
         with torch.no_grad():
              features_num = torch.tensor(X_num, dtype=torch.float32).to(self.device)
              features_cat = torch.tensor(X_cat, dtype=torch.int64).to(self.device)
              # 调用 forward 并请求注意力权重
              outputs = self.model_(features_num, features_cat, return_attention=True)
              if 'attention_weights' in outputs and outputs['attention_weights']:
                   att_list = outputs['attention_weights']
                   # 处理层索引
                   if layer_index < 0: layer_index = len(att_list) + layer_index
                   if 0 <= layer_index < len(att_list):
                       # 返回指定层的权重 (移回 CPU)
                       attention_weights = [att_list[layer_index].cpu()] # 放入列表以保持一致性
                   else:
                       warnings.warn(f"请求的层索引 {layer_index} 超出范围。", RuntimeWarning, stacklevel=2)
              else:
                   warnings.warn("模型未返回注意力权重。", RuntimeWarning, stacklevel=2)
         if self.verbose: print("  获取注意力权重完成。")
         return attention_weights # 返回包含单个样本权重的列表，或 None