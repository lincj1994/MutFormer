import torch
import torch.nn as nn
import warnings
from typing import Literal, Dict, Optional, Union, List, Any, Tuple, Self


# --------------------------------------------------------------------------
# Transformer 辅助模块
# --------------------------------------------------------------------------
class MultiheadAttentionWithResidual(nn.Module):
    """
    一个辅助模块，封装了多头注意力 (MultiheadAttention) 层，
    并集成了残差连接 (Residual Connection) 和层归一化 (Layer Normalization)。
    这是 Transformer 模型中的标准组件。
    """
    def __init__(self, d_model, nhead, dropout=0.1, batch_first=True):
        """
        Args:
            d_model (int): 输入输出的特征维度 (嵌入维度)。
            nhead (int): 多头注意力机制中的头数。d_model 必须能被 nhead 整除。
            dropout (float): Attention 权重上的 dropout 比率。
            batch_first (bool): 如果为 True，则输入输出张量的形状为 (batch, seq, feature)，否则为 (seq, batch, feature)。
        """
        super().__init__() # 初始化父类
        # 实例化 PyTorch 内置的多头注意力层
        self.mha = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        # 实例化层归一化
        self.norm = nn.LayerNorm(d_model)
        # 实例化 Dropout 层
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        """
        前向传播函数。
        Args:
            query, key, value (Tensor): 注意力机制的 Q, K, V 输入。对于自注意力，它们是相同的。
            attn_mask (Tensor, optional): 注意力掩码，防止注意力关注到不应关注的位置 (例如 padding)。
            key_padding_mask (Tensor, optional): 键填充掩码，指示哪些 K 值是填充的，不应参与计算。
        Returns:
            tuple: (注意力层输出, 注意力权重)
        """
        # 调用多头注意力层，得到注意力的输出和权重
        # attn_output 形状通常与 query 相同
        # attn_weights 形状取决于具体实现，例如 (batch, num_heads, seq_len_q, seq_len_k) 或 (batch, seq_len_q, seq_len_k)
        attn_output, attn_weights = self.mha(query, key, value,
                                             attn_mask=attn_mask,
                                             key_padding_mask=key_padding_mask)
        # 残差连接：将注意力层的输出与原始输入 (query) 相加
        # Dropout 应用于注意力输出，防止过拟合
        # Layer Normalization 应用于残差连接后的结果，稳定训练
        output = query + self.dropout(attn_output) # 残差连接
        output = self.norm(output)                 # 层归一化
        return output, attn_weights                # 返回处理后的输出和注意力权重

class FeedForwardWithResidual(nn.Module):
    """
    一个辅助模块，封装了前馈网络 (Feed Forward Network)，
    同样集成了残差连接和层归一化。这也是 Transformer 的标准组件。
    通常包含两个线性层和一个非线性激活函数。
    """
    def __init__(self, d_model, dim_feedforward, dropout=0.1):
        """
        Args:
            d_model (int): 输入输出的特征维度。
            dim_feedforward (int): 前馈网络中间隐藏层的维度 (通常是 d_model 的 2 或 4 倍)。
            dropout (float): Dropout 比率。
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward) # 第一个线性层，扩展维度
        self.dropout1 = nn.Dropout(dropout)
        self.activation = nn.GELU() # 使用 GELU 激活函数 (也可以用 ReLU)
        self.linear2 = nn.Linear(dim_feedforward, d_model) # 第二个线性层，恢复维度
        self.dropout2 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)                  # 层归一化

    def forward(self, x):
        """
        前向传播函数。
        Args:
            x (Tensor): 输入张量，形状 (..., d_model)。
        Returns:
            Tensor: 输出张量，形状与输入相同。
        """
        residual = x # 保存原始输入用于残差连接
        # 通过前馈网络
        x = self.linear1(x)
        x = self.dropout1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        # 残差连接和层归一化
        output = residual + x # 残差连接
        output = self.norm(output) # 层归一化
        return output

class TransformerBlock(nn.Module):
    """
    一个完整的 Transformer 编码器块 (Encoder Block)，
    它按顺序组合了 自注意力 (Self-Attention) 和 前馈网络 (Feed Forward Network)。
    这是标准的列间注意力 (处理特征间关系) 的模块。
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, batch_first=True):
        """
        Args:
            d_model (int): 特征维度。
            nhead (int): 注意力头数。
            dim_feedforward (int): 前馈网络隐藏层维度。
            dropout (float): Dropout 比率。
            batch_first (bool): 输入输出格式是否 batch 在第一维。
        """
        super().__init__()
        # 实例化带残差连接的自注意力模块
        self.self_attn = MultiheadAttentionWithResidual(d_model, nhead, dropout, batch_first)
        # 实例化带残差连接的前馈网络模块
        self.feed_forward = FeedForwardWithResidual(d_model, dim_feedforward, dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        前向传播函数。
        Args:
            src (Tensor): 输入序列，形状 (batch, seq_len, d_model) 或 (seq_len, batch, d_model)。
            src_mask (Tensor, optional): 自注意力的掩码。
            src_key_padding_mask (Tensor, optional): 输入序列的填充掩码。
        Returns:
            tuple: (Transformer 块的输出, 自注意力权重)
        """
        # 1. 通过自注意力层
        #    对于自注意力，Q, K, V 都来自输入 src
        attn_output, attn_weights = self.self_attn(src, src, src,
                                                  attn_mask=src_mask,
                                                  key_padding_mask=src_key_padding_mask)
        # 2. 通过前馈网络层
        output = self.feed_forward(attn_output)
        return output, attn_weights # 返回最终输出和该块的注意力权重

class InterSampleAttentionBlock(nn.Module):
    """
    行间注意力（样本间注意力）模块。
    这个模块让模型关注 **不同样本** 在 **相同特征维度** 上的关系。
    实现方式：将 Batch 维度和 Token (特征) 维度进行转置，然后应用标准的自注意力。
    灵感来源于 SAINT 等模型。
    """
    def __init__(self, d_model, nhead, dropout=0.1):
        """
        Args:
            d_model (int): 特征维度。
            nhead (int): 注意力头数。
            dropout (float): Dropout 比率。
        """
        super().__init__()
        # 实例化带残差连接的注意力模块
        # 注意：这里 batch_first=False，因为我们转置后，原来的 Token 维度（num_tokens）在前面，Batch 维度在中间。
        # MultiheadAttention 默认期望 (seq_len, batch, feature)
        self.inter_sample_attn = MultiheadAttentionWithResidual(d_model, nhead, dropout, batch_first=False)

    def forward(self, x):
        """
        前向传播函数。
        Args:
            x (torch.Tensor): 输入张量，形状 (batch_size, num_tokens, d_model)。
                              num_tokens 通常是 特征数 + CLS Token 数。
        Returns:
            torch.Tensor: 输出张量，形状与输入相同 (batch_size, num_tokens, d_model)。
        """
        # 1. 转置 Batch 维度和 Token 维度
        #    原始形状: (batch_size, num_tokens, d_model)
        #    转置后:   (num_tokens, batch_size, d_model)
        #    现在，batch_size 成了序列长度，模型将在不同样本间计算注意力
        x_permuted = x.permute(1, 0, 2)

        # 2. 应用注意力机制
        #    Q, K, V 都来自转置后的 x_permuted
        #    注意力是在 batch_size 这个维度上计算的
        attn_output, _ = self.inter_sample_attn(x_permuted, x_permuted, x_permuted)
        #    我们通常不关心行间注意力的权重，所以用 _ 忽略它

        # 3. 转置回原始形状
        #    从 (num_tokens, batch_size, d_model) 转回 (batch_size, num_tokens, d_model)
        output = attn_output.permute(1, 0, 2)
        return output



# --------------------------------------------------------------------------
# 统一的 Transformer 模型类
# --------------------------------------------------------------------------
class UnifiedTabularTransformer(nn.Module):
    """
    统一的 Transformer 模型类，支持预训练和分类/微调模式切换。
    """
    def __init__(self, *,
                 num_numerical_features: int,         # 模型接收的数值输入总数 (含指示器等)
                 category_sizes_dict: Dict[str, int], # 分类特征名 -> 类别数字典
                 d_model: int,                        # 嵌入维度
                 nhead: int,                          # 注意力头数
                 num_layers: int,                     # Transformer 层数
                 dim_feedforward: int,                # FFN 中间层维度
                 dropout: float = 0.1,                # Dropout 比率
                 num_classes: int = 2,                # 最终分类任务的类别数
                 use_cls_token: bool = True           # 是否使用 CLS Token
                 ):
        super().__init__()
        # --- 保存配置 ---
        self.num_numerical = num_numerical_features
        self.category_info = category_sizes_dict
        self.categorical_feature_names = list(category_sizes_dict.keys())
        self.num_categorical = len(self.categorical_feature_names)
        # !! 模型接收的数值输入(含指示器) + 分类输入总数 !!
        self.num_features_model_in = self.num_numerical + self.num_categorical
        self.d_model = d_model
        self.use_cls_token = use_cls_token

        # --- 核心 Transformer 组件 ---
        # 数值特征嵌入层 (为每个数值输入创建一个线性层)
        self.numerical_embedders = nn.ModuleList(
            [nn.Linear(1, d_model) for _ in range(self.num_numerical)]
        )
        # 分类特征嵌入层 (为每个分类特征创建一个 Embedding 层)
        self.categorical_embedders = nn.ModuleDict({
            name: nn.Embedding(size, d_model) for name, size in self.category_info.items()
        })
        # 列位置编码 (可学习)
        self.column_positional_embeddings = nn.Parameter(
            torch.randn(1, self.num_features_model_in, d_model) # 形状 (1, N_model_in, D)
        )
        # CLS Token 嵌入 (如果使用)
        self.cls_token_embedding: Optional[nn.Parameter] = None
        self.num_special_tokens = 0
        if self.use_cls_token:
            self.cls_token_embedding = nn.Parameter(torch.randn(1, 1, d_model)) # 形状 (1, 1, D)
            self.num_special_tokens = 1
        # Mask Token 嵌入 (用于预训练)
        self.mask_token_embedding = nn.Parameter(torch.randn(1, 1, d_model)) # 形状 (1, 1, D)

        # Transformer 层堆叠
        self.layers = nn.ModuleList([
            nn.ModuleList([
                TransformerBlock(d_model, nhead, dim_feedforward, dropout, batch_first=True),
                InterSampleAttentionBlock(d_model, nhead, dropout),
                FeedForwardWithResidual(d_model, dim_feedforward, dropout)
            ]) for _ in range(num_layers)])

        # --- 动态任务头部 (初始化为空) ---
        self.regression_head: Optional[nn.Linear] = None
        self.classification_heads_cat: Optional[nn.ModuleDict] = None
        self.final_norm: Optional[nn.LayerNorm] = None
        self.classifier: Optional[nn.Linear] = None
        self.dropout_layer: Optional[nn.Dropout] = None

        # --- 默认添加分类头 ---
        self.add_classification_head(num_classes, dropout)

    def _embed_features(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """
        一个内部辅助函数，用于对数值和分类特征进行嵌入。
        Args:
            x_num (Tensor): 数值特征，形状 (batch_size, num_numerical)。
            x_cat (Tensor): 分类特征 (整数索引)，形状 (batch_size, num_categorical)。
        Returns:
            Tensor: 嵌入后的特征张量，形状 (batch_size, num_features, d_model)。
        """
        batch_size = x_num.shape[0]
        embeddings = []
        # 处理数值特征
        for i in range(self.num_numerical):
            # 输入 (B, 1), 输出 (B, D)
            num_emb = self.numerical_embedders[i](x_num[:, i:i+1])
            embeddings.append(num_emb)
        # 处理分类特征
        for i, name in enumerate(self.categorical_feature_names):
            # 输入 (B,), 输出 (B, D)
            cat_emb = self.categorical_embedders[name](x_cat[:, i])
            embeddings.append(cat_emb)
        # 在特征维度上堆叠
        # 输出 (B, N_model_in, D)
        return torch.stack(embeddings, dim=1)

    def add_pretraining_heads(self):
        """切换到预训练模式：添加掩码预测头，移除分类头"""
        if self.verbose: print("切换到预训练头部。")
        # 获取当前设备，确保新层在同一设备上
        current_device = self.column_positional_embeddings.device

        # 创建数值预测头
        self.regression_head = nn.Linear(self.d_model, 1).to(current_device)
        # 创建分类预测头 (为每个分类特征创建一个)
        self.classification_heads_cat = nn.ModuleDict()
        for name, size in self.category_info.items():
            self.classification_heads_cat[name] = nn.Linear(self.d_model, size).to(current_device)

        # 移除分类任务相关的头
        self.final_norm = None
        self.classifier = None
        self.dropout_layer = None

    def add_classification_head(self, num_classes: int, dropout: float):
        """切换到分类/微调模式：添加分类头，移除预训练头"""
        if hasattr(self, 'verbose') and self.verbose: print("切换到分类头部。")
         # 获取当前设备
        current_device = self.column_positional_embeddings.device

        # 创建分类任务相关的头
        self.final_norm = nn.LayerNorm(self.d_model).to(current_device)
        self.classifier = nn.Linear(self.d_model, num_classes).to(current_device)
        self.dropout_layer = nn.Dropout(dropout)

        # 移除预训练任务相关的头
        self.regression_head = None
        self.classification_heads_cat = None

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor,
                masked_indices_num: Optional[torch.Tensor] = None,
                masked_indices_cat: Optional[torch.Tensor] = None,
                return_attention: bool = True,
                return_final_embeddings: bool = True
                ) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        统一的前向传播函数。

        Args:
            x_num: 数值特征输入 (B, N_num)。
            x_cat: 分类特征输入 (B, N_cat)。
            masked_indices_num: 预训练时数值特征的掩码 (B, N_num)。
            masked_indices_cat: 预训练时分类特征的掩码 (B, N_cat)。
            return_attention (bool): 如果为 True，则在返回字典中包含 'attention_weights'。
            return_final_embeddings (bool): 如果为 True，则在返回字典中包含 'final_embeddings'。

        Returns:
            torch.Tensor or dict:
                如果 return_attention 和 return_final_embeddings 都为 False，则返回 logits 张量。
                否则，返回一个包含 'logits' 以及可选 'attention_weights' 和 'final_embeddings' 的字典。
                'attention_weights' 是一个列表，包含每个 TransformerBlock 的列注意力权重。
                'final_embeddings' 是最后一个 Transformer 块输出的所有 token 的隐藏状态。
        """
        batch_size = x_num.shape[0]

        # 1. 获取基础特征嵌入
        # x 形状: (B, N_model_in, D)
        x = self._embed_features(x_num, x_cat)

        # 2. 添加列位置编码
        x = x + self.column_positional_embeddings

        # 3. 判断模式并处理掩码（仅预训练）
        is_pretraining = self.regression_head is not None
        if is_pretraining:
            if masked_indices_num is not None and masked_indices_cat is not None:
                # 展开 MASK token 嵌入以匹配批次大小
                mask_token_expanded = self.mask_token_embedding.expand(batch_size, -1, -1) # (B, 1, D)
                # 合并数值和分类掩码
                combined_mask = torch.cat([masked_indices_num, masked_indices_cat], dim=1) # (B, N_model_in)
                # 扩展掩码维度以匹配嵌入张量 x
                mask_condition = combined_mask.unsqueeze(-1).expand_as(x) # (B, N_model_in, D)
                # 使用 torch.where 将掩码位置的 *嵌入向量* 替换为 MASK token 嵌入
                x = torch.where(mask_condition, mask_token_expanded.expand_as(x), x)
                # --------------------
            elif self.training: # 只在训练模式下预训练但未收到掩码时警告
                 warnings.warn("模型处于预训练模式但未接收到掩码索引。输入未被掩码。", RuntimeWarning, stacklevel=2)

        # 4. 添加 CLS Token (如果使用)
        if self.use_cls_token:
            assert self.cls_token_embedding is not None
            cls_tokens = self.cls_token_embedding.expand(batch_size, -1, -1) # (B, 1, D)
            x = torch.cat([cls_tokens, x], dim=1) # 形状: (B, N_model_in + 1, D)

        # 5. 通过 Transformer 层
        all_attention_weights: List[torch.Tensor] = []
        for layer_idx, (col_attn_block, row_attn_block, row_ffn_block) in enumerate(self.layers):
            # a) 列间注意力 (TransformerBlock)
            x, col_attn_weights = col_attn_block(x)
            # 如果需要，存储注意力权重
            if return_attention:
                all_attention_weights.append(col_attn_weights.detach()) # 分离计算图

            # b) 行间注意力
            x = row_attn_block(x)

            # c) 行间注意力后的前馈网络
            x = row_ffn_block(x)

        # 6. 保存最终隐藏状态
        # x 形状: (B, N_model_in + N_special, D)
        final_hidden_states = x

        # --- 7. 构建返回字典或张量 ---
        return_payload: Dict[str, Any] = {}
        # 如果请求，添加最终嵌入（所有 token）
        if return_final_embeddings:
            return_payload['final_embeddings'] = final_hidden_states
        # 如果请求，添加注意力权重列表
        if return_attention and all_attention_weights:
            return_payload['attention_weights'] = all_attention_weights

        # a) 如果是预训练模式
        if is_pretraining:
            # 提取特征部分的隐藏状态（跳过 CLS）
            if self.use_cls_token:
                feature_hidden_states = final_hidden_states[:, self.num_special_tokens:, :]
            else:
                feature_hidden_states = final_hidden_states
            # 分离数值和分类部分的隐藏状态
            hidden_states_num = feature_hidden_states[:, :self.num_numerical, :] # (B, N_num, D)
            hidden_states_cat = feature_hidden_states[:, self.num_numerical:, :] # (B, N_cat, D)

            # 通过各自的预测头
            assert self.regression_head is not None
            pred_num = self.regression_head(hidden_states_num) # (B, N_num, 1)
            pred_cat_logits: Dict[str, torch.Tensor] = {}
            assert self.classification_heads_cat is not None
            for i, name in enumerate(self.categorical_feature_names):
                cat_hidden_state = hidden_states_cat[:, i, :] # (B, D)
                pred_cat_logits[name] = self.classification_heads_cat[name](cat_hidden_state) # (B, C_cat)

            # 添加到返回字典
            return_payload['pred_num'] = pred_num
            return_payload['pred_cat'] = pred_cat_logits
            # 预训练模式总是返回字典
            return return_payload

        # b) 如果是分类/微调模式
        elif self.classifier is not None:
            # 提取用于分类的 token 表示
            if self.use_cls_token:
                output_token = final_hidden_states[:, 0, :] # CLS token
            else: # 使用特征 token 的平均池化
                feature_tokens = final_hidden_states[:, self.num_special_tokens:, :]
                output_token = feature_tokens.mean(dim=1)

            # 通过最终的 LayerNorm, Dropout 和分类器
            assert self.final_norm is not None and self.dropout_layer is not None
            output_token = self.final_norm(output_token)
            output_token = self.dropout_layer(output_token)
            logits = self.classifier(output_token) # (B, N_classes)

            # 添加到返回字典
            return_payload['logits'] = logits

            # 根据请求决定最终返回内容
            if not return_final_embeddings and not return_attention:
                return logits # 默认只返回 logits
            else:
                return return_payload # 返回包含额外信息的字典
        else:
            # 如果模型没有配置任何有效的输出头，则抛出错误
            raise RuntimeError("模型既没有预训练头也没有分类头！")