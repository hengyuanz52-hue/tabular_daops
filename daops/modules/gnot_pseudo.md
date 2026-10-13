# GNOT 整体结构伪代码说明

基于 `gnot.py` 与 GNOT 论文 (https://arxiv.org/abs/2302.14376)，以下用伪代码概括从前向到各子模块的逻辑。

---

## 一、主模型 GNOT：输入 → 输出

```
函数 GNOT_forward(x, y, mask_query, masks_functions, key):

    # ---------- 1. 坐标与 query 的预处理与提升 ----------
    coords = x[:, :spatial_dim]                    # 取出空间坐标
    coords = coord_transform(coords)                # 标准化到 [-1,1] 等
    x = MLP_trunk(x)                                # (len_query, input_dim) -> (len_query, latent_dim)

    # ---------- 2. 各输入函数的预处理与编码 ----------
    y_out = {}
    for func_name in y:
        y_normed = function_transforms[func_name](y[func_name])   # 归一化/标准化
        y_out[func_name] = MLP_branch[func_name](y_normed)        # 提升到 latent_dim

    # ---------- 3. 堆叠 N 个 Gated Attention Block ----------
    for block in attn_blocks:
        x = GatedAttentionBlock(x, coords, y_out, mask_query, masks_functions, key)

    # ---------- 4. 投影到输出并反归一化 ----------
    out = projection(x)           # (len_query, latent_dim) -> (len_query, output_dim)
    return output_transform(out)  # 反归一化到物理量
```

- **x**: 查询点，形状 `(len_query, input_dim)`，前 `spatial_dim` 维是坐标。
- **y**: 多组输入函数，`y[func_name]` 形状 `(len_key, d_func)`。
- **coord_transform / function_transforms**: 标准化/归一化（非可学习或简单可学习）。
- **MLP_trunk / MLP_branch / projection**: 均为 MLP，将输入提升到 `latent_dim` 或映射到 `output_dim`。

---

## 二、单个 Gated Attention Block

```
函数 GatedAttentionBlock(x, coords, y, mask_query, masks_functions, key):

    # ---------- 门控：由坐标得到专家权重 (softmax，非 sigmoid) ----------
    gate_scores = MLP_gate(coords)           # (len_query, spatial_dim) -> (len_query, n_experts)
    gate_scores = softmax(gate_scores, dim=-1)

    # ---------- 对 x、y 做 LayerNorm ----------
    x = LayerNorm_x(x)
    for fname in y:
        y[fname] = LayerNorm_y[fname](y[fname])

    # ---------- Cross-Attention：query=x, key/value=各 function 的 y ----------
    x = x + dropout(CrossAttention(x, y, mask_query, masks_functions))

    # ---------- MoE：多个专家 MLP，用 gate_scores 加权 ----------
    moe_out = 0
    for i in range(n_experts):
        moe_out += gate_scores[:, i:i+1] * MLP_expert_cross[i](x)
    x = x + LayerNorm(moe_out)

    # ---------- Self-Attention ----------
    x = x + dropout(SelfAttention(x, mask_query))

    # ---------- 再次 MoE ----------
    moe_out = 0
    for i in range(n_experts):
        moe_out += gate_scores[:, i:i+1] * MLP_expert_self[i](x)
    x = x + LayerNorm(moe_out)

    return x
```

- 门控是 **softmax over experts**，不是 sigmoid。
- Cross / Self 两路各自接一层「专家 MLP + 门控加权」，即 MoE。

---

## 三、CrossAttention（归一化线性注意力）

```
函数 CrossAttention(x, y, mask_query, masks_functions):

    # Query：来自 x
    Q = Linear_query(x)                    # (len_query, latent_dim)
    Q = reshape_to_heads(Q)                # (n_heads, len_query, d_head)
    Q = softmax(Q, axis=-1)                # 归一化线性注意力的要求

    out = Q   # 先初始化为 Q，再逐个 function 加 attention 输出
    for fname in y:
        # Key, Value：来自该 function 的编码 y[fname]
        K = Linear_key[fname](y[fname])    # (len_key, latent_dim)
        K = reshape_to_heads(K)
        K = softmax(K, axis=-1)

        V = Linear_value[fname](y[fname])
        V = reshape_to_heads(V)

        # 归一化线性注意力核：output = (Q @ K^T @ V) / alpha
        alpha = sum(K, axis=0) * Q
        alpha = sum(alpha, axis=-1, keepdim=True)
        attn_out = (Q @ K.T @ V) / alpha
        out = out + attn_out

    out = reshape_back(out)
    out = Linear_proj(out)
    return out
```

- 对 Q、K 在最后一维做 **softmax**，再用 `(Q @ K^T @ V) / alpha`，即 **Normalized Linear Attention**，复杂度对序列长度为线性。

---

## 四、SelfAttention

```
函数 SelfAttention(x, mask_query):

    # 把同一 x 既当 query 又当 key/value
    return CrossAttention(x, y={'latent': x}, mask_query, masks_functions={'latent': mask_query})
```

---

## 五、归一化线性注意力的核（base 中的公式）

```
函数 normalized_linear_dot_product(Q, K, V):
    # Q (len_query, d), K (len_key, d), V (len_key, d)
    # 要求 Q、K 已在对应维度上做过 softmax
    alpha = sum(K, axis=0, keepdim=True) * Q
    alpha = sum(alpha, axis=1, keepdim=True)
    return (Q @ K.T @ V) / alpha
```

---

## 六、模块与形状一览

| 模块 | 输入 | 输出 |
|------|------|------|
| coord_transform | coords (len_query, spatial_dim) | 同形状，标准化 |
| MLP_trunk | x (len_query, input_dim) | (len_query, latent_dim) |
| function_transforms[f] | y[f] (len_key, d_f) | 同形状，归一化 |
| MLP_branch[f] | 归一化后的 y[f] | (len_key, latent_dim) |
| MLP_gate | coords (len_query, spatial_dim) | (len_query, n_experts) |
| CrossAttention | x, y | (len_query, latent_dim) |
| SelfAttention | x | (len_query, latent_dim) |
| projection | x (len_query, latent_dim) | (len_query, output_dim) |
| output_transform | 模型输出 | 反归一化到物理量 |

---

## 七、GNOTAdaptiveGatedAttentionBlock 与 GNOTGatedAttentionBlock 的差别

- **GNOTGatedAttentionBlock**：门控输入是 **坐标** `coords`，即 `gate_scores = MLP_gate(coords)`。
- **GNOTAdaptiveGatedAttentionBlock**：门控输入是 **cross-attention 的几何相关输出** `geo_attn`（来自部分 function 的 attention 结果），即 `gate_scores = MLP_gate(geo_attn)`，用于需要几何感知的变体（如 GeoGNOT）。

以上即 `gnot.py` 从前向到各子模块的伪代码级解释。
