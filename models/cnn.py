import torch
import torch.nn as nn

class base_Model(nn.Module):
    def __init__(self, configs):
        super(base_Model, self).__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=configs.cnn_input_channels_1, out_channels=configs.cnn_output_channels_1, kernel_size=configs.kernel_size, stride=3, padding=2),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=configs.cnn_input_channels_2, out_channels=configs.cnn_output_channels_2, kernel_size=configs.kernel_size, stride=3, padding=2), 
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=configs.cnn_input_channels_3, out_channels=configs.cnn_output_channels_3, kernel_size=configs.kernel_size, stride=3, padding=2), 
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool1d(output_size=1)
        )

        # 选择不同的 predictor
        
        print("Using predictor: ", configs.output_module)
        if configs.output_module == 'predictor_mlp':
            self.predictor = Predictor_mlp(configs, input_dim=configs.cnn_output_channels_3)
        elif configs.output_module == 'predictor_cnn':
            self.predictor = Predictor_cnn(configs, input_dim=configs.cnn_output_channels_3)
        elif configs.output_module == 'predictor_lstm':
            self.predictor = Predictor_lstm(configs, input_dim=configs.cnn_output_channels_3)
        elif configs.output_module == 'predictor_transformer':
            self.predictor = Predictor_transformer(configs, input_dim=configs.cnn_output_channels_3)
        else:
            self.predictor = nn.Linear(configs.cnn_output_channels_3, 1)

        # 新增：用于存储特征重要性排序的属性 (在训练后评估阶段赋值)
        self.feature_ranking = None 
        self.active_feature_count = None # 当前实验激活的特征数量
    def forward(self, x):
        batch_size, cycles, channels, seq_len = x.size()
        x = x.view(batch_size * cycles, channels, seq_len)
        x = self.conv_block(x)
        x = x.squeeze(-1)
        x = x.view(batch_size, cycles, -1)
        # === 特征筛选逻辑 ===
        if self.feature_ranking is not None and self.active_feature_count is not None:
            # 创建一个全0掩码
            mask = torch.zeros_like(x)
            
            # 获取 Top-K 的特征索引
            # feature_ranking 应该是一个列表或 tensor，包含按重要性排序的索引 [idx_most_imp, ..., idx_least_imp]
            top_k_indices = self.feature_ranking[:self.active_feature_count]
            
            # 将 Top-K 的位置设为 1 (保留原始值)，其余位置保持 0 (屏蔽)
            # 注意：这里假设你不想改变输入维度大小，只是把不重要的特征抹零
            mask[:, :, top_k_indices] = 1.0
            x = x * mask
        # ============================
        
        predictions = self.predictor(x).squeeze(-1)
        return predictions, x
    

class Predictor_mlp(nn.Module):
    def __init__(self, configs, input_dim=None):
        super(Predictor_mlp, self).__init__()
        inp = input_dim if input_dim is not None else configs.cnn_output_channels_3
        # 使用与 base_Model 类似的隐藏结构，但输入维度为 inp（已扁平）
        self.fc1 = nn.Linear(inp, configs.hidden_channels)
        self.fc2 = nn.Linear(configs.hidden_channels, configs.hidden_channels)
        self.fc3 = nn.Linear(configs.hidden_channels, configs.mlp_final_output)
        self.fc4 = nn.Linear(configs.mlp_final_output, 1)
        self.dropout = nn.Dropout(configs.dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        features = self.fc3(x)
        logits = self.fc4(features)
        return logits


class Predictor_cnn(nn.Module):
    """
    在 cycle 维度上做小型 temporal CNN，输入 x: [batch, cycles, feat]
    输出 logits: [batch, cycles, 1]
    保证 hidden 不小于输入通道数以减少信息丢失。
    """
    def __init__(self, configs, input_dim=None):
        super(Predictor_cnn, self).__init__()
        inp = input_dim if input_dim is not None else configs.cnn_output_channels_3
        base_hidden = getattr(configs, "hidden_channels", 64)
        hidden = max(inp, base_hidden)   # 保证容量
        mid = getattr(configs, "mlp_final_output", hidden)
        # 两层 1D 卷积：conv1 (inp -> hidden) -> conv2 (hidden -> mid)
        # 使用残差连接：把 conv1 的输出作为残差加到 conv2 的输出上
        self.conv1 = nn.Conv1d(in_channels=inp, out_channels=hidden, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=hidden, out_channels=mid, kernel_size=3, padding=1)
        # 当 residual 的通道数与 conv2 输出通道不匹配时，用 1x1 卷积投影
        if hidden == mid:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv1d(in_channels=hidden, out_channels=mid, kernel_size=1)
        self.conv_out = nn.Conv1d(in_channels=mid, out_channels=1, kernel_size=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(getattr(configs, "dropout", 0.0))

    def forward(self, x):
        # x: [batch, cycles, feat] -> [batch, feat, cycles]
        x = x.permute(0, 2, 1)
        # conv1 -> act -> dropout
        res = self.conv1(x)             # res: [batch, hidden, cycles]
        out = self.relu(res)
        out = self.dropout(out)
        # conv2
        out = self.conv2(out)          # out: [batch, mid, cycles]
        # 残差投影并相加
        res_proj = self.shortcut(res)  # [batch, mid, cycles] 或 Identity
        out = out + res_proj
        out = self.relu(out)
        logits = self.conv_out(out).permute(0, 2, 1)  # -> [batch, cycles, 1]
        return logits


class Predictor_lstm(nn.Module):
    """
    LSTM over cycles. 输入 x: [batch, cycles, feat]
    输出 logits: [batch, cycles, 1]
    """
    def __init__(self, configs, input_dim=None):
        super(Predictor_lstm, self).__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1) # 轻微 Dropout 防止过拟合
        )
        self.lstm = nn.LSTM(
            input_size=64,       # 承接降维后的维度
            hidden_size=128,      # 保持足够的记忆容量
            num_layers=2,         # 2层通常比1层能学到更复杂的退化模式
            batch_first=True,
            bidirectional=False,  # 【关键】如果是实时预测，必须为 False
            dropout=0.2           # 层间 Dropout
        )
        self.ln = nn.LayerNorm(128)

        # 4. 输出头 (Regression Head)
        self.fc_out = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)      # 输出 SOH 值
        )

    def forward(self, x):
        # x: [batch, cycles, feat]
        x = self.input_proj(x)    # -> [batch, cycles, 64]
        # Step 2: LSTM 时序提取
        # out: [batch, cycles, 128]
        # LSTM 会为每一个 time step 输出一个隐状态
        out, _ = self.lstm(x)     
        # Step 3: 归一化
        out = self.ln(out)
        
        # Step 4: 预测
        # 这里的 fc_out 会自动广播到每一个 cycle (TimeDistributed)
        logits = self.fc_out(out) # -> [batch, cycles, 1]
        return logits

class Predictor_transformer(nn.Module):
    """
    针对短序列 (cycles≈10) 的轻量化 Transformer predictor：
    - 先用小型 temporal Conv 提取局部信息
    - 小 d_model、少 layer、少 head，较高 dropout
    - encoder 后 residual + LayerNorm
    - 返回 logits: [b, seq, 1] 和 features: [b, seq, feat_out]
    """
    def __init__(self, configs, input_dim=None):
        super(Predictor_transformer, self).__init__()
        inp = input_dim if input_dim is not None else configs.cnn_output_channels_3

        # 更保守的默认超参，适合短序列
        d_model = getattr(configs, "transformer_d_model", 32)
        nhead = getattr(configs, "transformer_nhead", 1)
        num_layers = getattr(configs, "transformer_layers", 1)
        dim_feedforward = getattr(configs, "transformer_ff", max(d_model * 2, 64))
        dropout = getattr(configs, "dropout", 0.2)
        feat_out = getattr(configs, "mlp_final_output", 64)

        # 保证 d_model 可被 nhead 整除（必要时降 head）
        if nhead < 1:
            nhead = 1
        while nhead > 1 and (d_model % nhead != 0):
            nhead -= 1

        # 先做局部 temporal conv（保持通道数或小幅投影）
        proj_dim = min(max(16, inp), 64)
        self.conv_pre = nn.Conv1d(in_channels=inp, out_channels=proj_dim, kernel_size=3, padding=1)
        self.pre_relu = nn.ReLU()
        self.pre_drop = nn.Dropout(dropout)

        # 将 conv 输出投影到 d_model
        self.input_proj = nn.Linear(proj_dim, d_model)

        # Transformer Encoder (batch_first=True 在新 pytorch 可用)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='relu', batch_first=True
        )
        # 兼容不同 pytorch 版本
        try:
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self._use_old = False
        except TypeError:
            encoder_layer_old = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                           dim_feedforward=dim_feedforward,
                                                           dropout=dropout, activation='relu')
            self.encoder = nn.TransformerEncoder(encoder_layer_old, num_layers=num_layers)
            self._use_old = True

        # 可学习的位置编码（短序列可设置为 seq_len 默认为 32）
        max_cycles = getattr(configs, "max_cycles", 32)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_cycles, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        self.layernorm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # 输出层
        self.fc_feat = nn.Linear(d_model, feat_out)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(feat_out, 1)

    def forward(self, x):
        # x: [batch, cycles, feat]
        b, seq_len, _ = x.shape

        # conv_pre expects [batch, channels, seq]
        x_conv = x.permute(0, 2, 1)                 # [b, feat, seq]
        x_conv = self.conv_pre(x_conv)             # [b, proj_dim, seq]
        x_conv = self.pre_relu(x_conv)
        x_conv = self.pre_drop(x_conv)
        x_conv = x_conv.permute(0, 2, 1)           # [b, seq, proj_dim]

        # project to d_model and add positional embedding
        proj = self.input_proj(x_conv)             # [b, seq, d_model]
        pos = self.pos_emb[:, :seq_len, :].expand(b, -1, -1)
        proj = proj + pos
        proj = self.dropout(proj)

        # transformer encoder (兼容旧/new)
        if self._use_old:
            proj_t = proj.permute(1, 0, 2)         # [seq, b, d_model]
            enc = self.encoder(proj_t)
            enc = enc.permute(1, 0, 2)            # [b, seq, d_model]
        else:
            enc = self.encoder(proj)              # [b, seq, d_model]

        # residual + layernorm
        enc = self.layernorm(enc + proj)
        enc = self.dropout(enc)

        features = self.fc_feat(enc)
        features = self.relu(features)
        logits = self.fc_out(features)             # [b, seq, 1]
        return logits