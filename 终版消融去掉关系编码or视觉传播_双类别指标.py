# 在 中文clip/混合数据局部交叉注意力.py（Chinese-CLIP ViT-B/16，样本级交叉注意力）基础上修正OT核：
# 关键修正：OTKernelProjection 不再把所有token/patch的投影求和后expand复制（那会抹平token差异），
# 而是保留每个文本token、每个图像patch自己的32维地标softmax分布（t_rkhs [B,Nt,32] / v_rkhs [B,Nv,32]）。
# 同时把Sinkhorn的reg从1.0调到0.1（代价已归一化到0~1，reg=1.0太平滑，reg=0.1运输矩阵明显尖锐）。
# 其余保持不变：图到文 + 拓扑 + 运输成本、LoRA、对比损失、分类器维度1057、训练超参、日志。
# 注意：此修正会改变模型实际行为，涉及OT/关系聚合的实验需用本版重新训练，旧结果不能与修正版混入同一消融表。
# 命令（微博真假混合）：
# nohup python -u "/home/xuxiaobao/fake_news_project/checkpoints/fake_news_code/中文clip/混合局部改OT核.py" --fusion_mode tot --dataset weibo --seed 123 --save_dir "./checkpoints/weibo_中文clip_局部交叉_改OT核_seed123" > run_中文clip_局部交叉_改OT核_weibo_seed123.log 2>&1 &
# 命令（MR2中文版）：
# nohup python -u "/home/xuxiaobao/fake_news_project/checkpoints/fake_news_code/中文clip/混合局部改OT核.py" --fusion_mode tot --dataset mr2 --mr2_language zh --seed 123 --save_dir "./checkpoints/mr2c_中文clip_局部交叉_改OT核_seed123" > run_中文clip_局部交叉_改OT核_mr2c_seed123.log 2>&1 &
#
# 本文件（_双类别指标.py）新增：同时输出 real(0) 与 fake(1) 两类的 Precision/Recall/F1/Support。
# 仅改动指标计算，模型结构、消融开关、训练、早停、保存最佳模型（仍以fake F1为准）逻辑均与源文件一致。
import os
import json
import math
import time
import random
import argparse
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import ot
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import get_cosine_schedule_with_warmup
import torchvision.transforms as transforms
import numpy as np

# Chinese-CLIP 模型目录（Hugging Face格式）
CHINESE_CLIP_PATH = "/home/xuxiaobao/data/chinese_clip_model"


# ===========================
# LoRA: 低秩适配CLIP
# ===========================
class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_a = nn.Parameter(torch.zeros(rank, in_dim))
        self.lora_b = nn.Parameter(torch.zeros(out_dim, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

    def forward(self, x):
        return (x @ self.lora_a.t() @ self.lora_b.t()) * self.scaling


class LoRALinear(nn.Module):
    def __init__(self, original_linear, rank=8, alpha=16):
        super().__init__()
        self.original = original_linear
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False
        self.lora = LoRALayer(original_linear.in_features, original_linear.out_features, rank, alpha)

    def forward(self, x):
        return self.original(x) + self.lora(x)


def apply_lora_to_chinese_clip(clip, rank=8, alpha=16):
    """对Chinese-CLIP文本和视觉编码器的Q/K/V/O注入LoRA。"""

    # 中文文本编码器：BERT结构
    for layer in clip.text_model.encoder.layer:
        layer.attention.self.query = LoRALinear(
            layer.attention.self.query, rank, alpha
        )
        layer.attention.self.key = LoRALinear(
            layer.attention.self.key, rank, alpha
        )
        layer.attention.self.value = LoRALinear(
            layer.attention.self.value, rank, alpha
        )
        layer.attention.output.dense = LoRALinear(
            layer.attention.output.dense, rank, alpha
        )

    # 视觉编码器：ViT结构
    for layer in clip.vision_model.encoder.layers:
        layer.self_attn.q_proj = LoRALinear(
            layer.self_attn.q_proj, rank, alpha
        )
        layer.self_attn.k_proj = LoRALinear(
            layer.self_attn.k_proj, rank, alpha
        )
        layer.self_attn.v_proj = LoRALinear(
            layer.self_attn.v_proj, rank, alpha
        )
        layer.self_attn.out_proj = LoRALinear(
            layer.self_attn.out_proj, rank, alpha
        )

    return clip


# ===========================
# TOT创新1: OT核投影（创建32个科学习地标点，把文本和图像token投影到这32个地标点，输出：每个token到地标的软分配权重分布。比喻：把100个单词用32个“主题词”来概括表达。）
# ===========================
class OTKernelProjection(nn.Module):
    """
    将每个文本 token 和图像 patch 投影到由可学习地标构成的关系空间。

    输入：
        text_tokens:  [B, Nt, 512]
        image_tokens: [B, Nv, 512]

    输出：
        t_rkhs: [B, Nt, 32]
        v_rkhs: [B, Nv, 32]
    """

    def __init__(self, dim=512, num_landmarks=32):
        super().__init__()

        # 32个可学习的跨模态关系地标
        self.landmarks = nn.Parameter(
            torch.randn(num_landmarks, dim)
        )

        # 控制软分配分布的集中程度
        self.scale = nn.Parameter(
            torch.tensor(2.0)
        )

    def forward(self, text_tokens, image_tokens):

        # 对地标和输入特征进行归一化，
        # 后续内积等价于余弦相似度
        landmarks = F.normalize(
            self.landmarks,
            p=2,
            dim=-1
        )

        text_tokens = F.normalize(
            text_tokens,
            p=2,
            dim=-1
        )

        image_tokens = F.normalize(
            image_tokens,
            p=2,
            dim=-1
        )

        # 每个文本token与32个地标之间的相似度
        # [B, Nt, 512] @ [512, 32]
        # -> [B, Nt, 32]
        text_logits = torch.matmul(
            text_tokens,
            landmarks.t()
        ) * self.scale

        # 每个图像patch与32个地标之间的相似度
        # [B, Nv, 512] @ [512, 32]
        # -> [B, Nv, 32]
        image_logits = torch.matmul(
            image_tokens,
            landmarks.t()
        ) * self.scale

        # 每个token/patch在32个地标上的软分配
        t_rkhs = F.softmax(
            text_logits,
            dim=-1
        )

        v_rkhs = F.softmax(
            image_logits,
            dim=-1
        )

        return t_rkhs, v_rkhs


# ===========================
# TOT创新2: 拓扑感知
# ===========================
class TopologyReasoning(nn.Module):
    # 消融开关（命令行 --no_rel_enc / --no_vis_prop 控制）：
    #   use_rel_enc : 保留 边关系编码分支 (edge_conv → t_edge_agg)
    #   use_vis_prop: 保留 视觉语义传播分支 (v_to_t = T_row @ V，图像→文本搬运)
    #  全量          : node_conv 输入 = [t_tokens; v_to_t; t_edge_agg]  → dim + dim + hidden
    #  w/o 关系编码  : 只剩 [t_tokens; v_to_t]                          → dim + dim
    #  w/o 视觉传播  : 只剩 [t_tokens; t_edge_agg]                      → dim + hidden
    def __init__(self, dim=512, hidden=128, use_rel_enc=True, use_vis_prop=True):
        super().__init__()    #将P矩阵中的数值（权重）编码为高维边特征
        self.use_rel_enc = use_rel_enc
        self.use_vis_prop = use_vis_prop
        if use_rel_enc:
            self.edge_conv = nn.Sequential(
                nn.Linear(1, hidden), nn.ReLU(), nn.Linear(hidden, hidden),
            )
        # 融合 文本语义(dim) [+ 视觉传播语义(dim)] [+ 拓扑边特征(hidden)]，生成具有拓扑感知的新文本节点表示
        node_in_dim = dim + (dim if use_vis_prop else 0) + (hidden if use_rel_enc else 0)
        self.node_conv = nn.Sequential(
            nn.Linear(node_in_dim, dim), nn.LayerNorm(dim), nn.ReLU(),
        )
        self.graph_readout = nn.Sequential(    #汇聚文本图和图像图整体信息，生成整个跨模态图全局拓扑表示
            nn.Linear(dim * 2, 128), nn.ReLU(), nn.Linear(128, 32),
        )
    ## 真正的"图到文"更新：只更新文本节点，但文本更新时真实使用了图像信息。
    ## V_t = T V：用行归一化后的OT矩阵，把相关图像patch语义搬运到每个文本token节点。
    ## T' = f_t([T; V_t; E_t])：融合 原始文本 + 搬运图像语义 + 拓扑边特征，更新文本节点。
    def forward(self, t_tokens, v_tokens, transport_T):
        Nt, D = t_tokens.shape
        # 行归一化：Sinkhorn输出的每行质量和约为 1/N_t，若直接 transport_T @ v_tokens，
        # 搬运后的图像特征数值会非常小；行归一化后才是每个文本token对相关图像patch的加权平均
        T_row = transport_T / (transport_T.sum(dim=1, keepdim=True) + 1e-8)

        node_inputs = [t_tokens]
        if self.use_vis_prop:
            v_to_t = torch.mm(T_row, v_tokens)  # V_t = T V：图像语义搬运到文本节点(视觉传播)
            node_inputs.append(v_to_t)
        if self.use_rel_enc:
            T_expanded = transport_T.unsqueeze(-1)  # 将OT变成边，原来是n×m，现在是n×m×1
            edge_feat = self.edge_conv(T_expanded)  # 对每条边进行编码，得到边特征（将原本只有一个权重值的边映射为高维向量，使边不仅表示"连接强弱"，还能学习更丰富的连接关系。）
            t_edge_agg = (T_row.unsqueeze(-1) * edge_feat).sum(dim=1) #与v_to_t保持同一节点尺度：用行归一化的T_row聚合每个文本Token关联的所有边特征，得到拓扑边特征 E_t (边关系编码)
            node_inputs.append(t_edge_agg)
        t_updated = self.node_conv(torch.cat(node_inputs, dim=-1))  # T' = f_t([T; V_t; E_t])，更新文本节点表示（图像节点不更新）
        # 汇聚所有节点得到图级表示：z_t = MeanPool(T'), z_v = MeanPool(V)
        t_graph = t_updated.mean(dim=0)
        v_graph = v_tokens.mean(dim=0)
        graph_feat = torch.cat([t_graph, v_graph], dim=0)
        out = self.graph_readout(graph_feat)  # z_topo = MLP([z_t; z_v])
        return out


# ===========================
# 数据预处理
# ===========================
def preprocess_data(text, image_path, processor):
    if not text or not isinstance(text, str) or text.strip() == "":
        text = "[NO_TEXT]"
    text_input = processor(text=text, return_tensors="pt", truncation=True, max_length=77)
    if image_path == "Null" or not image_path:
        place = r"/mnt/data/xuxiaobao/weibo/image_place.jpg"
        image = Image.open(place).convert("RGB").resize((224, 224))
    else:
        image = Image.open(image_path).convert("RGB").resize((224, 224))
    image_input = processor(images=image, return_tensors="pt")
    return text_input, image_input


# -------------------------------
# 数据集类
# -------------------------------
class RumorDataset(Dataset):
    def __init__(self, dataframe, processor):
        self.dataframe = dataframe
        self.processor = processor

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        text = row["clean_text"] if pd.notna(row["clean_text"]) else "[NO_TEXT]"
        image_path = row["img_path"] if pd.notna(row["img_path"]) and row["img_path"] != "Null" else None
        label = int(row["label"])
        return text, image_path, label


# -------------------------------
# MR2数据集类
# -------------------------------
class MR2Dataset(Dataset):
    # 各split中中文(MR2-C)和英文(MR2-E)的编号范围（按实际数据标定）
    LANGUAGE_RANGES = {
        "train": {"zh": (0, 6405), "en": (6406, 11183)},
        "val": {"zh": (0, 713), "en": (714, 1308)},
        "test": {"zh": (729, 1307), "en": (0, 728)},
    }

    def __init__(self, json_path, img_root, processor, split="train", language="all"):
        self.processor = processor
        self.img_root = img_root
        with open(json_path, "r") as f:
            raw = json.load(f)

        if language not in ["all", "zh", "en"]:
            raise ValueError(
                f"未知语言设置：{language}，"
                f"应为 all、zh 或 en"
            )

        self.samples = []

        for sample_key, sample in raw.items():
            sample_id = int(sample_key)
            label = int(sample["label"])

            # 只保留label 0和1，删除不可验证类别2
            if label == 2:
                continue

            # 根据编号范围筛选中文或英文
            if language != "all":
                start_id, end_id = self.LANGUAGE_RANGES[split][language]

                if not (start_id <= sample_id <= end_id):
                    continue

            self.samples.append({
                "sample_id": sample_id,
                "caption": sample["caption"],
                "image_path": sample["image_path"],
                "label": label,
            })

        language_name = {
            "all": "中英文混合",
            "zh": "中文MR2-C",
            "en": "英文MR2-E",
        }[language]

        print(
            f"[MR2Dataset] split={split} | "
            f"language={language_name} | "
            f"samples={len(self.samples)}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        text = (
            sample["caption"]
            if sample["caption"]
            else "[NO_TEXT]"
        )

        image_path = os.path.join(
            self.img_root,
            sample["image_path"]
        )

        label = sample["label"]

        return text, image_path, label


# -------------------------------
# Consistency
# -------------------------------
class Consistency(nn.Module):
    def __init__(self, dim=512, views=2, num_cls=2):
        super().__init__()
        self.views = views
        self.num_cls = num_cls
        self.cls_token = nn.Parameter(torch.zeros(1, num_cls, dim))
        self.QW = nn.Linear(dim, dim * views)
        self.FW = nn.ModuleList([nn.Linear(dim, dim) for _ in range(views)])
        self.FW2 = nn.ModuleList([nn.Linear(dim, dim) for _ in range(views)])
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, *datas):
        Qs = self.QW(self.cls_token).chunk(self.views, dim=-1)
        sims, Vs = [], []
        for i in range(self.views):
            Q = Qs[i]
            F1 = self.FW[i](datas[i])
            F2 = self.FW2[i](datas[i])
            sim = F.cosine_similarity(Q, F1.unsqueeze(1), dim=-1)
            sims.append(sim)
            Vs.append(F2.unsqueeze(1))
        sims = torch.stack(sims, dim=1)
        Vs = torch.stack(Vs, dim=1)
        fused = (sims.unsqueeze(-1) * Vs).sum(dim=1)
        return fused, sims


# -------------------------------
# StableCLIPModel (with LoRA + transport cost)
# -------------------------------
class StableCLIPModel(nn.Module):
    def __init__(self, num_classes=2, fusion_mode='original', lora_rank=8, rel_enc=True, vis_prop=True):
        super().__init__()
        self.num_classes = num_classes
        self.fusion_mode = fusion_mode
        self.clip = ChineseCLIPModel.from_pretrained(
            CHINESE_CLIP_PATH,
            local_files_only=True
        )

        # 启动检查：打印CLIP配置
        print(f"CLIP类型: {type(self.clip).__name__}")
        print(f"CLIP路径: {CHINESE_CLIP_PATH}")
        print(f"文本隐藏维度: {self.clip.config.text_config.hidden_size}")
        print(f"视觉隐藏维度: {self.clip.config.vision_config.hidden_size}")
        print(f"投影维度: {self.clip.config.projection_dim}")
        print(f"Patch size: {self.clip.config.vision_config.patch_size}")

        # 冻结所有CLIP参数
        for p in self.clip.parameters():
            p.requires_grad = False

        # LoRA: 解冻所有Attention层的低秩适配器
        if fusion_mode in ('tot', 'lora'):
            self.clip = apply_lora_to_chinese_clip(self.clip, rank=lora_rank, alpha=16)
            # LoRA参数默认requires_grad=True（因为nn.Parameter默认需要梯度）
            # 但仍需确保原始linear的bias可能已经require_grad=False
            # 这里不做额外操作，因为LoRALinear里已经设了requires_grad=False

            # LoRA注入完整性检查
            lora_linear_count = sum(
                1 for module in self.clip.modules()
                if isinstance(module, LoRALinear)
            )
            expected_lora_count = (
                len(self.clip.text_model.encoder.layer)
                + len(self.clip.vision_model.encoder.layers)
            ) * 4
            print(
                f"LoRALinear数量: {lora_linear_count}, "
                f"预期数量: {expected_lora_count}"
            )
            if lora_linear_count != expected_lora_count:
                raise RuntimeError(
                    f"LoRA注入不完整：实际{lora_linear_count}，"
                    f"预期{expected_lora_count}"
                )

        self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
        self.norm_text = nn.LayerNorm(512)
        self.norm_image = nn.LayerNorm(512)
        text_hidden_dim = self.clip.config.text_config.hidden_size
        vision_hidden_dim = self.clip.config.vision_config.hidden_size
        projection_dim = self.clip.config.projection_dim

        if projection_dim != 512:
            raise ValueError(
                f"Expected Chinese-CLIP projection_dim=512, got {projection_dim}"
            )

        self.image_proj = nn.Linear(vision_hidden_dim, 512)
        self.consistency = Consistency(dim=512, views=2, num_cls=num_classes)
        self.temperature = nn.Parameter(torch.tensor(0.07))

        # TOT级联模块
        if fusion_mode == 'tot':
            self.ot_kernel = OTKernelProjection(dim=512, num_landmarks=32)
            # 消融：--no_rel_enc 去掉关系编码(edge_conv边编码分支)；--no_vis_prop 去掉视觉传播(v_to_t搬运分支)
            self.topology = TopologyReasoning(dim=512, hidden=128,
                                              use_rel_enc=rel_enc, use_vis_prop=vis_prop)
            # 最终特征: Et(512) + Ev(512) + topo(32) + cost(1) = 1057
            cls_in_dim = 512 + 512 + 32 + 1
        elif fusion_mode == 'original':
            cls_in_dim = 2057
        else:  # lora模式（同original结构，但加了LoRA）
            cls_in_dim = 2057

        self.classifier = nn.Sequential(
            nn.Linear(cls_in_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, self.num_classes)
        )

    def forward(self, text_input, image_input, return_extra_loss=False):
        text_feat = self.clip.get_text_features(**text_input)
        image_feat = self.clip.get_image_features(**image_input)
        ht = F.normalize(text_feat, dim=-1)
        hv = F.normalize(image_feat, dim=-1)

        # token特征（提前提取，供样本级交叉注意力使用；Chinese-CLIP文本token 768维→text_projection映射到512，图像patch自动196个）
        text_model_inputs = {
            "input_ids": text_input["input_ids"],
            "attention_mask": text_input["attention_mask"],
        }
        if "token_type_ids" in text_input:
            text_model_inputs["token_type_ids"] = text_input["token_type_ids"]

        text_outputs = self.clip.text_model(**text_model_inputs)
        image_outputs = self.clip.vision_model(pixel_values=image_input["pixel_values"])

        text_tokens_raw = text_outputs.last_hidden_state[:, 1:, :]
        image_tokens_raw = image_outputs.last_hidden_state[:, 1:, :]

        text_tokens = self.clip.text_projection(text_tokens_raw)
        image_tokens = self.image_proj(image_tokens_raw)

        text_tokens = F.normalize(text_tokens, dim=-1)
        image_tokens = F.normalize(image_tokens, dim=-1)

        # 首次前向传播打印一次shape
        if not hasattr(self, "_shape_checked"):
            print(f"ht shape: {tuple(ht.shape)}")
            print(f"hv shape: {tuple(hv.shape)}")
            print(f"text_tokens shape: {tuple(text_tokens.shape)}")
            print(f"image_tokens shape: {tuple(image_tokens.shape)}")

            assert ht.size(-1) == 512
            assert hv.size(-1) == 512
            assert text_tokens.size(-1) == 512
            assert image_tokens.size(-1) == 512

            self._shape_checked = True

        # 文本padding mask：去掉[CLS]后与text_tokens逐位置对齐，[真实token|EOS]=1，[PAD]=0
        mask_t = text_input["attention_mask"][:, 1:]

        # 步骤1: 样本级（局部）交叉注意力：每样本独立，本样本文本token ↔ 本样本图像token，不跨batch。
        # 原版是 attention(ht, hv, hv)，ht.unsqueeze(0) 会把整个batch拼成一条序列，样本之间互相"偷看"；
        # 这里逐样本在token级做交叉注意力，mask加权池化后作为粗对齐特征，再与CLIP全局特征ht/hv相加做残差。
        Et_local_list, Ev_local_list = [], []
        for i in range(text_tokens.size(0)):
            t_i = text_tokens[i].unsqueeze(0)    # [1, Nt, 512]
            v_i = image_tokens[i].unsqueeze(0)   # [1, Nv, 512]
            htv, _ = self.attention(t_i, v_i, v_i)   # 文本token以本样本图像token为key/value
            hvt, _ = self.attention(v_i, t_i, t_i)   # 图像token以本样本文本token为key/value
            m_i = mask_t[i].float().unsqueeze(0).unsqueeze(-1)   # [1, Nt, 1]
            Et_local_list.append((((t_i + htv) * m_i).sum(dim=1) / m_i.sum().clamp(min=1)).squeeze(0))   # [512]
            Ev_local_list.append((v_i + hvt).mean(dim=1).squeeze(0))   # [512]
        Et_local = torch.stack(Et_local_list, dim=0)  # [B, 512]
        Ev_local = torch.stack(Ev_local_list, dim=0)  # [B, 512]
        Et = self.norm_text(ht + Et_local)
        Ev = self.norm_image(hv + Ev_local)
        Et = F.normalize(Et, p=2, dim=-1)
        Ev = F.normalize(Ev, p=2, dim=-1)

        # ============ TOT级联路径 ============
        if self.fusion_mode == 'tot':
            t_aug = text_tokens + Et.unsqueeze(1)
            v_aug = image_tokens + Ev.unsqueeze(1)
            t_aug = F.normalize(t_aug, dim=-1)
            v_aug = F.normalize(v_aug, dim=-1)

            topo_feats, cost_feats = [], []
            for i in range(t_aug.size(0)):
                # 只保留有效文本token（删除Padding）。
                # 必须在进入OT拓扑分支之前删除：仅把Padding置零不够，
                # OT仍会把全零节点计入文本边缘分布 a = 1/N_t。
                valid_mask = text_input["attention_mask"][i, 1:].bool()
                t_i = t_aug[i][valid_mask]
                v_i = v_aug[i]  # 图像patch数量由ViT-B/16动态得到（196个），无需去Padding

                # OT核投影（只针对有效文本节点）
                t_rkhs_i, v_rkhs_i = self.ot_kernel(t_i.unsqueeze(0), v_i.unsqueeze(0))

                cost = torch.cdist(t_rkhs_i[0].float(), v_rkhs_i[0].float(), p=2)
                cost = cost / (cost.max().detach() + 1e-8)

                a = torch.ones(cost.size(0), device=cost.device) / cost.size(0)
                b = torch.ones(cost.size(1), device=cost.device) / cost.size(1)
                T = ot.bregman.sinkhorn_log(a, b, cost, reg=0.1, numItermax=500, stopThr=1e-3)
                T = T.float()

                # 图到文更新与池化只作用于有效文本节点（t_updated.mean(dim=0)不含Padding）
                topo = self.topology(t_i, v_i, T)
                topo_feats.append(topo)

                # 运输成本作为显式不一致特征
                transport_cost = (T * cost).sum()
                cost_feats.append(transport_cost)

            topo_feats = torch.stack(topo_feats, dim=0)
            cost_feats = torch.stack(cost_feats, dim=0).unsqueeze(-1)  # [B, 1]

            # 显式地把transport cost喂给分类器
            final_feat = torch.cat([Et, Ev, topo_feats, cost_feats], dim=-1)  # [B, 1057]
            assert final_feat.size(-1) == 1057
            logits = self.classifier(final_feat)
            return logits, Et, Ev, topo_feats

        # ============ original / lora 路径（保持原版逻辑） ============
        local_text, local_image, ot_stats = [], [], []
        for i in range(text_tokens.size(0)):
            t_i = text_tokens[i][mask_t[i] > 0]
            v_i = image_tokens[i]
            cost = torch.cdist(t_i, v_i, p=2)
            cost = cost / (cost.max().detach() + 1e-8)
            a = torch.ones(t_i.size(0), device=t_i.device) / t_i.size(0)
            b = torch.ones(v_i.size(0), device=v_i.device) / v_i.size(0)
            T = ot.bregman.sinkhorn_log(a, b, cost, reg=0.1, numItermax=500, stopThr=1e-3)
            T = T.float()

            entropy_per_token = -(T * torch.log(T + 1e-8)).sum(dim=1)
            max_match_per_token = T.max(dim=1)[0]
            stats_i = torch.stack([
                entropy_per_token.mean(), entropy_per_token.max(), entropy_per_token.std(),
                max_match_per_token.mean(), max_match_per_token.min(), max_match_per_token.std(),
                (T * cost).sum(), T.view(-1).var(), (max_match_per_token < 0.1).float().mean()
            ])
            ot_stats.append(stats_i)

            v_align = torch.matmul(T, v_i)
            h_local, _ = self.attention(t_i.unsqueeze(0), v_align.unsqueeze(0), v_align.unsqueeze(0))
            t_local = (t_i + h_local.squeeze(0)).mean(dim=0)
            v_local = v_align.mean(dim=0)
            local_text.append(t_local)
            local_image.append(v_local)

        local_text = F.normalize(torch.stack(local_text, dim=0), dim=-1)
        local_image = F.normalize(torch.stack(local_image, dim=0), dim=-1)
        ot_stats = torch.stack(ot_stats, dim=0)

        fused, sims = self.consistency(Et, Ev)
        global_feat = torch.cat([Et, Ev], dim=-1)
        local_feat = torch.cat([local_text, local_image], dim=-1)
        final_feat = torch.cat([global_feat, local_feat, ot_stats], dim=-1)

        logits = self.classifier(final_feat)
        return logits, Et, Ev, ot_stats


# -------------------------------
# collate_fn
# -------------------------------
def get_collate_fn(processor, augment=False):
    transform = None
    if augment:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0))
        ])

    def collate(batch):
        texts, image_paths, labels = zip(*batch)
        text_inputs = processor(text=list(texts), return_tensors="pt", padding=True, truncation=True, max_length=77)
        images = []
        for path in image_paths:
            if path is None:
                img = Image.open(r"/mnt/data/xuxiaobao/weibo/image_place.jpg").convert("RGB")
            else:
                img = Image.open(path).convert("RGB")
            img = img.resize((224, 224))
            if transform:
                img = transform(img)
            images.append(img)
        image_inputs = processor(images=images, return_tensors="pt", padding=True)
        labels = torch.tensor(labels, dtype=torch.long)
        return text_inputs, image_inputs, labels

    return collate


# -------------------------------
# 统一指标计算：同时输出 real(0) 与 fake(1) 两类指标
# -------------------------------
def calculate_metrics(all_labels, all_preds):
    # labels=[0, 1] 保证第0项始终对应real，第1项对应fake
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels,
        all_preds,
        labels=[0, 1],
        average=None,
        zero_division=0
    )

    return {
        "acc": accuracy_score(all_labels, all_preds),

        "real_precision": precision[0],
        "real_recall": recall[0],
        "real_f1": f1[0],
        "real_support": int(support[0]),

        "fake_precision": precision[1],
        "fake_recall": recall[1],
        "fake_f1": f1[1],
        "fake_support": int(support[1]),
    }


# -------------------------------
# 训练与验证
# -------------------------------
def train_one_epoch(model, loader, optimizer, scheduler, loss_fn, device, lambda_cl, epoch):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for text_input, image_input, labels in loader:
        text_input = {k: v.to(device) for k, v in text_input.items()}
        image_input = {k: v.to(device) for k, v in image_input.items()}
        labels = labels.to(device)

        logits, Et, Ev, _ = model(text_input, image_input)
        ce_loss = loss_fn(logits, labels)

        sim_matrix = torch.mm(Et, Ev.t()) / model.temperature
        labels_cl = torch.arange(Et.size(0), device=device)
        Lcl = F.cross_entropy(sim_matrix, labels_cl)

        loss = ce_loss + lambda_cl * Lcl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    metrics = calculate_metrics(all_labels, all_preds)

    print(
        f"[Train] Epoch {epoch} | "
        f"Loss: {avg_loss:.4f} | Acc: {metrics['acc']:.4f}"
    )
    print(
        f"        Real(0) | "
        f"Prec: {metrics['real_precision']:.4f} | "
        f"Rec: {metrics['real_recall']:.4f} | "
        f"F1: {metrics['real_f1']:.4f} | "
        f"Support: {metrics['real_support']}"
    )
    print(
        f"        Fake(1) | "
        f"Prec: {metrics['fake_precision']:.4f} | "
        f"Rec: {metrics['fake_recall']:.4f} | "
        f"F1: {metrics['fake_f1']:.4f} | "
        f"Support: {metrics['fake_support']}"
    )

    # 保持main中的原有接收方式不变
    return avg_loss, metrics["acc"], metrics["fake_f1"]


def validate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for text_input, image_input, labels in loader:
            text_input = {k: v.to(device) for k, v in text_input.items()}
            image_input = {k: v.to(device) for k, v in image_input.items()}
            labels = labels.to(device)

            logits, _, _, _ = model(text_input, image_input)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    metrics = calculate_metrics(all_labels, all_preds)

    print(f"[Valid] Acc: {metrics['acc']:.4f}")
    print(
        f"        Real(0) | "
        f"Prec: {metrics['real_precision']:.4f} | "
        f"Rec: {metrics['real_recall']:.4f} | "
        f"F1: {metrics['real_f1']:.4f} | "
        f"Support: {metrics['real_support']}"
    )
    print(
        f"        Fake(1) | "
        f"Prec: {metrics['fake_precision']:.4f} | "
        f"Rec: {metrics['fake_recall']:.4f} | "
        f"F1: {metrics['fake_f1']:.4f} | "
        f"Support: {metrics['fake_support']}"
    )

    return metrics


# -------------------------------
# 日志记录
# -------------------------------
LOG_FILE = "experiment_log.json"

def save_log(args, best_val_f1, best_val_acc, best_val_prec, best_val_rec, epochs_run, total_trainable):
    log_entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "fusion_mode": args.fusion_mode,
        "lora_rank": args.lora_rank,
        "hyperparams": {
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "patience": args.patience,
            "lambda_cl": args.lambda_cl,
            "augment": args.augment,
        },
        "results": {
            "best_f1": round(best_val_f1, 4),
            "best_acc": round(best_val_acc, 4),
            "best_prec": round(best_val_prec, 4),
            "best_rec": round(best_val_rec, 4),
            "epochs_trained": epochs_run,
        },
        "model_info": {
            "trainable_params_k": round(total_trainable / 1024, 1),
        }
    }

    # 读取已有日志，追加，再写回
    logs = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r") as f:
                logs = json.load(f)
        except:
            logs = []

    logs.append(log_entry)
    # 按f1降序排序，方便比较
    logs.sort(key=lambda x: x["results"]["best_f1"], reverse=True)

    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)

    print(f"\n📋 日志已保存到 {LOG_FILE}")
    print(f"当前排名: 在 {len(logs)} 次实验中排名第 "
          f"{sum(1 for l in logs if l['results']['best_f1'] > best_val_f1) + 1}")


# -------------------------------
# 主函数
# -------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print(f"随机种子: {args.seed}")

    processor = ChineseCLIPProcessor.from_pretrained(
        CHINESE_CLIP_PATH,
        local_files_only=True
    )
    num_classes = 2
    model = StableCLIPModel(num_classes=num_classes, fusion_mode=args.fusion_mode, lora_rank=args.lora_rank,
                            rel_enc=not args.no_rel_enc, vis_prop=not args.no_vis_prop).to(device)

    # LoRA参数数量统计
    lora_params = sum(p.numel() for n, p in model.named_parameters() if 'lora' in n and p.requires_grad)
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LoRA参数量: {lora_params/1024:.1f}K | 总可训练: {total_trainable/1024:.1f}K")

    if args.dataset == "mr2":
        data_root = r"/home/xuxiaobao/data/MR2数据集"

        train_dataset = MR2Dataset(
            os.path.join(data_root, "dataset_items_train.json"),
            data_root, processor,
            split="train",
            language=args.mr2_language,
        )

        val_dataset = MR2Dataset(
            os.path.join(data_root, "dataset_items_val.json"),
            data_root, processor,
            split="val",
            language=args.mr2_language,
        )

        test_dataset = MR2Dataset(
            os.path.join(data_root, "dataset_items_test.json"),
            data_root, processor,
            split="test",
            language=args.mr2_language,
        )
    else:
        df_train = pd.concat([
            pd.read_csv(r"/mnt/data/xuxiaobao/weibo/train_rumor_data.csv"),
            pd.read_csv(r"/mnt/data/xuxiaobao/weibo/train_nonrumor_data.csv")
        ])

        df_val = pd.concat([
            pd.read_csv(r"/mnt/data/xuxiaobao/weibo/test_rumor_data.csv"),
            pd.read_csv(r"/mnt/data/xuxiaobao/weibo/test_nonrumor_data.csv")
        ])

        train_dataset = RumorDataset(df_train, processor)
        val_dataset = RumorDataset(df_val, processor)

    collate_fn = get_collate_fn(processor, augment=args.augment)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
    if args.dataset == "mr2":
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    # LoRA参数用稍大lr，其余用小lr
    if args.fusion_mode == 'tot':
        lora_params_list = [p for n, p in model.named_parameters() if 'lora' in n]
        other_params = [p for n, p in model.named_parameters() if 'lora' not in n and p.requires_grad]
        optimizer = optim.AdamW([
            {'params': other_params, 'lr': args.lr},
            {'params': lora_params_list, 'lr': args.lr * 2},
        ], weight_decay=args.weight_decay)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    num_training_steps = len(train_loader) * args.epochs
    num_warmup_steps = int(num_training_steps * 0.1)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    best_val_f1 = 0.0
    best_val_acc = 0.0
    best_val_prec = 0.0
    best_val_rec = 0.0
    patience_counter = 0
    best_model_path = os.path.join(args.save_dir, "best_model.pth")
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, optimizer, scheduler, loss_fn, device,
            args.lambda_cl, epoch
        )

        val_metrics = validate(model, val_loader, device)

        val_acc = val_metrics["acc"]
        val_prec = val_metrics["fake_precision"]
        val_rec = val_metrics["fake_recall"]
        val_f1 = val_metrics["fake_f1"]

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            best_val_prec = val_prec
            best_val_rec = val_rec
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"→ 保存最佳模型 (F1={best_val_f1:.4f}) 到 {best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"早停触发！连续 {args.patience} 个epoch无提升")
                break

    epochs_run = epoch

    # ===== 最终测试集评估 =====
    print("\n" + "="*50)
    print("加载最佳模型，在测试集上评估...")
    print("="*50)
    model.load_state_dict(torch.load(best_model_path))
    model.to(device)

    if args.dataset == "mr2":
        test_loader_to_use = test_loader
    else:
        test_loader_to_use = val_loader  # 微博没有独立测试集，用val（就是test）

    test_metrics = validate(model, test_loader_to_use, device)

    print(f"\n[最终测试] Overall Acc: {test_metrics['acc']:.4f}")
    print(
        f"[最终测试] Real(0) | "
        f"Prec: {test_metrics['real_precision']:.4f} | "
        f"Rec: {test_metrics['real_recall']:.4f} | "
        f"F1: {test_metrics['real_f1']:.4f} | "
        f"Support: {test_metrics['real_support']}"
    )
    print(
        f"[最终测试] Fake(1) | "
        f"Prec: {test_metrics['fake_precision']:.4f} | "
        f"Rec: {test_metrics['fake_recall']:.4f} | "
        f"F1: {test_metrics['fake_f1']:.4f} | "
        f"Support: {test_metrics['fake_support']}"
    )

    # 保存日志
    save_log(args, best_val_f1, best_val_acc, best_val_prec, best_val_rec, epochs_run, total_trainable)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP + LoRA + TOT")

    parser.add_argument("--batch_size", type=int, default=32, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-5, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--epochs", type=int, default=25, help="训练轮数")
    parser.add_argument("--patience", type=int, default=12, help="早停")
    parser.add_argument("--lambda_cl", type=float, default=0.1, help="对比损失权重（LoRA+tot下建议0.1）")
    parser.add_argument("--fusion_mode", type=str, default='tot',
                        choices=['original', 'tot'],
                        help="original=原版 | tot=级联TOT+LoRA+运输成本")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA秩大小")
    parser.add_argument("--augment", action="store_true", help="数据增强")
    parser.add_argument("--no_rel_enc", action="store_true",
                        help="消融：去掉 关系编码 (TopologyReasoning 的 edge_conv 边特征编码分支 → 仅保留视觉传播)")
    parser.add_argument("--no_vis_prop", action="store_true",
                        help="消融：去掉 视觉传播 (图像→文本语义搬运 v_to_t=T_row@V → 仅保留关系编码)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--dataset", type=str, default='weibo',
                        choices=['weibo', 'mr2'],
                        help="weibo=微博数据 | mr2=MR2数据")
    parser.add_argument(
        "--mr2_language",
        type=str,
        default="all",
        choices=["all", "zh", "en"],
        help=(
            "MR2语言子集："
            "all=中英文混合，"
            "zh=MR2-C中文，"
            "en=MR2-E英文"
        )
    )
    parser.add_argument("--save_dir", type=str, default="./checkpoints_3", help="模型保存目录")

    args = parser.parse_args()
    main(args)
