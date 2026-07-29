#创新3去掉最优运输相关的
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
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import get_cosine_schedule_with_warmup
import torchvision.transforms as transforms
import numpy as np


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
        self.original.bias.requires_grad = False
        self.lora = LoRALayer(original_linear.in_features, original_linear.out_features, rank, alpha)

    def forward(self, x):
        return self.original(x) + self.lora(x)


def apply_lora_to_clip(clip, rank=8, alpha=16):
    """给CLIP的所有Self-Attention的Q/K/V/O加上LoRA"""
    for layer in clip.text_model.encoder.layers:
        layer.self_attn.q_proj = LoRALinear(layer.self_attn.q_proj, rank, alpha)
        layer.self_attn.k_proj = LoRALinear(layer.self_attn.k_proj, rank, alpha)
        layer.self_attn.v_proj = LoRALinear(layer.self_attn.v_proj, rank, alpha)
        layer.self_attn.out_proj = LoRALinear(layer.self_attn.out_proj, rank, alpha)

    for layer in clip.vision_model.encoder.layers:
        layer.self_attn.q_proj = LoRALinear(layer.self_attn.q_proj, rank, alpha)
        layer.self_attn.k_proj = LoRALinear(layer.self_attn.k_proj, rank, alpha)
        layer.self_attn.v_proj = LoRALinear(layer.self_attn.v_proj, rank, alpha)
        layer.self_attn.out_proj = LoRALinear(layer.self_attn.out_proj, rank, alpha)

    return clip


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
    def __init__(self, json_path, img_root, processor):
        self.processor = processor
        self.img_root = img_root
        with open(json_path, "r") as f:
            raw = json.load(f)
        # 只保留label 0和1，去掉label 2（无法验证）
        self.samples = []
        for k, v in raw.items():
            label = int(v["label"])
            if label == 2:
                continue
            self.samples.append({
                "caption": v["caption"],
                "image_path": v["image_path"],
                "label": label,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample["caption"] if sample["caption"] else "[NO_TEXT]"
        image_path = os.path.join(self.img_root, sample["image_path"])
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
    def __init__(self, num_classes=2, fusion_mode='original', lora_rank=8):
        super().__init__()
        self.num_classes = num_classes
        self.fusion_mode = fusion_mode
        self.clip = CLIPModel.from_pretrained(
            "/home/xuxiaobao/data/clip_model",
            local_files_only=True
        )

        # 冻结所有CLIP参数
        for p in self.clip.parameters():
            p.requires_grad = False

        # LoRA: 解冻所有Attention层的低秩适配器
        if fusion_mode in ('tot', 'lora'):
            self.clip = apply_lora_to_clip(self.clip, rank=lora_rank, alpha=16)
            # LoRA参数默认requires_grad=True（因为nn.Parameter默认需要梯度）
            # 但仍需确保原始linear的bias可能已经require_grad=False
            # 这里不做额外操作，因为LoRALinear里已经设了requires_grad=False

        self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
        self.norm_text = nn.LayerNorm(512)
        self.norm_image = nn.LayerNorm(512)
        self.image_proj = nn.Linear(768, 512)
        self.consistency = Consistency(dim=512, views=2, num_cls=num_classes)
        self.temperature = nn.Parameter(torch.tensor(0.07))

        # TOT级联模块（无OT核、无最优运输、无拓扑感知）
        if fusion_mode == 'tot':
            # 最终特征: Et(512) + Ev(512) = 1024
            cls_in_dim = 512 + 512
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

        # 步骤1: 交叉注意力粗对齐
        htv, _ = self.attention(ht.unsqueeze(0), hv.unsqueeze(0), hv.unsqueeze(0))
        hvt, _ = self.attention(hv.unsqueeze(0), ht.unsqueeze(0), ht.unsqueeze(0))
        Et = self.norm_text(ht + htv.squeeze(0))
        Ev = self.norm_image(hv + hvt.squeeze(0))
        Et = F.normalize(Et, p=2, dim=-1)
        Ev = F.normalize(Ev, p=2, dim=-1)

        # token特征
        text_outputs = self.clip.text_model(input_ids=text_input["input_ids"], attention_mask=text_input["attention_mask"])
        image_outputs = self.clip.vision_model(pixel_values=image_input["pixel_values"])
        text_tokens = text_outputs.last_hidden_state[:, 1:, :]
        image_tokens = image_outputs.last_hidden_state[:, 1:, :]
        image_tokens = self.image_proj(image_tokens)
        text_tokens = F.normalize(text_tokens, dim=-1)
        image_tokens = F.normalize(image_tokens, dim=-1)

        # ============ TOT级联路径（无OT，直接Et+Ev分类）============
        if self.fusion_mode == 'tot':
            final_feat = torch.cat([Et, Ev], dim=-1)  # [B, 1024]
            logits = self.classifier(final_feat)
            return logits, Et, Ev, Et

        # ============ original / lora 路径（保持原版逻辑） ============
        local_text, local_image, ot_stats = [], [], []
        for i in range(text_tokens.size(0)):
            t_i = text_tokens[i]
            v_i = image_tokens[i]
            cost = torch.cdist(t_i, v_i, p=2)
            cost = cost / (cost.max().detach() + 1e-8)
            a = torch.ones(t_i.size(0), device=t_i.device) / t_i.size(0)
            b = torch.ones(v_i.size(0), device=v_i.device) / v_i.size(0)
            T = ot.bregman.sinkhorn_log(a, b, cost, reg=1.0, numItermax=500, stopThr=1e-3)
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
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)

    print(f"[Train] Epoch {epoch} | Loss: {avg_loss:.4f} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")
    return avg_loss, acc, f1


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

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)

    print(f"[Valid] Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")
    return acc, prec, rec, f1


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

    processor = CLIPProcessor.from_pretrained(
        "/home/xuxiaobao/data/clip_model",
        local_files_only=True
    )
    num_classes = 2
    model = StableCLIPModel(num_classes=num_classes, fusion_mode=args.fusion_mode, lora_rank=args.lora_rank).to(device)

    # LoRA参数数量统计
    lora_params = sum(p.numel() for n, p in model.named_parameters() if 'lora' in n and p.requires_grad)
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LoRA参数量: {lora_params/1024:.1f}K | 总可训练: {total_trainable/1024:.1f}K")

    if args.dataset == "mr2":
        data_root = r"/home/xuxiaobao/data/MR2数据集"
        train_dataset = MR2Dataset(
            os.path.join(data_root, "dataset_items_train.json"),
            data_root, processor
        )
        val_dataset = MR2Dataset(
            os.path.join(data_root, "dataset_items_val.json"),
            data_root, processor
        )
        test_dataset = MR2Dataset(
            os.path.join(data_root, "dataset_items_test.json"),
            data_root, processor
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

        val_acc, val_prec, val_rec, val_f1 = validate(model, val_loader, device)

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

    test_acc, test_prec, test_rec, test_f1 = validate(model, test_loader_to_use, device)
    print(f"\n[最终测试] Acc: {test_acc:.4f} | Prec: {test_prec:.4f} | Rec: {test_rec:.4f} | F1: {test_f1:.4f}")

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
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--dataset", type=str, default='weibo',
                        choices=['weibo', 'mr2'],
                        help="weibo=微博数据 | mr2=MR2数据")
    parser.add_argument("--save_dir", type=str, default="./checkpoints_3", help="模型保存目录")

    args = parser.parse_args()
    main(args)