import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Mapping
from transformers import AutoModel
from peft import get_peft_model, LoraConfig


class Norm(nn.Module):
    def __init__(self, mode='clip'):
        super().__init__()
        self.mode = mode
        if mode == 'clip':
            self.register_buffer('mean', torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))
        else:
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std

class DINO(nn.Module):
    def __init__(self, dinov3_path, finetune=True):
        super(DINO, self).__init__()
        print(f"Loading Backbone from: {dinov3_path}")
        self.dino = AutoModel.from_pretrained(dinov3_path, weights_only=False)
        self.dino.requires_grad_(False)


        model_type = getattr(self.dino.config, 'model_type', '')
        self.is_v3 = model_type.startswith('dinov3') or hasattr(self.dino, "layer") or "dinov3" in str(dinov3_path).lower()

        if finetune:
            self._apply_lora()

    def _apply_lora(self):

        target_modules = ["q_proj", "k_proj", "v_proj"] if self.is_v3 else ["query", "key", "value"]
        config = LoraConfig(r=8, lora_alpha=16, target_modules=target_modules)


        if hasattr(self.dino, 'layer'):
            encoder_layers = self.dino.layer
        elif hasattr(self.dino, 'encoder') and hasattr(self.dino.encoder, 'layer'):
            encoder_layers = self.dino.encoder.layer
        elif hasattr(self.dino, 'model') and hasattr(self.dino.model, 'layer'):
            encoder_layers = self.dino.model.layer
        else:
            raise ValueError('The backbone does not expose supported transformer layers for LoRA')
        for i in range(len(encoder_layers)):
            encoder_layers[i] = get_peft_model(encoder_layers[i], config)

    def forward(self, x):
        outputs = self.dino(pixel_values=x)
        last_hidden_state = outputs[0]


        feat_cls = last_hidden_state[:, 0]
        feat_tokens = last_hidden_state[:, 1:]
        return feat_tokens, feat_cls


class MirrorMemoryBank(nn.Module):

    def __init__(self, feature_dim, mem_slots=4096, num_heads=8, top_k=128):
        super().__init__()
        if num_heads < 1 or feature_dim % num_heads:
            raise ValueError('feature_dim must be divisible by a positive num_heads')
        if not 1 <= top_k <= mem_slots:
            raise ValueError('top_k must be between 1 and mem_slots')
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        self.head_dim = feature_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.top_k = top_k


        self.memory = nn.Parameter(torch.randn(mem_slots, feature_dim))


        self.q_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        self.k_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        self.v_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        self.out_proj = nn.Linear(feature_dim, feature_dim)

    def forward(self, x):
        B, N, C = x.shape
        M_slots, _ = self.memory.shape


        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)


        mem_ext = self.memory.unsqueeze(0).expand(B, -1, -1)
        k = self.k_proj(mem_ext).reshape(B, M_slots, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(mem_ext).reshape(B, M_slots, self.num_heads, self.head_dim).transpose(1, 2)


        attn_logits = (q @ k.transpose(-2, -1)) * self.scale


        topk_indices = torch.topk(attn_logits, k=self.top_k, dim=-1).indices


        mask = torch.zeros_like(attn_logits, dtype=torch.bool).scatter_(-1, topk_indices, True)
        masked_logits = attn_logits.masked_fill(~mask, float('-inf'))
        if masked_logits.dtype in (torch.float16, torch.bfloat16):
            masked_logits = masked_logits.float()
        sparse_attn = torch.softmax(masked_logits, dim=-1).to(v.dtype)


        recon = (sparse_attn @ v).transpose(1, 2).reshape(B, N, C)
        recon = self.out_proj(recon)


        return recon, sparse_attn


class DualBranchClassifier(nn.Module):
    def __init__(self, feat_dim, hidden_dim=512):
        super().__init__()


        self.perplexity_mlp = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )


        self.residual_mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 256)
        )


        self.head = nn.Sequential(
            nn.Linear(64 + 256 + feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, attn_weights, f_last, f_recon, feat_cls):


        metric_weights = attn_weights.float() if attn_weights.dtype in (torch.float16, torch.bfloat16) else attn_weights
        max_scores = metric_weights.max(dim=-1)[0].mean(dim=[1, 2]).unsqueeze(-1)


        entropy = -(metric_weights * torch.log(metric_weights + 1e-9)).sum(dim=-1).mean(dim=[1, 2]).unsqueeze(-1)


        evidence = torch.cat([max_scores, entropy], dim=-1).to(self.perplexity_mlp[0].weight.dtype)
        v_per = self.perplexity_mlp(evidence)


        residual_map = f_last - f_recon

        residual_gap = residual_map.mean(dim=1)


        v_res = self.residual_mlp(residual_gap)


        combined = torch.cat([v_per, v_res, feat_cls], dim=-1)

        return self.head(combined)


class MIRROR_Detector(nn.Module):
    def __init__(self, dino_path, memory_path=None, feature_dim=None):
        super(MIRROR_Detector, self).__init__()


        self.norm = Norm(mode='imagenet')


        self.backbone = DINO(dino_path, finetune=True)
        backbone_dim = self.backbone.dino.config.hidden_size
        if feature_dim is None:
            feature_dim = backbone_dim
        elif feature_dim != backbone_dim:
            raise ValueError(f'feature_dim={feature_dim} does not match backbone hidden_size={backbone_dim}')


        self.memory_bank = MirrorMemoryBank(feature_dim=feature_dim)

        if memory_path:
            print(f"Loading Memory Bank from {memory_path}")
            state_dict = torch.load(memory_path, map_location='cpu', weights_only=False)
            load_memory_checkpoint(self.memory_bank, state_dict)


        self.memory_bank.eval()
        for param in self.memory_bank.parameters():
            param.requires_grad = False


        self.detector = DualBranchClassifier(feat_dim=feature_dim)

    def forward(self, x):

        x = self.norm(x)


        feat_tokens, feat_cls = self.backbone(x)


        f_recon, attn_weights = self.memory_bank(feat_tokens)


        logits = self.detector(attn_weights, feat_tokens, f_recon, feat_cls)

        return logits, f_recon, feat_tokens

def build_mirror(memory_path=None, backbone_path='./weight/dinov3-huge'):

    model = MIRROR_Detector(dino_path=backbone_path, memory_path=memory_path)
    return model

def _checkpoint_state(checkpoint, keys):
    if not isinstance(checkpoint, Mapping):
        raise ValueError('Checkpoint must contain a state dictionary')
    state = checkpoint
    for key in keys:
        if isinstance(checkpoint.get(key), Mapping):
            state = checkpoint[key]
            break
    clean = {}
    for name, value in state.items():
        if not isinstance(name, str):
            raise ValueError('State dictionary keys must be strings')
        while name.startswith('module.'):
            name = name[len('module.'):]
        if name in clean:
            raise ValueError(f'Duplicate checkpoint parameter: {name}')
        clean[name] = value
    return clean

def load_memory_checkpoint(memory_bank, checkpoint):
    config = checkpoint.get('memory_config', {}) if isinstance(checkpoint, Mapping) else {}
    expected = {
        'feature_dim': memory_bank.feature_dim,
        'mem_slots': memory_bank.memory.shape[0],
        'num_heads': memory_bank.num_heads,
        'top_k': memory_bank.top_k,
    }
    for key, value in expected.items():
        if key in config and config[key] != value:
            raise ValueError(f'Memory checkpoint {key}={config[key]} does not match {value}')
    state = _checkpoint_state(checkpoint, ('model_state_dict', 'model', 'state_dict'))
    if any(name.startswith('memory_bank.') for name in state):
        state = {name[len('memory_bank.'):]: value for name, value in state.items() if name.startswith('memory_bank.')}
    return memory_bank.load_state_dict(state, strict=True)

def load_detector_checkpoint(model, checkpoint):
    state = _checkpoint_state(checkpoint, ('model', 'state_dict', 'model_state_dict'))
    expected = model.state_dict()
    remapped = {}
    for name, value in state.items():
        target = name
        if name not in expected:
            for source, destination in (
                ('backbone.dino.layer.', 'backbone.dino.model.layer.'),
                ('backbone.dino.model.layer.', 'backbone.dino.layer.'),
            ):
                if name.startswith(source):
                    candidate = destination + name[len(source):]
                    if candidate in expected:
                        target = candidate
                        break
        if target in remapped:
            raise ValueError(f'Duplicate checkpoint parameter: {target}')
        remapped[target] = value
    return model.load_state_dict(remapped, strict=True)


if __name__ == "__main__":


    pass
