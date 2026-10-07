import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import AutoModel, DINOv3ViTConfig, DINOv3ViTModel, Dinov2Config, Dinov2Model

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mmbank
from models.mirror import (
    DualBranchClassifier,
    MIRROR_Detector,
    MirrorMemoryBank,
    load_detector_checkpoint,
    load_memory_checkpoint,
)


def offline_backbone(name, **kwargs):
    common = dict(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        image_size=32,
        patch_size=8,
    )
    if 'dinov3' in name:
        config = DINOv3ViTConfig(**common, num_register_tokens=4)
        config._attn_implementation = 'eager'
        return DINOv3ViTModel(config)
    config = Dinov2Config(**common)
    config._attn_implementation = 'eager'
    return Dinov2Model(config)


def detector(name='offline-dinov3'):
    with patch.object(AutoModel, 'from_pretrained', side_effect=offline_backbone):
        return MIRROR_Detector(name)


class MirrorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(42)

    def test_backbone_tokens_and_checkpoint_roundtrip(self):
        for name, tokens in [('offline-dinov3', 20), ('offline-dinov2', 16)]:
            with self.subTest(backbone=name):
                model = detector(name).eval()
                restored = detector(name).eval()
                load_detector_checkpoint(restored, {'model': model.state_dict()})
                images = torch.rand(2, 3, 32, 32)
                with torch.no_grad():
                    expected = model(images)
                    actual = restored(images)
                self.assertEqual(actual[2].shape, (2, tokens, 16))
                self.assertEqual(model.detector.head[0].in_features, 64 + 256 + 16)
                for left, right in zip(actual, expected):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_memory_freezes_weights_without_detaching_lora(self):
        model = detector().train()
        features, _ = model.backbone(model.norm(torch.rand(2, 3, 32, 32)))
        features.retain_grad()
        reconstruction, attention = model.memory_bank(features)
        loss = reconstruction.square().mean() - (attention * torch.log(attention + 1e-9)).sum(-1).mean()
        loss.backward()
        self.assertGreater(features.grad.abs().sum().item(), 0)
        gradients = [
            parameter.grad.abs().sum().item()
            for name, parameter in model.named_parameters()
            if 'lora_' in name and parameter.grad is not None
        ]
        self.assertGreater(max(gradients), 0)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.memory_bank.parameters()))

    def test_sparse_half_precision_forward_and_backward_are_finite(self):
        bank = MirrorMemoryBank(feature_dim=16, mem_slots=32, top_k=4).half()
        head = DualBranchClassifier(feat_dim=16, hidden_dim=32).half().eval()
        features = torch.randn(2, 7, 16, dtype=torch.float16, requires_grad=True)
        reconstruction, attention = bank(features)
        logits = head(attention, features, reconstruction, features.mean(1))
        self.assertTrue((attention == 0).any().item())
        for tensor in [reconstruction, attention, logits]:
            self.assertTrue(torch.isfinite(tensor).all().item())
        logits.float().square().mean().backward()
        self.assertTrue(torch.isfinite(features.grad).all().item())

    def test_checkpoint_formats_and_missing_keys(self):
        model = detector()
        state = {'module.' + key: value for key, value in model.state_dict().items()}
        for key in ['model', 'state_dict', 'model_state_dict', None]:
            with self.subTest(envelope=key):
                load_detector_checkpoint(model, {key: state} if key else state)
        incomplete = dict(model.state_dict())
        incomplete.pop('memory_bank.memory')
        with self.assertRaisesRegex(RuntimeError, 'memory_bank.memory'):
            load_detector_checkpoint(model, {'model': incomplete})

    def test_phase_one_uses_the_detector_memory_and_validates_config(self):
        first = mmbank.MirrorMemoryBank(feature_dim=16, mem_slots=32, top_k=4, num_heads=8)
        second = MirrorMemoryBank(feature_dim=16, mem_slots=32, top_k=4, num_heads=8)
        checkpoint = {
            'model_state_dict': first.state_dict(),
            'memory_config': {'feature_dim': 16, 'mem_slots': 32, 'top_k': 4, 'num_heads': 8},
        }
        load_memory_checkpoint(second, checkpoint)
        features = torch.randn(2, 7, 16)
        _, expected = first(features)
        actual, _ = second(features)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        checkpoint['memory_config']['num_heads'] = 4
        with self.assertRaisesRegex(ValueError, 'num_heads'):
            load_memory_checkpoint(second, checkpoint)


if __name__ == '__main__':
    unittest.main()
