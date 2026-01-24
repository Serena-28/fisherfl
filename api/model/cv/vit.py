"""
Custom ViT variants for small-scale datasets (CIFAR-10, etc.)
"""
import torch.nn as nn
from torchvision.models.vision_transformer import VisionTransformer


def init_vit_zero_weights(model):
    """Initialize ViT layers that are incorrectly set to zero."""
    for name, param in model.named_parameters():
        if param.data.abs().sum() == 0 and param.requires_grad:
            if 'class_token' in name:
                nn.init.normal_(param.data, mean=0, std=0.02)
            elif 'head.weight' in name:
                nn.init.normal_(param.data, mean=0, std=0.01)


def vit_tiny(num_classes=10, image_size=32):
    """ViT-Tiny: ~5.4M params (similar to DeiT-Tiny)"""
    patch_size = 4 if image_size == 64 else 2  # 64/4=16, 32/2=16 patches
    model = VisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        num_layers=12,
        num_heads=3,
        hidden_dim=192,
        mlp_dim=768,
        num_classes=num_classes,
    )
    init_vit_zero_weights(model)
    return model


def vit_gpt(num_classes=10, image_size=32):
    patch_size = 4 if image_size == 64 else 2  # 64/4=16, 32/2=16 patches
    model = VisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        num_layers=6,
        num_heads=4,
        hidden_dim=192,
        mlp_dim=768,
        num_classes=num_classes,
    )
    init_vit_zero_weights(model)
    return model

def vit_gpt_tiny(num_classes=10, image_size=32):
    patch_size = 4 if image_size == 64 else 2  # 64/4=16, 32/2=16 patches
    model = VisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        num_layers=4,
        num_heads=3,
        hidden_dim=192,
        mlp_dim=576,
        num_classes=num_classes,
    )
    init_vit_zero_weights(model)
    return model

def vit_nano(num_classes=10, image_size=32):
    """ViT-Nano: ~1.2M params"""
    patch_size = 4 if image_size == 64 else 2  # 64/4=16, 32/2=16 patches
    model = VisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        num_layers=6,
        num_heads=2,
        hidden_dim=128,
        mlp_dim=512,
        num_classes=num_classes,
    )
    init_vit_zero_weights(model)
    return model


def vit_pico(num_classes=10, image_size=32):
    """ViT-Pico: ~0.46M params (similar to ShuffleNet)"""
    patch_size = 4 if image_size == 64 else 2  # 64/4=16, 32/2=16 patches
    model = VisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        num_layers=4,
        num_heads=2,
        hidden_dim=96,
        mlp_dim=384,
        num_classes=num_classes,
    )
    init_vit_zero_weights(model)
    return model


if __name__ == "__main__":
    import torch
    
    models = [
        ("vit_tiny", vit_tiny),
        ("vit_nano", vit_nano),
        ("vit_pico", vit_pico),
    ]
    
    print(f"{'Model':<15} {'Params':<15}")
    print("-" * 30)
    
    for name, fn in models:
        model = fn(num_classes=10)
        params = sum(p.numel() for p in model.parameters())
        print(f"{name:<15} {params:>10,} ({params/1e6:.2f}M)")
        
        # Test forward
        x = torch.randn(1, 3, 32, 32)
        y = model(x)
        assert y.shape == (1, 10), f"Output shape mismatch: {y.shape}"
    
    print("\nAll models OK!")





