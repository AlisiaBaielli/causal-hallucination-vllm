"""Diffusion-based visual noise for VCD (Visual Contrastive Decoding).

Adapted from Leng et al., "Mitigating Object Hallucinations in Large
Vision-Language Models through Visual Contrastive Decoding", CVPR 2024.
Reference: https://github.com/DAMO-NLP-SG/VCD
"""
import torch

def add_diffusion_noise(image_tensor, noise_step):
    num_steps = 1000

    betas = torch.linspace(-6,6,num_steps)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5

    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    noise = torch.randn_like(image_tensor)
    return alphas_bar_sqrt[noise_step] * image_tensor + one_minus_alphas_bar_sqrt[noise_step] * noise

