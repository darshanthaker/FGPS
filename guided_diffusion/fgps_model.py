import torch
import torch.nn as nn
from torch.fft import fftn, ifftn, fftshift, ifftshift

class FGPSModel(nn.Module):
        def __init__(self, device):
            super(FGPSModel, self).__init__()
            self.device = device

        def get_mask(self, x_shape, cutoff):
            N, C, h, w = x_shape 
            y, x = torch.meshgrid(torch.arange(-h//2, h//2), torch.arange(-w//2, w//2), indexing='ij')
            distance = torch.sqrt(x**2 + y**2)
            mask = (distance < cutoff).float()
            mask = mask.unsqueeze(0).unsqueeze(0).repeat(1, C, 1, 1).to(self.device)
            return mask

        def forward(self, x, mask):
            channel_fft = fftn(x, dim=(-2, -1),norm='ortho')
            channel_fft_shifted = fftshift(channel_fft, dim=(-2, -1))
        
            mask = self.get_mask(x.shape, mask)
            channel_fft_shifted_filtered = channel_fft_shifted * mask
            channel_fft_filtered = ifftshift(channel_fft_shifted_filtered, dim=(-2, -1))
            return channel_fft_filtered