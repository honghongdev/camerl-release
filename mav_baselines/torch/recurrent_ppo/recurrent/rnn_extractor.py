from typing import Dict, List, Tuple, Type, Union

import gym
import torch as th
import torch
import torch.nn.functional as F
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.type_aliases import TensorDict

class OverlapPatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size, stride, padding):
        super().__init__()
        self.cn1 = nn.Conv2d(in_channels, out_channels, kernel_size=patch_size, stride = stride, padding = padding)
        self.layerNorm = nn.LayerNorm(out_channels)

    def forward(self, patches):
        x = self.cn1(patches)
        _,_,H, W = x.shape
        x = x.flatten(2).transpose(1,2) #Flatten - (B,C,H*W); transpose B,HW, C
        x = self.layerNorm(x)
        return x,H,W #B, N, EmbedDim
    
class EfficientSelfAttention(nn.Module):
    def __init__(self, channels, reduction_ratio, num_heads):
        super().__init__()
        assert channels % num_heads == 0, f"channels {channels} should be divided by num_heads {num_heads}."

        self.heads= num_heads

        #### Self Attention Block consists of 2 parts - Reduction and then normal Attention equation of queries and keys###
        
        # Reduction Parameters #
        self.cn1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=reduction_ratio, stride= reduction_ratio)
        self.ln1 = nn.LayerNorm(channels)
        # Attention Parameters #
        self.keyValueExtractor = nn.Linear(channels, channels * 2)
        self.query = nn.Linear(channels, channels)
        self.smax = nn.Softmax(dim=-1)
        self.finalLayer = nn.Linear(channels, channels) 
    def forward(self, x, H, W):
        B,N,C = x.shape
        # B, N, C -> B, C, N
        x1 = x.clone().permute(0,2,1)
        # BCN -> BCHW
        x1 = x1.reshape(B,C,H,W)
        x1 = self.cn1(x1)
        x1 = x1.reshape(B,C,-1).permute(0,2,1).contiguous()
        x1 = self.ln1(x1)
        # We have got the Reduced Embeddings! We need to extract key and value pairs now
        keyVal = self.keyValueExtractor(x1)
        keyVal = keyVal.reshape(B, -1 , 2, self.heads, int(C/self.heads)).permute(2,0,3,1,4).contiguous()
        k,v = keyVal[0],keyVal[1] #b,heads, n, c/heads
        q = self.query(x).reshape(B, N, self.heads, int(C/self.heads)).permute(0, 2, 1, 3).contiguous()

        dimHead = (C/self.heads)**0.5
        attention = self.smax(q@k.transpose(-2, -1)/dimHead)
        attention = (attention@v).transpose(1,2).reshape(B,N,C)

        x = self.finalLayer(attention) #B,N,C        
        return x

class MixFFN(nn.Module):
    def __init__(self, channels, expansion_factor):
        super().__init__()
        expanded_channels = channels*expansion_factor
        #MLP Layer        
        self.mlp1 = nn.Linear(channels, expanded_channels)
        #Depth Wise CNN Layer
        self.depthwise = nn.Conv2d(expanded_channels, expanded_channels, kernel_size=3,  padding='same', groups=channels)
        #GELU
        self.gelu = nn.GELU()
        #MLP to predict
        self.mlp2 = nn.Linear(expanded_channels, channels)

    def forward(self, x, H, W):
        # Input BNC instead of BCHW
        # BNC -> B,N,C*exp 
        x = self.mlp1(x)
        B,N,C = x.shape
        # Prepare for the CNN operation, channel should be 1st dim
        # B,N, C*exp -> B, C*exp, H, W 
        x = x.transpose(1,2).view(B,C,H,W)

        #Depth Conv - B, N, Cexp 
        x = self.gelu(self.depthwise(x).flatten(2).transpose(1,2))

        #Back to the orignal shape
        x = self.mlp2(x) # BNC
        return x

# class MixTransformerEncoderLayer(nn.Module):
#     def __init__(self, in_channels, out_channels, patch_size, stride, padding, 
#                  n_layers, reduction_ratio, num_heads, expansion_factor):
#         super().__init__()
#         self.patchMerge = OverlapPatchMerging(in_channels, out_channels, patch_size, stride, padding) # B N embed dim
#         #You might be wondering why I didn't used a cleaner implementation but the input to each forward function is different
#         self._attn = nn.ModuleList([EfficientSelfAttention(out_channels, reduction_ratio, num_heads) for _ in range(n_layers)])
#         self._ffn = nn.ModuleList([MixFFN(out_channels,expansion_factor) for _ in range(n_layers)])
#         self._lNorm = nn.ModuleList([nn.LayerNorm(out_channels) for _ in range(n_layers)])

#     def forward(self, x):
#         B,C,H,W = x.shape
#         for i in range(len(self._attn)):
#             x = x + self._attn[i].forward(x, H, W) #BNC
#             x = x + self._ffn[i].forward(x, H, W) #BNC
#             x = self._lNorm[i].forward(x) #BNC
#         x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() #BCHW
#         return x

class MixTransformerEncoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size, stride, padding, 
                 n_layers, reduction_ratio, num_heads, expansion_factor):
        super().__init__()
        self.patchMerge = OverlapPatchMerging(in_channels, out_channels, patch_size, stride, padding) # B N embed dim
        #You might be wondering why I didn't used a cleaner implementation but the input to each forward function is different
        self._attn = nn.ModuleList([EfficientSelfAttention(out_channels, reduction_ratio, num_heads) for _ in range(n_layers)])
        self._ffn = nn.ModuleList([MixFFN(out_channels,expansion_factor) for _ in range(n_layers)])
        self._ln_attn = nn.ModuleList([nn.LayerNorm(out_channels) for _ in range(n_layers)])
        self._ln_ffn  = nn.ModuleList([nn.LayerNorm(out_channels) for _ in range(n_layers)])

    def forward(self, x):
        B,C,H,W = x.shape
        x,H,W = self.patchMerge(x)      # (B,N,C)
        for i in range(len(self._attn)):
            x = x + self._attn[i]( self._ln_attn[i](x), H, W )
            x = x + self._ffn[i](  self._ln_ffn[i](x),  H, W )
        x = x.reshape(B, H, W, -1).permute(0,3,1,2).contiguous()
        return x
  
class ViTBackbone(nn.Module):
    def __init__(self, img_ch=1):
        super().__init__()
        self.stage1 = MixTransformerEncoderLayer(
            in_channels=img_ch, out_channels=32,
            patch_size=7, stride=4, padding=3,
            n_layers=2, reduction_ratio=8, num_heads=1, expansion_factor=8
        )
        self.stage2 = MixTransformerEncoderLayer(
            in_channels=32, out_channels=64,
            patch_size=3, stride=2, padding=1,
            n_layers=2, reduction_ratio=4, num_heads=2, expansion_factor=8
        )
        self.px = nn.PixelShuffle(2)          # (64,H2,W2)->(16,2H2,2W2)
        self.fuse = nn.Conv2d(48, 12, 3, padding=1)
        self.head = nn.Linear(12 * 64 * 64, 512)

    def forward(self, x):                     # x: (B, 1, 256, 256)
        y1 = self.stage1(x)                   # (B, 32, 64, 64)          
        y2 = self.stage2(y1)    # (B, 64, 32, 32)

        y2 = self.px(y2)                      # (B, 64, 32, 32) → (B, 16, 64, 64)
        y  = torch.cat([y2, y1], dim=1)       # (B,48,64,64)
        y  = self.fuse(y)                     # (B,12,64,64)

        B, C, Hf, Wf = y.shape
        flat = y.flatten(1)                   # (B,12*64*64)
        h = self.head(flat)                   # (B,12*64*64) --> (B,512)
        return h

class ViTEncoder(nn.Module):
    def __init__(self, img_channels, latent_size):
        super().__init__()
        self.backbone = ViTBackbone(img_ch=img_channels)
        self.fc_mu = nn.Linear(512, latent_size)
        self.fc_logsigma = nn.Linear(512, latent_size)

    def forward(self, x):
        h = self.backbone(x)                  # (B,512)
        mu = self.fc_mu(h)                    # (B,Z)
        logsigma = self.fc_logsigma(h)        # (B,Z)
        logsigma = torch.clamp(logsigma, min=-2.0, max=2.0)
        return mu, logsigma

# class Encoder(BaseFeaturesExtractor):
#     def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 64):

#         image_observation_space = None
#         for key, subspace in observation_space.spaces.items():
#             if is_image_space(subspace):
#                 image_observation_space = subspace
#         n_input_channels = image_observation_space.shape[0]
#         self.backbone = ViTBackbone(img_ch=n_input_channels)
#         self.fc_mu = nn.Linear(512, features_dim)
#         self.fc_logsigma = nn.Linear(512, features_dim)

#     def forward(self, observations: th.Tensor) -> th.Tensor:
#         h = self.backbone(observations)
#         mu = self.fc_mu(h)
#         logsigma = self.fc_logsigma(h)
#         return mu
    
class Encoder(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 64):

        image_observation_space = None
        for key, subspace in observation_space.spaces.items():
            if is_image_space(subspace):
                image_observation_space = subspace
        super(Encoder, self).__init__(image_observation_space, features_dim)
        n_input_channels = image_observation_space.shape[0]
        self.conv1 = nn.Conv2d(n_input_channels, 8, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=4, stride=2)
        self.conv6 = nn.Conv2d(128, 256, kernel_size=4, stride=2)
        # Compute shape by doing one forward pass
        # with th.no_grad():
        #     n_flatten = self.conv6(self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(th.as_tensor(image_observation_space.sample()[None][:, :1, :, :]).float())))))).shape
        # self.linear = nn.Linear(2*2*256, features_dim)
        self.fc_mu = nn.Linear(2*2*256, features_dim)
        self.fc_logsigma = nn.Linear(2*2*256, features_dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = nn.functional.relu(self.conv1(observations))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.relu(self.conv4(x))
        x = nn.functional.relu(self.conv5(x))
        x = nn.functional.relu(self.conv6(x))
        mu = self.fc_mu(x.view(observations.size(0), -1))
        # logsigma = self.fc_logsigma(x.view(observations.size(0), -1))
        # sigma = logsigma.exp()
        # eps = th.randn_like(sigma)
        # z = eps.mul(sigma).add_(mu)
        return mu
    
class Decoder(nn.Module):
    def __init__(self, observation_space: gym.spaces.Box, lstm_hidden_dim: int = 64) -> None:
        super(Decoder, self).__init__()
        for key, subspace in observation_space.spaces.items():
            if is_image_space(subspace):
                image_observation_space = subspace
        n_input_channels = image_observation_space.shape[0]
        self.fc = nn.Linear(lstm_hidden_dim, 2*2*256)
        self.deconv1 = nn.ConvTranspose2d(2*2*256, 128, kernel_size=5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2)
        self.deconv4 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2)
        self.deconv5 = nn.ConvTranspose2d(16, 8, kernel_size=5, stride=2)
        self.deconv6 = nn.ConvTranspose2d(8, n_input_channels, kernel_size=4, stride=2)

    def forward(self, latent)-> th.Tensor:
        x = nn.functional.relu(self.fc(latent))
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = nn.functional.relu(self.deconv1(x))
        x = nn.functional.relu(self.deconv2(x))
        x = nn.functional.relu(self.deconv3(x))
        x = nn.functional.relu(self.deconv4(x))
        x = nn.functional.relu(self.deconv5(x))
        reconstruction = th.sigmoid(self.deconv6(x))
        return reconstruction

class MultiExtractor(BaseFeaturesExtractor):
    """
    Combined feature extractor for Dict observation spaces.
    Builds a feature extractor for each key of the space. Input from each space
    is fed through a separate submodule (CNN or MLP, depending on input shape),
    the output features are concatenated and fed through additional MLP network ("combined").

    :param observation_space:
    :param cnn_output_dim: Number of features to output from each CNN submodule(s). Defaults to
        256 to avoid exploding network sizes.
    """

    def __init__(self, observation_space: gym.spaces.Dict, cnn_output_dim: int = 64):
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super(MultiExtractor, self).__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if is_image_space(subspace):
                extractors[key] = Encoder(subspace, features_dim=cnn_output_dim)
                total_concat_size += cnn_output_dim
                continue
            else:
                # The observation key is a vector, flatten it if needed
                extractors[key] = nn.Flatten()
                total_concat_size += 6

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations: TensorDict) -> th.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return th.cat(encoded_tensor_list, dim=1)
