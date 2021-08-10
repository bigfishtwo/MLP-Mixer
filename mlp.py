import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
from functools import partial
from itertools import repeat
import collections.abc
import math
from typing import Any, Callable, Optional, Tuple

# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse
to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def named_apply(fn: Callable, module: nn.Module, name='', depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = '.'.join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MixerBlock(nn.Module):
    """ Residual Block w/ token mixing and channel MLPs
    Based on: 'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    def __init__(self, dim, seq_len, mlp_ratio=(0.5, 4.0), mlp_layer=Mlp,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, drop=0., drop_path=0.):
        super().__init__()
        tokens_dim, channels_dim = [int(x * dim) for x in to_2tuple(mlp_ratio)]
        self.norm1 = norm_layer(dim)
        self.mlp_tokens = mlp_layer(seq_len, tokens_dim, act_layer=act_layer, drop=drop)
        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp_channels = mlp_layer(dim, channels_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class MlpMixer(nn.Module):

    def __init__(
            self,
            num_classes=1000,
            img_size=224,
            in_chans=3,
            patch_size=16,
            num_blocks=8,
            embed_dim=512,
            mlp_ratio=(0.5, 4.0),
            block_layer=MixerBlock,
            mlp_layer=Mlp,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            drop_rate=0.,
            drop_path_rate=0.,
            nlhb=False,
            stem_norm=False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.stem = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dim, norm_layer=norm_layer if stem_norm else None)
        # FIXME drop_path (stochastic depth scaling rule or all the same?)
        self.blocks = nn.Sequential(*[
            block_layer(
                embed_dim, self.stem.num_patches, mlp_ratio, mlp_layer=mlp_layer, norm_layer=norm_layer,
                act_layer=act_layer, drop=drop_rate, drop_path=drop_path_rate)
            for _ in range(num_blocks)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # self.init_weights(nlhb=nlhb)

    # def init_weights(self, nlhb=False):
    #     head_bias = -math.log(self.num_classes) if nlhb else 0.
    #     named_apply(partial(_init_weights, head_bias=head_bias), module=self)  # depth-first

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

# def _init_weights(module: nn.Module, name: str, head_bias: float = 0., flax=False):
#     """ Mixer weight initialization (trying to match Flax defaults)
#     """
#     if isinstance(module, nn.Linear):
#         if name.startswith('head'):
#             nn.init.zeros_(module.weight)
#             nn.init.constant_(module.bias, head_bias)
#         else:
#             if flax:
#                 # Flax defaults
#                 lecun_normal_(module.weight)
#                 if module.bias is not None:
#                     nn.init.zeros_(module.bias)
#             else:
#                 # like MLP init in vit (my original init)
#                 nn.init.xavier_uniform_(module.weight)
#                 if module.bias is not None:
#                     if 'mlp' in name:
#                         nn.init.normal_(module.bias, std=1e-6)
#                     else:
#                         nn.init.zeros_(module.bias)
#     elif isinstance(module, nn.Conv2d):
#         lecun_normal_(module.weight)
#         if module.bias is not None:
#             nn.init.zeros_(module.bias)
#     elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
#         nn.init.ones_(module.weight)
#         nn.init.zeros_(module.bias)
#     elif hasattr(module, 'init_weights'):
#         # NOTE if a parent module contains init_weights method, it can override the init of the
#         # child modules as this will be called in depth-first order.
#         module.init_weights()




transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

training_set = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=transform
)

test_set = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=transform
)


train_loader = DataLoader(training_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MlpMixer(num_classes=10, img_size=32, in_chans=3).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

train_loss,train_acc = [], []
test_loss,test_acc = [], []


for epoch in range(10):
    print('Epoch:{}'.format(epoch))
    print('-' * 10)

    for phase in ['train','test']:
        running_loss = 0.0
        running_corr = 0.0
        count = 0

        if phase == 'train':
            model.train()
        else:
            model.eval()
        loader = train_loader if phase=='train' else test_loader
        for data,label in loader:
            data = data.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase=='train'):
                out = model(data)
                out = F.softmax(out, dim=1)
                _, pred = out.max(dim=1)
                loss = criterion(out, label)
                if phase=='train':
                    loss.backward()
                    optimizer.step()

        running_loss += loss.item()
        running_corr += torch.sum(pred == label.data).item()
        count += 1

        epoch_loss = running_loss / count
        epoch_acc = running_corr/len(loader)
        print('Epoch:{} Phase:{} Loss:{:.4f} Acc:{:.4f}'.format(epoch, phase, epoch_loss, epoch_acc))

        if phase=='train':
            train_loss.append(epoch_loss)
            train_acc.append(epoch_acc)

        else:
            test_loss.append(epoch_loss)
            test_acc.append(epoch_acc)

fig = plt.figure('loss')
plt.plot(train_loss, label='train loss')
plt.plot(test_loss, label='test loss')
plt.legend()

fig1 = plt.figure('acc')
plt.plot(train_acc, label='train acc')
plt.plot(test_acc, label='test acc')
plt.legend()
plt.show()