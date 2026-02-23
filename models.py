"""
Model definitions for CIFAR-10 extreme compression pipeline.
- WideResNet-28-10 (Teacher)
- ResNet-20 (Student)
- BitNet b1.58 layers (BitConv2d, BitLinear)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =============================================================================
# WideResNet-28-10 (Teacher)
# =============================================================================

class WideBasicBlock(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, 3, padding=1, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)
        return out


class WideResNet(nn.Module):
    """WideResNet-28-10 for CIFAR-10."""

    def __init__(self, depth=28, widen_factor=10, dropout_rate=0.3, num_classes=10):
        super().__init__()
        assert (depth - 4) % 6 == 0, "Depth should be 6n+4"
        n = (depth - 4) // 6
        k = widen_factor

        nStages = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = nn.Conv2d(3, nStages[0], 3, stride=1, padding=1, bias=False)
        self.layer1 = self._wide_layer(WideBasicBlock, nStages[0], nStages[1], dropout_rate, n, stride=1)
        self.layer2 = self._wide_layer(WideBasicBlock, nStages[1], nStages[2], dropout_rate, n, stride=2)
        self.layer3 = self._wide_layer(WideBasicBlock, nStages[2], nStages[3], dropout_rate, n, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3])
        self.linear = nn.Linear(nStages[3], num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _wide_layer(self, block, in_planes, planes, dropout_rate, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(in_planes, planes, dropout_rate, s))
            in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# =============================================================================
# ResNet-20 (Student)
# =============================================================================

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet20(nn.Module):
    """ResNet-20 for CIFAR-10."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(BasicBlock, 16, 3, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 32, 3, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 64, 3, stride=2)
        self.linear = nn.Linear(64, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# =============================================================================
# BitNet b1.58 Layers
# =============================================================================

def round_clip(x, min_val, max_val):
    """Round and clip to {-1, 0, 1}."""
    return torch.clamp(torch.round(x), min_val, max_val)


def weight_quant_158(w):
    """BitNet b1.58 weight quantization: w -> {-1, 0, 1}."""
    scale = w.abs().mean() + 1e-5
    w_q = round_clip(w / scale, -1, 1)
    return w_q, scale


def activation_quant_8bit(x):
    """Absmax 8-bit activation quantization."""
    scale = x.abs().max() + 1e-5
    x_q = torch.clamp(x * 127.0 / scale, -128, 127).round()
    return x_q, scale


class BitConv2d(nn.Module):
    """1.58-bit Conv2d with STE for QAT."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, *self.kernel_size)
        )
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

    def forward(self, x):
        # Activation quantization (8-bit absmax)
        x_q, x_scale = activation_quant_8bit(x)

        # Weight quantization {-1, 0, 1}
        w_q, w_scale = weight_quant_158(self.weight)

        # STE: use quantized values in forward, pass gradients through
        w_ste = self.weight + (w_q * w_scale - self.weight).detach()
        x_ste = x + (x_q * (x_scale / 127.0) - x).detach()

        out = F.conv2d(x_ste, w_ste, self.bias,
                       stride=self.stride, padding=self.padding)
        return out


class BitLinear(nn.Module):
    """1.58-bit Linear with STE for QAT."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_normal_(self.weight)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    def forward(self, x):
        # Activation quantization (8-bit absmax)
        x_q, x_scale = activation_quant_8bit(x)

        # Weight quantization {-1, 0, 1}
        w_q, w_scale = weight_quant_158(self.weight)

        # STE
        w_ste = self.weight + (w_q * w_scale - self.weight).detach()
        x_ste = x + (x_q * (x_scale / 127.0) - x).detach()

        out = F.linear(x_ste, w_ste, self.bias)
        return out


# =============================================================================
# Utility: Convert standard model to BitNet model
# =============================================================================

def convert_to_bitnet(model):
    """Replace Conv2d and Linear layers with BitConv2d and BitLinear."""
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            bit_conv = BitConv2d(
                module.in_channels, module.out_channels,
                module.kernel_size[0],
                stride=module.stride[0],
                padding=module.padding[0],
                bias=module.bias is not None,
            )
            # Copy weights
            bit_conv.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                bit_conv.bias.data.copy_(module.bias.data)
            setattr(model, name, bit_conv)
        elif isinstance(module, nn.Linear):
            bit_linear = BitLinear(
                module.in_features, module.out_features,
                bias=module.bias is not None,
            )
            bit_linear.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                bit_linear.bias.data.copy_(module.bias.data)
            setattr(model, name, bit_linear)
        else:
            # Recurse into child modules
            convert_to_bitnet(module)
    return model
