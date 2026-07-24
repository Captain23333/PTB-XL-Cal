"""4-head model with pluggable 1D backbone.

Heads:
    head_super: 5-d sigmoid (NORM, MI, STTC, CD, HYP)
    head_sub:   23-d sigmoid (subdiagnostic classes; NORM has no sub)
    head_stmt:  71-d sigmoid (default per-statement prediction at inference)
    head_ord:   71 x 4 monotone CORN sigmoids (binary K-1 for likelihood ordering)

Backbones (backbone_name):
    "simplresnet1d"  – SimpleResNet1D, 4.08M params, default
    "inceptiontime"  – InceptionTime, 0.46M params (depth=6, nb_filters=32)
"""
import torch
import torch.nn as nn


def _build_xresnet1d101(c_in=12):
    return _SimpleResNet1D(c_in=c_in), 256


class _SimpleResNet1D(nn.Module):
    """Lightweight fallback 1D ResNet (used if fastai not available)."""
    def __init__(self, c_in=12, base=64, n_blocks=(2, 2, 2, 2)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(c_in, base, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base), nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        chans = [base, base*2, base*4, base*8]
        self.layers = nn.ModuleList()
        in_c = base
        for ci, n in zip(chans, n_blocks):
            blocks = []
            stride = 2 if ci != base else 1
            blocks.append(_ResBlock1D(in_c, ci, stride=stride))
            for _ in range(n - 1):
                blocks.append(_ResBlock1D(ci, ci))
            self.layers.append(nn.Sequential(*blocks))
            in_c = ci
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_c, 256)

    def forward(self, x):
        x = self.stem(x)
        for L in self.layers:
            x = L(x)
        x = self.pool(x); x = self.flatten(x); x = self.fc(x)
        return x


class _ResBlock1D(nn.Module):
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(c_in, c_out, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.skip = (nn.Sequential(nn.Conv1d(c_in, c_out, 1, stride=stride, bias=False),
                                   nn.BatchNorm1d(c_out))
                     if stride != 1 or c_in != c_out else nn.Identity())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.relu(y + self.skip(x))


class _InceptionBlock1D(nn.Module):
    def __init__(self, c_in, nb_filters=32, bottleneck=32, kernel_sizes=(39, 19, 9)):
        super().__init__()
        self.use_bottleneck = c_in > 1
        if self.use_bottleneck:
            self.bottleneck = nn.Conv1d(c_in, bottleneck, 1, bias=False)
            conv_in = bottleneck
        else:
            conv_in = c_in
        self.convs = nn.ModuleList([
            nn.Conv1d(conv_in, nb_filters, k, padding=k // 2, bias=False)
            for k in kernel_sizes
        ])
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(c_in, nb_filters, 1, bias=False),
        )
        self.bn = nn.BatchNorm1d(nb_filters * (len(kernel_sizes) + 1))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        inp = x
        if self.use_bottleneck:
            x = self.bottleneck(x)
        branches = [c(x) for c in self.convs] + [self.maxpool_conv(inp)]
        return self.relu(self.bn(torch.cat(branches, dim=1)))


class _InceptionTime1D(nn.Module):
    """InceptionTime backbone (Fawaz et al. 2020) for 1D multi-channel input.

    Input : [N, c_in, L]
    Output: [N, out_dim]  (global average pool → fc)
    """
    def __init__(self, c_in=12, nb_filters=32, bottleneck=32,
                 kernel_sizes=(39, 19, 9), depth=6, out_dim=128):
        super().__init__()
        c_block = nb_filters * (len(kernel_sizes) + 1)  # 32*4 = 128
        self.blocks = nn.ModuleList()
        self.residuals = nn.ModuleList()  # one entry per 3-block group, None for non-group-end
        in_c = c_in
        for i in range(depth):
            self.blocks.append(
                _InceptionBlock1D(in_c, nb_filters=nb_filters,
                                  bottleneck=bottleneck, kernel_sizes=kernel_sizes)
            )
            if i % 3 == 0:
                # start of group: record shortcut from group-start channels → c_block
                group_start_c = in_c
            if i % 3 == 2:
                # end of group: add shortcut
                self.residuals.append(
                    nn.Sequential(
                        nn.Conv1d(group_start_c, c_block, 1, bias=False),
                        nn.BatchNorm1d(c_block),
                    ) if group_start_c != c_block else nn.Sequential(nn.BatchNorm1d(c_block))
                )
            else:
                self.residuals.append(None)
            in_c = c_block
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(c_block, out_dim)

    def forward(self, x):
        residual = x
        for i, block in enumerate(self.blocks):
            if i % 3 == 0:
                residual = x  # reset shortcut at start of each 3-block group
            x = block(x)
            if self.residuals[i] is not None:
                x = self.relu(x + self.residuals[i](residual))
        x = self.pool(x)
        x = self.flatten(x)
        return self.fc(x)


class FourHead(nn.Module):
    def __init__(self, num_super=5, num_sub=23, num_stmt=71, num_ord_bins=4,
                 backbone_name="simplresnet1d"):
        super().__init__()
        if backbone_name == "inceptiontime":
            self.backbone = _InceptionTime1D(c_in=12, out_dim=128)
            d = 128
        else:
            self.backbone, d = _build_xresnet1d101(c_in=12)
        self.head_super = nn.Linear(d, num_super)
        self.head_sub = nn.Linear(d, num_sub)
        self.head_stmt = nn.Linear(d, num_stmt)
        self.head_ord = nn.Linear(d, num_stmt * num_ord_bins)
        self.num_stmt = num_stmt
        self.num_ord_bins = num_ord_bins

    def forward(self, x):
        f = self.backbone(x)
        return {
            "super_logits": self.head_super(f),
            "sub_logits": self.head_sub(f),
            "stmt_logits": self.head_stmt(f),
            "ord_logits": self.head_ord(f).view(-1, self.num_stmt, self.num_ord_bins),
            "feature": f,
        }


def build_model(num_super=5, num_sub=23, num_stmt=71, num_ord_bins=4,
                backbone_name="simplresnet1d"):
    return FourHead(num_super, num_sub, num_stmt, num_ord_bins,
                    backbone_name=backbone_name)


class SingleHead(nn.Module):
    """Single-head model for external datasets (e.g. Chapman 5-family labels)."""
    def __init__(self, n_classes=5, backbone_name="simplresnet1d"):
        super().__init__()
        if backbone_name == "inceptiontime":
            self.backbone = _InceptionTime1D(c_in=12, out_dim=128)
            d = 128
        else:
            self.backbone, d = _build_xresnet1d101(c_in=12)
        self.head = nn.Linear(d, n_classes)

    def forward(self, x):
        f = self.backbone(x)
        return {"logits": self.head(f)}


def build_chapman_model(n_classes=5, backbone_name="simplresnet1d"):
    return SingleHead(n_classes=n_classes, backbone_name=backbone_name)
