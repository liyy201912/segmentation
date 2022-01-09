import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as modelzoo

backbone_url = 'https://github.com/CoinCheung/BiSeNet/releases/download/0.0.0/backbone_v2.pth'


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        num_mid_filter = _make_divisible(channel // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels=channel, out_channels=num_mid_filter, kernel_size=1, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=num_mid_filter, out_channels=channel, kernel_size=1, bias=True)
        self.act2 = h_sigmoid()

    # def __init__(self, channel, reduction=4):
    #     super(SELayer, self).__init__()
    #     num_mid_filter = _make_divisible(channel // reduction, 8)
    #     self.avg_pool = nn.AdaptiveAvgPool2d(1)
    #     self.fcconv = nn.Sequential(
    #         nn.Conv2d(in_channels=channel, out_channels=num_mid_filter, kernel_size=1, bias=True),
    #         nn.ReLU(inplace=True),
    #         nn.Conv2d(in_channels=num_mid_filter, out_channels=channel, kernel_size=1, bias=True),
    #         h_sigmoid()
    #     )

    def forward(self, x):
        out = self.pool(x)
        out = self.act1(self.conv1(out))
        out = self.act2(self.conv2(out))
        return out * x
        # y = self.avg_pool(x)
        #
        # y = self.fcconv(y)
        # # print(y.size())
        # return x * y


class SSELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SSELayer, self).__init__()
        num_mid_filter = _make_divisible(channel // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels=channel, out_channels=num_mid_filter, kernel_size=1, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=num_mid_filter, out_channels=channel, kernel_size=1, bias=True)
        self.act2 = h_sigmoid()

    # def __init__(self, channel, reduction=4):
    #     super(SELayer, self).__init__()
    #     num_mid_filter = _make_divisible(channel // reduction, 8)
    #     self.avg_pool = nn.AdaptiveAvgPool2d(1)
    #     self.fcconv = nn.Sequential(
    #         nn.Conv2d(in_channels=channel, out_channels=num_mid_filter, kernel_size=1, bias=True),
    #         nn.ReLU(inplace=True),
    #         nn.Conv2d(in_channels=num_mid_filter, out_channels=channel, kernel_size=1, bias=True),
    #         h_sigmoid()
    #     )

    def forward(self, x):
        out = self.pool(x)
        out = self.act1(self.conv1(out))
        out = self.act2(self.conv2(out))
        return out * x
        # y = self.avg_pool(x)
        #
        # y = self.fcconv(y)
        # # print(y.size())
        # return x * y


class ConvBNReLU(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1,
                 dilation=1, groups=1, bias=False):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            in_chan, out_chan, kernel_size=ks, stride=stride,
            padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv(x)
        feat = self.bn(feat)
        feat = self.relu(feat)
        return feat


class ConvBNReLURes(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1,
                 dilation=1, groups=1, bias=False):
        super(ConvBNReLURes, self).__init__()
        self.conv = nn.Conv2d(
            in_chan, out_chan, kernel_size=ks, stride=stride,
            padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv(x)
        feat = self.bn(feat) + x
        feat = self.relu(feat)
        return feat


class UpSample(nn.Module):

    def __init__(self, n_chan, factor=2):
        super(UpSample, self).__init__()
        out_chan = n_chan * factor * factor
        self.proj = nn.Conv2d(n_chan, out_chan, 1, 1, 0)
        self.up = nn.PixelShuffle(factor)
        self.init_weight()

    def forward(self, x):
        feat = self.proj(x)
        feat = self.up(feat)
        return feat

    def init_weight(self):
        nn.init.xavier_normal_(self.proj.weight, gain=1.)


class DetailBranch(nn.Module):

    def __init__(self, ch=64):
        super(DetailBranch, self).__init__()
        self.S1 = nn.Sequential(
            ConvBNReLU(3, ch//2, 3, stride=2),
            ConvBNReLU(ch//2, ch//2, 3, stride=1),
        )
        self.S2 = nn.Sequential(
            ConvBNReLU(ch//2, ch, 3, stride=2),
            ConvBNReLU(ch, ch, 3, stride=1),
            ConvBNReLU(ch, ch, 3, stride=1),
        )
        # self.S1 = nn.Sequential(
        #     ConvBNReLU(3, ch, 3, stride=2),
        #     ConvBNReLU(ch, ch, 3, stride=1),
        # )
        # self.S2 = nn.Sequential(
        #     ConvBNReLU(ch, ch, 3, stride=2),
        #     ConvBNReLU(ch, ch, 3, stride=1),
        #     ConvBNReLU(ch, ch, 3, stride=1),
        # )
        self.S3 = nn.Sequential(
            ConvBNReLU(ch, ch * 2, 3, stride=2),
            ConvBNReLU(ch * 2, ch * 2, 3, stride=1),
            ConvBNReLU(ch * 2, ch * 2, 3, stride=1),
        )

    def forward(self, x):
        feat = self.S1(x)
        feat = self.S2(feat)
        feat = self.S3(feat)
        return feat


# class DetailBranch(nn.Module):
#
#     def __init__(self, ch=64):
#         super(DetailBranch, self).__init__()
#         self.S1 = nn.Sequential(
#             ConvBNReLU(3, ch//2, 3, stride=2),
#             ConvBNReLURes(ch//2, ch//2, 3, stride=1),
#         )
#         self.S2 = nn.Sequential(
#             ConvBNReLU(ch//2, ch, 3, stride=2),
#             ConvBNReLURes(ch, ch, 3, stride=1),
#             ConvBNReLURes(ch, ch, 3, stride=1),
#         )
#         self.S3 = nn.Sequential(
#             ConvBNReLU(ch, ch * 2, 3, stride=2),
#             ConvBNReLURes(ch * 2, ch * 2, 3, stride=1),
#             ConvBNReLURes(ch * 2, ch * 2, 3, stride=1),
#         )
#
#     def forward(self, x):
#         feat = self.S1(x)
#         feat = self.S2(feat)
#         feat = self.S3(feat)
#         return feat


class StemBlock(nn.Module):

    def __init__(self):
        super(StemBlock, self).__init__()
        self.conv = ConvBNReLU(3, 16, 3, stride=2)
        self.left = nn.Sequential(
            ConvBNReLU(16, 8, 1, stride=1, padding=0),
            ConvBNReLU(8, 16, 3, stride=2),
        )
        self.right = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, ceil_mode=False)
        self.fuse = ConvBNReLU(32, 16, 3, stride=1)

    def forward(self, x):
        feat = self.conv(x)
        feat_left = self.left(feat)
        feat_right = self.right(feat)
        feat = torch.cat([feat_left, feat_right], dim=1)
        feat = self.fuse(feat)
        return feat


class CEBlock(nn.Module):

    def __init__(self, ch=128):
        super(CEBlock, self).__init__()
        self.bn = nn.BatchNorm2d(ch)
        self.conv_gap = ConvBNReLU(ch, ch, 1, stride=1, padding=0)
        # TODO: in paper here is naive conv2d, no bn-relu
        self.conv_last = ConvBNReLU(ch, ch, 3, stride=1)
        # self.conv_last = ConvBNReLURes(ch, ch, 3, stride=1)

    def forward(self, x):
        feat = torch.mean(x, dim=(2, 3), keepdim=True)
        feat = self.bn(feat)
        feat = self.conv_gap(feat)
        feat = feat + x
        feat = self.conv_last(feat)
        return feat


# class GELayerS1(nn.Module):
#
#     def __init__(self, in_chan, out_chan, exp_ratio=6):
#         super(GELayerS1, self).__init__()
#         mid_chan = in_chan * exp_ratio
#         self.conv1 = ConvBNReLU(in_chan, in_chan, 3, stride=1)
#         self.dwconv = nn.Sequential(
#             nn.Conv2d(
#                 in_chan, mid_chan, kernel_size=3, stride=1,
#                 padding=1, groups=in_chan, bias=False),
#             nn.BatchNorm2d(mid_chan),
#             nn.ReLU(inplace=True),  # not shown in paper
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(
#                 mid_chan, out_chan, kernel_size=1, stride=1,
#                 padding=0, bias=False),
#             nn.BatchNorm2d(out_chan),
#         )
#         self.conv2[1].last_bn = True
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         feat = self.conv1(x)
#         feat = self.dwconv(feat)
#         feat = self.conv2(feat)
#         feat = feat + x
#         feat = self.relu(feat)
#         return feat
#
#
# class GELayerS2(nn.Module):
#
#     def __init__(self, in_chan, out_chan, exp_ratio=6):
#         super(GELayerS2, self).__init__()
#         mid_chan = in_chan * exp_ratio
#         self.conv1 = ConvBNReLU(in_chan, in_chan, 3, stride=1)
#         self.dwconv1 = nn.Sequential(
#             nn.Conv2d(
#                 in_chan, mid_chan, kernel_size=3, stride=2,
#                 padding=1, groups=in_chan, bias=False),
#             nn.BatchNorm2d(mid_chan),
#         )
#         self.dwconv2 = nn.Sequential(
#             nn.Conv2d(
#                 mid_chan, mid_chan, kernel_size=3, stride=1,
#                 padding=1, groups=mid_chan, bias=False),
#             nn.BatchNorm2d(mid_chan),
#             nn.ReLU(inplace=True),  # not shown in paper
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(
#                 mid_chan, out_chan, kernel_size=1, stride=1,
#                 padding=0, bias=False),
#             nn.BatchNorm2d(out_chan),
#         )
#         self.conv2[1].last_bn = True
#         self.shortcut = nn.Sequential(
#             nn.Conv2d(
#                 in_chan, in_chan, kernel_size=3, stride=2,
#                 padding=1, groups=in_chan, bias=False),
#             nn.BatchNorm2d(in_chan),
#             nn.Conv2d(
#                 in_chan, out_chan, kernel_size=1, stride=1,
#                 padding=0, bias=False),
#             nn.BatchNorm2d(out_chan),
#         )
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         feat = self.conv1(x)
#         feat = self.dwconv1(feat)
#         feat = self.dwconv2(feat)
#         feat = self.conv2(feat)
#         shortcut = self.shortcut(x)
#         feat = feat + shortcut
#         feat = self.relu(feat)
#         return feat


class GELayerS1(nn.Module):

    def __init__(self, in_chan, out_chan, exp_ratio=6):
        super(GELayerS1, self).__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1 = ConvBNReLU(in_chan, in_chan, 3, stride=1)
        # change group wise to 1x1
        self.conv1x1 = ConvBNReLU(in_chan, mid_chan, ks=1, stride=1, padding=0)
        self.dwconv = nn.Sequential(
            nn.Conv2d(
                mid_chan, mid_chan, kernel_size=3, stride=1,
                padding=1, groups=mid_chan, bias=False),
            nn.BatchNorm2d(mid_chan),
            nn.ReLU(inplace=True),  # not shown in paper
        )
        # SE block
        # self.se = SELayer(mid_chan)

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                mid_chan, out_chan, kernel_size=1, stride=1,
                padding=0, bias=False),
            nn.BatchNorm2d(out_chan),
        )
        self.conv2[1].last_bn = True
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv1(x)

        feat = self.conv1x1(feat)
        feat = self.dwconv(feat)

        # feat = self.se(feat)
        feat = self.conv2(feat)
        feat = feat + x
        feat = self.relu(feat)
        return feat


class GELayerS2(nn.Module):

    def __init__(self, in_chan, out_chan, exp_ratio=6):
        super(GELayerS2, self).__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1 = ConvBNReLU(in_chan, in_chan, 3, stride=1)
        # change group wise to regular
        self.conv1x1 = ConvBNReLU(in_chan, mid_chan, ks=1, stride=1, padding=0)
        self.dwconv1 = nn.Sequential(
            nn.Conv2d(
                mid_chan, mid_chan, kernel_size=3, stride=2,
                padding=1, groups=mid_chan, bias=False),
            nn.BatchNorm2d(mid_chan),
        )
        self.dwconv2 = nn.Sequential(
            nn.Conv2d(
                mid_chan, mid_chan, kernel_size=3, stride=1,
                padding=1, groups=mid_chan, bias=False),
            nn.BatchNorm2d(mid_chan),
            nn.ReLU(inplace=True),  # not shown in paper
        )

        # add SE
        # self.se = SELayer(mid_chan)

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                mid_chan, out_chan, kernel_size=1, stride=1,
                padding=0, bias=False),
            nn.BatchNorm2d(out_chan),
        )
        self.conv2[1].last_bn = True
        self.shortcut = nn.Sequential(
            nn.Conv2d(
                in_chan, in_chan, kernel_size=3, stride=2,
                padding=1, groups=in_chan, bias=False),
            nn.BatchNorm2d(in_chan),
            nn.Conv2d(
                in_chan, out_chan, kernel_size=1, stride=1,
                padding=0, bias=False),
            nn.BatchNorm2d(out_chan),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.conv1x1(feat)
        feat = self.dwconv1(feat)
        feat = self.dwconv2(feat)

        # feat = self.se(feat)

        feat = self.conv2(feat)
        shortcut = self.shortcut(x)
        feat = feat + shortcut
        feat = self.relu(feat)
        return feat


class SegmentBranch(nn.Module):

    def __init__(self, ch=32):
        super(SegmentBranch, self).__init__()
        self.S1S2 = StemBlock()
        self.S3 = nn.Sequential(
            GELayerS2(16, ch),
            GELayerS1(ch, ch),
        )
        self.S4 = nn.Sequential(
            GELayerS2(ch, ch * 2),
            GELayerS1(ch * 2, ch * 2),
        )
        self.S5_4 = nn.Sequential(
            GELayerS2(ch * 2, ch * 4),
            GELayerS1(ch * 4, ch * 4),
            GELayerS1(ch * 4, ch * 4),
            GELayerS1(ch * 4, ch * 4),
        )
        self.S5_5 = CEBlock(ch * 4)

    def forward(self, x):
        feat2 = self.S1S2(x)
        feat3 = self.S3(feat2)
        feat4 = self.S4(feat3)
        feat5_4 = self.S5_4(feat4)
        feat5_5 = self.S5_5(feat5_4)
        return feat2, feat3, feat4, feat5_4, feat5_5


class BGALayer(nn.Module):

    def __init__(self, left_width=128, right_width=128, out_width=128):
        super(BGALayer, self).__init__()
        self.out_width = out_width
        self.left1 = nn.Sequential(
            nn.Conv2d(
                left_width, left_width, kernel_size=3, stride=1,
                padding=1, groups=left_width, bias=False),
            nn.BatchNorm2d(left_width),
            nn.Conv2d(
                left_width, self.out_width, kernel_size=1, stride=1,
                padding=0, bias=False),
        )
        self.left2 = nn.Sequential(
            nn.Conv2d(
                left_width, self.out_width, kernel_size=3, stride=2,
                padding=1, bias=False),
            nn.BatchNorm2d(self.out_width),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        )
        self.right1 = nn.Sequential(
            nn.Conv2d(
                right_width, self.out_width, kernel_size=3, stride=1,
                padding=1, bias=False),
            nn.BatchNorm2d(self.out_width),
        )
        self.right2 = nn.Sequential(
            nn.Conv2d(
                right_width, right_width, kernel_size=3, stride=1,
                padding=1, groups=right_width, bias=False),
            nn.BatchNorm2d(right_width),
            nn.Conv2d(
                right_width, self.out_width, kernel_size=1, stride=1,
                padding=0, bias=False),
        )
        self.up1 = nn.Upsample(scale_factor=4)
        self.up2 = nn.Upsample(scale_factor=4)
        ##TODO: does this really has no relu?
        self.conv = nn.Sequential(
            nn.Conv2d(
                self.out_width, self.out_width, kernel_size=3, stride=1,
                padding=1, bias=False),
            nn.BatchNorm2d(self.out_width),
            nn.ReLU(inplace=True),  # not shown in paper
        )
        # self.conv = ConvBNReLURes(right_width, right_width, 3, 1, 1, bias=False)
        self.sig = h_sigmoid()

    def forward(self, x_d, x_s):
        # dsize = x_d.size()[2:]
        left1 = self.left1(x_d)
        left2 = self.left2(x_d)
        right1 = self.right1(x_s)
        right2 = self.right2(x_s)
        right1 = self.up1(right1)

        # left = left1 * self.sig(right1)
        # right = left2 * self.sig(right2)
        left = left1 * torch.sigmoid(right1)
        right = left2 * torch.sigmoid(right2)

        right = self.up2(right)
        out = self.conv(left + right)
        return out


class SegmentHead(nn.Module):

    def __init__(self, in_chan, mid_chan, n_classes, up_factor=8, aux=True):
        super(SegmentHead, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, 3, stride=1)
        self.drop = nn.Dropout(0.1)
        self.up_factor = up_factor

        out_chan = n_classes
        mid_chan2 = up_factor * up_factor if aux else mid_chan
        up_factor = up_factor // 2 if aux else up_factor
        self.conv_out = nn.Sequential(
            nn.Sequential(
                nn.Upsample(scale_factor=2),
                ConvBNReLU(mid_chan, mid_chan2, 3, stride=1)
            ) if aux else nn.Identity(),
            nn.Conv2d(mid_chan2, out_chan, 1, 1, 0, bias=True),
            nn.Upsample(scale_factor=up_factor, mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        return feat


class NewSegmentHead(nn.Module):

    def __init__(self, in_chan, mid_chan, n_classes, up_factor=8, aux=True):
        super(NewSegmentHead, self).__init__()
        # self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = ConvBNReLU(in_chan, mid_chan, 3, stride=1)
        self.drop = nn.Dropout(0.1)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.up_factor = up_factor

        mid_chan2 = up_factor * up_factor if aux else mid_chan
        # up_factor = up_factor // 2 if aux else up_factor
        # print('up_factor', up_factor)
        self.conv_out = nn.Sequential(
            nn.Sequential(
                nn.Upsample(scale_factor=2),
                ConvBNReLU(mid_chan, mid_chan2, 3, stride=1)
            ) if aux else nn.Identity(),
            # ConvBNReLU(mid_chan, mid_chan2, 3, stride=1),
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(mid_chan2, n_classes, 1, 1, 0, bias=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        # feat = self.up1(x)
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.up2(feat)
        feat = self.conv_out(feat)
        return feat


class BiSeNetV2(nn.Module):

    def __init__(self, n_classes, aux_mode='train'):
        super(BiSeNetV2, self).__init__()
        self.ch = 16  # baseline 32
        self.aux_mode = aux_mode
        self.detail = DetailBranch(2 * self.ch)
        self.segment = SegmentBranch(self.ch)
        self.bga = BGALayer(4 * self.ch, 4 * self.ch, 4 * self.ch)

        # TODO: what is the number of mid chan ?
        # self.head = SegmentHead(128, 1024, n_classes, up_factor=8, aux=False)
        # self.head = SegmentHead(4 * self.ch, 32 * self.ch, n_classes, up_factor=8, aux=False)
        self.head = SegmentHead(4 * self.ch, 32 * self.ch, n_classes, up_factor=8, aux=False)
        # self.head = SegmentHead(4 * self.ch, 768, n_classes, up_factor=8, aux=False)
        # self.head = NewSegmentHead(4 * self.ch, 32 * self.ch, n_classes, up_factor=8, aux=False)

        if self.aux_mode == 'train':
            self.aux2 = SegmentHead(16, 128, n_classes, up_factor=4)
            # self.aux3 = SegmentHead(32, 128, n_classes, up_factor=8)
            # self.aux4 = SegmentHead(64, 128, n_classes, up_factor=16)
            # self.aux5_4 = SegmentHead(128, 128, n_classes, up_factor=32)
            self.aux3 = SegmentHead(self.ch, 128, n_classes, up_factor=8)
            self.aux4 = SegmentHead(2 * self.ch, 128, n_classes, up_factor=16)
            self.aux5_4 = SegmentHead(4 * self.ch, 128, n_classes, up_factor=32)

        self.init_weights()

    def forward(self, x):
        # size = x.size()[2:]
        feat_d = self.detail(x)
        feat2, feat3, feat4, feat5_4, feat_s = self.segment(x)
        feat_head = self.bga(feat_d, feat_s)

        logits = self.head(feat_head)
        if self.aux_mode == 'train':
            logits_aux2 = self.aux2(feat2)
            logits_aux3 = self.aux3(feat3)
            logits_aux4 = self.aux4(feat4)
            logits_aux5_4 = self.aux5_4(feat5_4)
            return logits, logits_aux2, logits_aux3, logits_aux4, logits_aux5_4
        elif self.aux_mode == 'eval':
            return logits,
        elif self.aux_mode == 'pred':
            pred = logits.argmax(dim=1)
            return pred
        else:
            raise NotImplementedError

    def init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
                if not module.bias is None: nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                if hasattr(module, 'last_bn') and module.last_bn:
                    nn.init.zeros_(module.weight)
                else:
                    nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        # self.load_pretrain()

    # def load_pretrain(self):
    #     state = modelzoo.load_url(backbone_url)
    #     for name, child in self.named_children():
    #         if name in state.keys():
    #             child.load_state_dict(state[name], strict=True)

    # def get_params(self):
    #     def add_param_to_list(mod, wd_params, nowd_params):
    #         for param in mod.parameters():
    #             if param.dim() == 1:
    #                 nowd_params.append(param)
    #             elif param.dim() == 4:
    #                 wd_params.append(param)
    #             else:
    #                 print(name)
    #
    #     wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
    #     for name, child in self.named_children():
    #         if 'head' in name or 'aux' in name:
    #             add_param_to_list(child, lr_mul_wd_params, lr_mul_nowd_params)
    #         else:
    #             add_param_to_list(child, wd_params, nowd_params)
    #     return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params


class ProfileConv(nn.Module):
    def __init__(self, model):
        super(ProfileConv, self).__init__()
        self.model = model
        self.hooks = []
        self.macs = []
        self.params = []

        def hook_conv(module, input, output):
            self.macs.append(output.size(1) * output.size(2) * output.size(3) *
                             module.weight.size(-1) * module.weight.size(-1) * input[0].size(1) / module.groups)
            self.params.append(module.weight.size(0) * module.weight.size(1) *
                               module.weight.size(2) * module.weight.size(3))

        def hook_linear(module, input, output):
            self.macs.append(module.weight.size(0) * module.weight.size(1))
            self.params.append(module.weight.size(0) * module.weight.size(1))

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                self.hooks.append(module.register_forward_hook(hook_conv))
            elif isinstance(module, nn.Linear):
                self.hooks.append(module.register_forward_hook(hook_linear))

    def forward(self, x):
        self.model.to(x.device)
        _ = self.model(x)
        for handle in self.hooks:
            handle.remove()
        return self.macs, self.params


if __name__ == "__main__":
    #  x = torch.randn(16, 3, 1024, 2048)
    #  detail = DetailBranch()
    #  feat = detail(x)
    #  print('detail', feat.size())
    #
    #  x = torch.randn(16, 3, 1024, 2048)
    #  stem = StemBlock()
    #  feat = stem(x)
    #  print('stem', feat.size())
    #
    #  x = torch.randn(16, 128, 16, 32)
    #  ceb = CEBlock()
    #  feat = ceb(x)
    #  print(feat.size())
    #
    #  x = torch.randn(16, 32, 16, 32)
    #  ge1 = GELayerS1(32, 32)
    #  feat = ge1(x)
    #  print(feat.size())
    #
    #  x = torch.randn(16, 16, 16, 32)
    #  ge2 = GELayerS2(16, 32)
    #  feat = ge2(x)
    #  print(feat.size())
    #
    #  left = torch.randn(16, 128, 64, 128)
    #  right = torch.randn(16, 128, 16, 32)
    #  bga = BGALayer()
    #  feat = bga(left, right)
    #  print(feat.size())
    #
    #  x = torch.randn(16, 128, 64, 128)
    #  head = SegmentHead(128, 128, 19)
    #  logits = head(x)
    #  print(logits.size())
    #
    #  x = torch.randn(16, 3, 1024, 2048)
    #  segment = SegmentBranch()
    #  feat = segment(x)[0]
    #  print(feat.size())
    #
    x = torch.randn(4, 3, 512, 1024)
    model = BiSeNetV2(n_classes=19, aux_mode='eval')
    # setattr(model, 'ch', 64)
    profile = ProfileConv(model)
    MACs, params = profile(x)
    # print('input size: ', crop.size())

    # torch.onnx.export(model, torch.randn(1, x.size(1), x.size(2), x.size(3)).to(x.device),
    #                   "BiSeNetV2_search_9G.onnx", verbose=True)
    # print('onnx exported')

    print('number of conv&fc layers:', len(MACs))
    print(sum(MACs) / 1e9, 'GMACs')
    print(sum(params) / 1e6, 'M parameters')
    # exit()
    # outs = model(x)
    # for out in outs:
    #     print(out.size())
    #  print(logits.size())

    #  for name, param in model.named_parameters():
    #      if len(param.size()) == 1:
    #          print(name)
