import torch
import torch.nn as nn

class ExtractGANModel:
    def __init__(self, opt):
        self.opt = opt
        self.isTrain = opt.isTrain
        self.gpu_ids = opt.gpu_ids
        self.device = 'cuda:{}'.format(self.gpu_ids[0]) if self.gpu_ids else 'cpu'
        if opt.resize_or_crop != 'scale_width':
            torch.backends.cudnn.benchmark = True


class Encoder(nn.Module):
    def __init__(self, input_nc=3, same_size_nf=[64, 64], downsampling_nf=[128, 256], norm_layer=nn.InstanceNorm2d, padding_type='zero'):
        super(Encoder, self).__init__()
        # self.input_nc = input_nc
        # self.output_nc = output_nc
        padding_layer = nn.ReflectionPad2d if padding_type == 'reflect' else nn.ZeroPad2d
        use_bias = norm_layer == nn.InstanceNorm2d
        
        model = []

        # same-size layers
        for nf in same_size_nf:
            model += [
                padding_layer(1),
                nn.Conv2d(input_nc, nf, kernel_size=3, padding=0, bias=use_bias),
                norm_layer(nf),
                nn.ReLU(inplace=True),
                ]
            

        # downsampling layers
        last_nf = same_size_nf[-1]
        for nf in downsampling_nf:
            model += [
                padding_layer(1),
                nn.Conv2d(last_nf, nf, kernel_size=3, stride=2, padding=0, bias=use_bias),
                norm_layer(nf),
                nn.ReLU(inplace=True),
                ]
            last_nf = nf

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class Decoder(nn.Module):
    def __init__(self, output_nc=3, same_size_nf=[64, 64], upsampling_nf=[256, 128], norm_layer=nn.InstanceNorm2d, padding_type='zero'):
        super(Decoder, self).__init__()
        padding_layer = nn.ReflectionPad2d if padding_type == 'reflect' else nn.ZeroPad2d
        use_bias = norm_layer == nn.InstanceNorm2d
        
        model = []

        # upsampling layers


    def forward(self, input):
        # return self.model(input)
        pass


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type='zero', norm_layer = nn.InstanceNorm2d, use_dropout=False):
        super(ResnetBlock, self).__init__()
        use_bias = norm_layer == nn.InstanceNorm2d
        padding_layer = nn.ReflectionPad2d if padding_type == 'reflect' else nn.ZeroPad2d
        conv_block = []

        conv_block += [
            padding_layer(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(True),
            ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        conv_block += [
            padding_layer(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias),
            norm_layer(dim),
            ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out