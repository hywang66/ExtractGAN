import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
import torchvision.models

class ExtractGANModel:
    def __init__(self, opt):
        self.opt = opt
        self.isTrain = opt.isTrain
        self.gpu_ids = opt.gpu_ids
        self.device = 'cuda:{}'.format(self.gpu_ids[0]) if self.gpu_ids else 'cpu'
        if opt.resize_or_crop != 'scale_width':
            torch.backends.cudnn.benchmark = True


class BaseModule(nn.Module):
    def __init__(self):
        super(BaseModule, self).__init__()
    
    def weights_init_func(self, m, init_type, gain):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.BatchNorm2d):
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)


class Encoder(BaseModule):
    def __init__(self, input_nc=3, same_size_nf=[64, 64], init_type='xavier', 
                 norm_layer=nn.InstanceNorm2d, padding_type='zero'):
        super(Encoder, self).__init__()
        padding_layer = nn.ReflectionPad2d if padding_type == 'reflect' else nn.ZeroPad2d
        use_bias = norm_layer == nn.InstanceNorm2d
        
        model = []

        # same-size layers
        last_nf = input_nc
        for nf in same_size_nf:
            model += [
                padding_layer(1),
                nn.Conv2d(last_nf, nf, kernel_size=3, padding=0, bias=use_bias),
                norm_layer(nf),
                nn.ReLU(inplace=True),
                ]
            last_nf = nf

        self.model = nn.Sequential(*model)
        
        weights_init_func = lambda m : self.weights_init_func(m, init_type, gain=0.02)
        self.apply(weights_init_func)
        print('Encoder weights initialized using %s.' % init_type)

    def forward(self, input):
        return self.model(input)


class Decoder(BaseModule):
    def __init__(self, output_nc=3, same_size_nf=[64, 64], init_type='xavier',
                 norm_layer=nn.InstanceNorm2d, padding_type='zero'):
        super(Decoder, self).__init__()
        padding_layer = nn.ReflectionPad2d if padding_type == 'reflect' else nn.ZeroPad2d
        use_bias = norm_layer == nn.InstanceNorm2d
        
        model = []

        # same-size layers
        for i, nf in enumerate(same_size_nf):
            next_nf = output_nc if i == len(same_size_nf) - 1 else same_size_nf[i + 1]
            model += [
                padding_layer(1),
                nn.ConvTranspose2d(nf, next_nf, kernel_size=3, padding=0, bias=use_bias),
                norm_layer(nf),
                nn.Tanh(),
                ]

        self.model = nn.Sequential(*model)

        weights_init_func = lambda m : self.weights_init_func(m, init_type, gain=0.02)
        self.apply(weights_init_func)
        print('Decoder weights initialized using %s.' % init_type)


    def forward(self, input):
        return self.model(input)


class ResnetBlock(nn.Module):
    def __init__(self, dim=64, padding_type='zero', norm_layer = nn.InstanceNorm2d, use_dropout=False):
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
        out = F.relu(x + self.conv_block(x), inplace=True)
        return out
  

class StyleExtractor(BaseModule):
    def __init__(self, n_kernel_channels=64, init_type='xavier', n_hidden=1024):
        super(StyleExtractor, self).__init__()
        # self.cfg = [[64], [64, 'M', 128], [128, 'M', 256], [256, 256, 'M', 512]]
        self.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        # output size: 224*224*64, 112*112*128, 56*56*256, 28*28*512, 14*14*512, 7*7*512
        
        # self.vgg_block_set = set()
        # self.vgg_block_name_template='vgg_block_%d'
        # self.nblocks = len(self.cfg)
        self.nkc = n_kernel_channels
        self.vgg16 = None
        self.make_partial_vgg16()
        # self.n_vgg_parameters = len(list(self.vgg16.parameters()))
        self.init_vgg16()
    
        self.representor = nn.Sequential(
            nn.Linear(512 * 7 * 7, n_hidden),
            nn.ReLU(True),
            nn.Dropout(),
            # nn.Linear(4096, 4096),
            # nn.ReLU(True),
            # nn.Dropout()
        )
        self.conv_kernel_gen_1 = nn.Linear(n_hidden, n_kernel_channels*n_kernel_channels*3*3)
        self.conv_kernel_gen_2 = nn.Linear(n_hidden, n_kernel_channels*n_kernel_channels*3*3)
        self.conv_kernel_gen_3 = nn.Linear(n_hidden, n_kernel_channels*n_kernel_channels*3*3)
        self.conv_kernel_gen_4 = nn.Linear(n_hidden, n_kernel_channels*n_kernel_channels*3*3)
        
        weights_init_func = lambda m : self.weights_init_func(m, init_type, gain=0.02)
        for module in self.children():
            if module is not self.vgg16:
                module.apply(weights_init_func)       
        print('StyleExtractor weights initialized using %s.' % init_type)
        
        
    def forward(self, x):
        conv_kernels = []
        features = self.vgg16(x).detach()
        deep_features = self.representor(features)
        conv_kernels.append(self.conv_kernel_gen_1(deep_features).view(self.nkc, self.nkc, 3, 3))
        conv_kernels.append(self.conv_kernel_gen_2(deep_features).view(self.nkc, self.nkc, 3, 3))
        conv_kernels.append(self.conv_kernel_gen_3(deep_features).view(self.nkc, self.nkc, 3, 3))
        conv_kernels.append(self.conv_kernel_gen_4(deep_features).view(self.nkc, self.nkc, 3, 3))
        # vgg_outputs = []
        # src = x
        # for i in range(self.nblocks):
        #     block = getattr(self, self.vgg_block_name_template%i)
        #     dst  = block(src)
        #     vgg_outputs.append(dst)
        #     src = dst
        return conv_kernels

    # def make_partial_vgg16(self):
    #     features = []
    #     in_channels = 3
    #     for i, c in enumerate(self.cfg):
    #         layers = []
    #         for v in c:
    #             if v == 'M':
    #                 layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
    #             else:
    #                 conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
    #                 layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
    #                 in_channels = v
    #         block = nn.Sequential(*layers)
    #         setattr(self, self.vgg_block_name_template%i, block)
    #         self.vgg_block_set.add(block)

    def make_partial_vgg16(self):
        layers = []
        in_channels = 3
        for v in self.cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                in_channels = v
        self.vgg16 = nn.Sequential(*layers)
                      
    def train(self, mode=True):
        r"""
        Override the train method inherited from nn.Module to keep vgg blocks always in train mode.
        """
        self.training = mode
        for module in self.children():
            # if module in self.vgg_block_set:
            if module in self.vgg_layer_set:
                module.train(False)
            else:
                module.train(mode)
        return self
    
    def init_vgg16(self):
        vgg16_state_dict = torchvision.models.vgg16_bn(pretrained=True).state_dict()
        dict_new = self.state_dict().copy()
        new_list = list(self.state_dict().keys())
        trained_list = list(vgg16_state_dict.keys())
        
        # for i in range(self.n_vgg_parameters):
        for i, _ in enumerate(self.vgg16.parameters()):
            dict_new[new_list[i]] = vgg16_state_dict[trained_list[i]]
        
        self.load_state_dict(dict_new)
        print('VGG parameters loaded.')
            

class StyleWhitener(BaseModule):
    def __init__(self, n_blocks=2, dim=64, init_type='xavier', padding_type='zero', 
                 norm_layer = nn.InstanceNorm2d, use_dropout=False):
        super(StyleWhitener, self).__init__()
        model = []
        for _ in range(n_blocks):
            model.append(
                ResnetBlock(dim=64, padding_type='zero', 
                norm_layer = nn.InstanceNorm2d, use_dropout=False)
                )
        self.model = nn.Sequential(*model)
        
        weights_init_func = lambda m : self.weights_init_func(m, init_type, gain=0.02)
        self.apply(weights_init_func)
        print('StyleWhitener weights initialized using %s.' % init_type)  

    def forward(self, x):
        return self.model(x)
 

class KernelSpecifiedResnetBlock(nn.Module):
    def __init__(self):
        super(KernelSpecifiedResnetBlock, self).__init__()

    def forward(self, x, kernel1, kernel2):
        conv1 = F.conv2d(x, kernel1, padding=1)
        conv1_relu = F.relu(conv1, inplace=True)
        conv2 = F.conv2d(conv1_relu, kernel2, padding=1)
        out = F.relu(x + conv2, inplace=True)
        return out
  


class Stylizer(nn.Module):
    def __init__(self):
        super(Stylizer, self).__init__()
        self.ksr_block1 = KernelSpecifiedResnetBlock()
        self.ksr_block2 = KernelSpecifiedResnetBlock()
    
    def forward(self, whitened, conv_kernels):
        tmp = self.ksr_block1(whitened, conv_kernels[0], conv_kernels[1])
        return self.ksr_block2(tmp, conv_kernels[3], conv_kernels[4])


class Generator(nn.Module):
    def __init__(self, input_nc=3, init_type='xavier', 
                 norm_layer=nn.InstanceNorm2d, padding_type='zero'):
        super(Generator, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.style_extractor = StyleExtractor()
        self.style_whitener = StyleWhitener()
        self.stylizer = Stylizer()
        print('Generator build success!')

    def forward(self, content_img, style_img):
        encoded = self.encoder(content_img)
        whitened = self.style_whitener(encoded)
        conv_kernels = self.style_extractor(style_img)
        stylized = self.stylizer(whitened, conv_kernels)
        out = self.decoder(stylized)
        return out






# class Encoder_downsampling(nn.Module):
#     def __init__(self, input_nc=3, same_size_nf=[64, 64], downsampling_nf=[128, 256], 
#                  norm_layer=nn.InstanceNorm2d, padding_type='zero'):
#         super(Encoder_downsampling, self).__init__()
#         padding_layer = nn.ReflectionPad2d if padding_type == 'reflect' else nn.ZeroPad2d
#         use_bias = norm_layer == nn.InstanceNorm2d
        
#         model = []

#         # same-size layers
#         last_nf = input_nc
#         for nf in same_size_nf:
#             model += [
#                 padding_layer(1),
#                 nn.Conv2d(last_nf, nf, kernel_size=3, padding=0, bias=use_bias),
#                 norm_layer(nf),
#                 nn.ReLU(inplace=True),
#                 ]
#             last_nf = nf
            

#         # downsampling layers
#         last_nf = same_size_nf[-1]
#         for nf in downsampling_nf:
#             model += [
#                 padding_layer(1),
#                 nn.Conv2d(last_nf, nf, kernel_size=3, stride=2, padding=0, bias=use_bias),
#                 norm_layer(nf),
#                 nn.ReLU(inplace=True),
#                 ]
#             last_nf = nf

#         self.model = nn.Sequential(*model)

#     def forward(self, input):
#         return self.model(input)


# class Decoder_upsampling(nn.Module):
#     def __init__(self, output_nc=3, same_size_nf=[64, 64], upsampling_nf=[256, 128], 
#                  norm_layer=nn.InstanceNorm2d, padding_type='zero'):
#         super(Decoder_upsampling, self).__init__()
#         padding_layer = nn.ReflectionPad2d if padding_type == 'reflect' else nn.ZeroPad2d
#         use_bias = norm_layer == nn.InstanceNorm2d
        
#         model = []

#         # upsampling layers
#         for i, nf in enumerate(upsampling_nf):
#             next_nf = same_size_nf[0] if i == len(upsampling_nf) - 1 else upsampling_nf[i + 1]
#             model += [
#                 padding_layer(1),
#                 nn.ConvTranspose2d(nf, next_nf, kernel_size=3, stride=2, padding=0, bias=use_bias),
#                 norm_layer(nf),
#                 nn.ReLU(inplace=True),
#                 ]


#         # same-size layers
#         for i, nf in enumerate(same_size_nf):
#             next_nf = output_nc if i == len(same_size_nf) - 1 else same_size_nf[i + 1]
#             model += [
#                 padding_layer(1),
#                 nn.ConvTranspose2d(nf, next_nf, kernel_size=3, padding=0, bias=use_bias),
#                 norm_layer(nf),
#                 nn.Tanh(),
#                 ]

#         self.model = nn.Sequential(*model)

#     def forward(self, input):
#         return self.model(input)
