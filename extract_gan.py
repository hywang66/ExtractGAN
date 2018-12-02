import itertools
import os
import time

import torch
import torch.nn as nn
import torchvision.models
import torchvision.transforms as transforms
from torch.nn import functional as F
from torch.nn import init

from options.train_options import TrainOptions


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
    def __init__(self, input_nc=3, same_size_nf=[64, 64], init_type='xavier', norm_layer=nn.InstanceNorm2d, padding_type='zero'):
        super(Encoder, self).__init__()
        padding_layer = nn.ReflectionPad2d if padding_type == 'reflect' else nn.ZeroPad2d
        use_bias = norm_layer == nn.InstanceNorm2d
        
        model = []

        # same-size layers
        last_nf = input_nc
        for nf in same_size_nf:
            model += [
                padding_layer(1),
                nn.Conv2d(last_nf, nf, stride=2, kernel_size=3, padding=0, bias=use_bias),
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
    def __init__(self, output_nc=3, same_size_nf=[64, 64], init_type='xavier', norm_layer=nn.InstanceNorm2d, padding_type='zero'):
        super(Decoder, self).__init__()
        padding_layer = nn.ReflectionPad2d if padding_type == 'reflect' else nn.ZeroPad2d
        use_bias = norm_layer == nn.InstanceNorm2d
        
        model = []

        # same-size layers
        for i, nf in enumerate(same_size_nf):
            next_nf = output_nc if i == len(same_size_nf) - 1 else same_size_nf[i + 1]
            model += [
                # padding_layer(1),
                nn.ConvTranspose2d(nf, next_nf, stride=2, kernel_size=3, padding=1, output_padding=1, bias=use_bias),
                norm_layer(nf),
                ]
            if i < len(same_size_nf) - 1:
                model += [nn.ReLU()]
            else:
                model += [nn.Sigmoid()]

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
    def __init__(self, vgg=None, n_kernel_channels=64, init_type='xavier', n_hidden=1024):
        super(StyleExtractor, self).__init__()

        self.nkc = n_kernel_channels
        self.vgg16 = vgg
        self.register_buffer('vgg_mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('vgg_std', torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.representor = nn.Sequential(
            nn.Linear(512 * 7 * 7, n_hidden),
            nn.ReLU(True),
            nn.Dropout(),
            # nn.Linear(4096, 4096),
            # nn.ReLU(True),
            # nn.Dropout()
        )
        self.conv_kernel_gen_1 = nn.Sequential(nn.Linear(n_hidden, n_kernel_channels*n_kernel_channels*3*3), nn.Tanh())
        self.conv_kernel_gen_2 = nn.Sequential(nn.Linear(n_hidden, n_kernel_channels*n_kernel_channels*3*3), nn.Tanh())
        self.conv_kernel_gen_3 = nn.Sequential(nn.Linear(n_hidden, n_kernel_channels*n_kernel_channels*3*3), nn.Tanh())
        self.conv_kernel_gen_4 = nn.Sequential(nn.Linear(n_hidden, n_kernel_channels*n_kernel_channels*3*3), nn.Tanh())
        
        weights_init_func = lambda m : self.weights_init_func(m, init_type, gain=0.02)
        for module in self.children():
            if module is not self.vgg16:
                module.apply(weights_init_func)       
        print('StyleExtractor weights initialized using %s.' % init_type)
        
        
    def forward(self, x):
        conv_kernels = []
        features = self.vgg16((x-self.vgg_mean)/self.vgg_std)
        batch_size  = features.size(0)
        features = features.view(batch_size, -1)
        deep_features = self.representor(features)
        conv_kernels.append([k for k in self.conv_kernel_gen_1(deep_features).view(batch_size, self.nkc, self.nkc, 3, 3)])
        conv_kernels.append([k for k in self.conv_kernel_gen_2(deep_features).view(batch_size, self.nkc, self.nkc, 3, 3)])
        conv_kernels.append([k for k in self.conv_kernel_gen_3(deep_features).view(batch_size, self.nkc, self.nkc, 3, 3)])
        conv_kernels.append([k for k in self.conv_kernel_gen_4(deep_features).view(batch_size, self.nkc, self.nkc, 3, 3)])

        return conv_kernels
                                  

class StyleWhitener(BaseModule):
    def __init__(self, n_blocks=2, dim=64, init_type='xavier', padding_type='zero', norm_layer = nn.InstanceNorm2d, use_dropout=False):
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

    def forward(self, x_batch_tensor, kernel1, kernel2):
        conv1_batch = [F.conv2d(x.unsqueeze(0), kernel1[i], padding=1) for i, x in enumerate(x_batch_tensor)]
        conv1_relu_batch = [F.relu(conv1, inplace=True) for conv1 in conv1_batch]
        conv2_batch = [F.conv2d(conv1_relu, kernel2[i], padding=1) for i, conv1_relu in enumerate(conv1_relu_batch)]
        out = F.relu(x_batch_tensor + torch.cat(conv2_batch, dim=0), inplace=True)
        return out

    
class Stylizer(nn.Module):
    def __init__(self):
        super(Stylizer, self).__init__()
        self.ksr_block1 = KernelSpecifiedResnetBlock()
        self.ksr_block2 = KernelSpecifiedResnetBlock()
    
    def forward(self, whitened, conv_kernels):
        tmp = self.ksr_block1(whitened, conv_kernels[0], conv_kernels[1])
        return self.ksr_block2(tmp, conv_kernels[2], conv_kernels[3])


class Generator(nn.Module):
    def __init__(self, vgg=None, input_nc=3, init_type='xavier', norm_layer=nn.InstanceNorm2d, padding_type='zero'):
        super(Generator, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.style_extractor = StyleExtractor(vgg=vgg)
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



class Discriminator(BaseModule):
    def __init__(self, vgg=None, init_type='xavier', n_hidden=1024):
        super(Discriminator, self).__init__()

        self.vgg16 = vgg
        self.register_buffer('vgg_mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('vgg_std', torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7 * 2, n_hidden),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(n_hidden, 1),
            nn.Sigmoid()
        )

        weights_init_func = lambda m : self.weights_init_func(m, init_type, gain=0.02)
        for module in self.children():
            if module is not self.vgg16:
                module.apply(weights_init_func)       
        print('Discriminator weights initialized using %s.' % init_type)

    def forward(self, img, ref):
        feature1 = self.vgg16((img - self.vgg_mean)/self.vgg_std).view(img.size(0), -1)
        feature2 = self.vgg16((ref - self.vgg_mean)/self.vgg_std).view(img.size(0), -1)
        feature_cat = torch.cat((feature1, feature2), 1)
        prob = self.classifier(feature_cat)
        return prob
    

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=False, same_style=1.0, diff_style=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('same_style', torch.tensor(same_style))
        self.register_buffer('diff_style', torch.tensor(diff_style))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, styles_are_same):
        if styles_are_same:
            target_tensor = self.same_style
        else:
            target_tensor = self.diff_style
        return target_tensor.expand_as(input)

    def __call__(self, input, styles_are_same):
        target_tensor = self.get_target_tensor(input, styles_are_same)
        return self.loss(input, target_tensor)

    
class ExtractGANModel:
    def __init__(self, opt):
        self.opt = opt
        self.isTrain = opt.isTrain
        self.name = opt.name
        self.save_dir = os.path.join(opt.checkpoints_dir, self.name)
        self.gpu_ids = opt.gpu_ids
        self.device = 'cuda:{}'.format(self.gpu_ids[0]) if self.gpu_ids else 'cpu'
        if opt.resize_or_crop != 'scale_width':
            torch.backends.cudnn.benchmark = True

        self.vgg16 = self.get_pretrained_vgg()
        self.G = Generator(vgg=self.vgg16)

        if self.isTrain:
            self.D = Discriminator(vgg=self.vgg16)
            self.criterionGAN = GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionCycle = torch.nn.L1Loss().to(self.device)
            # self.criterionAE = torch.nn.L1Loss().to(self.device)
            self.criterionAE = torch.nn.MSELoss().to(self.device)
            self.optimizer_G = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_AE = torch.optim.Adam(itertools.chain(self.G.encoder.parameters(), self.G.decoder.parameters()),
                                                lr=opt.lr_ae, betas=(opt.beta1, 0.999))
            # self.optimizer_AE = torch.optim.Adam(self.G.parameters(),
                                                # lr=opt.lr_ae, betas=(opt.beta1, 0.999))
                                                   
        print('New ExtractGAN model initialized!')

    def get_pretrained_vgg(self):
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                in_channels = v
        vgg = nn.Sequential(*layers)

        full_vgg = torchvision.models.vgg16_bn(pretrained=True)
        full_vgg_state_dict = full_vgg.state_dict()
        dict_new = vgg.state_dict().copy()
        new_list = list(vgg.state_dict().keys())
        trained_list = list(full_vgg_state_dict.keys())

        for i, _ in enumerate(full_vgg.parameters()):
            dict_new[new_list[i]] = full_vgg_state_dict[trained_list[i]]
        
        vgg.load_state_dict(dict_new)
        print('VGG parameters loaded.')

        for p in vgg.parameters():
            p.requires_grad = False

        return vgg

    def set_input(self, input):
        # The image that will be transfered, 224*224
        self.ori_img = input[0].to(self.device).requires_grad_(True)
        # The image in target style, 224*224
        self.style_img = input[1].to(self.device).requires_grad_(True)
        
        if self.isTrain:
            # Another image in target style for training D, 224*224
            self.style_ref_img = input[2].to(self.device).requires_grad_(True)
        
        # Another image in ori_img's style for reconstruct ori_img, 224*224
        self.style_ori_img = input[3].to(self.device).requires_grad_(True)

    def set_ae_input(self, img):
        # The image that will be transfered, 224*224
        self.ori_img = img.to(self.device).requires_grad_(True)

    # set requies_grad=Fasle to avoid computation, keep vgg16 requires_grad=False
    def set_requires_grad(self, nets, requires_grad: bool):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for p in net.parameters():
                    p.requires_grad = requires_grad
        for p in self.vgg16.parameters():
            p.requires_grad = False

    def set_requires_grad_AE(self, requires_grad: bool):
        for p in self.D.parameters():
            p.requires_grad = False
        for p in self.G.parameters():
            p.requires_grad = False
        for p in self.G.encoder.parameters():
            p.requies_grad = requires_grad
        for p in self.G.decoder.parameters():
            p.requies_grad = requires_grad        

    def forward(self):
        self.stylized_img = self.G(self.ori_img, self.style_img)
        self.rec_img = self.G(self.stylized_img, self.style_ori_img)

    def forward_AE(self):
        self.ae_img = self.G.decoder(self.G.encoder(self.ori_img))

    def backward_G(self):
        self.loss_G_gen = self.criterionGAN(self.D(self.stylized_img, self.style_ref_img), True)
        self.loss_cycle = self.criterionCycle(self.rec_img, self.ori_img)
        self.loss_G = self.loss_G_gen + self.loss_cycle
        self.loss_G.backward()

    def backward_D(self):
        self.loss_D_same = self.criterionGAN(self.D(self.style_img, self.style_ref_img), True)
        self.loss_D_diff = self.criterionGAN(self.D(self.stylized_img.detach(), self.style_ref_img), False)
        self.loss_D = (self.loss_D_same + self.loss_D_diff) * 0.5
        self.loss_D.backward()

    def backward_AE(self):
        self.loss_AE = self.criterionAE(self.ae_img, self.ori_img)
        self.loss_AE.backward()

    def optimize_parameters(self):
        self.forward()

        self.set_requires_grad(self.D, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        self.set_requires_grad(self.D, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
    
    def optimize_parameters_AE(self):
        # self.set_requires_grad_AE(True) wrong!
        self.set_requires_grad(self.G, True)

        self.forward_AE()

        self.optimizer_AE.zero_grad()
        self.backward_AE()
        self.optimizer_AE.step()

    def train(self, mode=True):
        self.G.train(mode)
        self.D.train(mode)
        self.vgg16.eval()

    def eval(self):
        self.G.eval()
        self.D.eval()

    # def train_AE(self, mode=True):
    #     self.G.eval()
    #     self.G.encoder.train(mode=mode)
    #     self.G.decoder.train(mode=mode)

    # def eval_AE(self):
    #     self.G.eval() 

    # used in test time, wrapping `forward` in no_grad() so we don't save
    # intermediate steps for backprop
    def test(self):
        with torch.no_grad():
            self.forward()
    
    # save models to the disk
    def save_networks(self, prefix):
        os.makedirs(self.save_dir, exist_ok=True)
        G_save_filename = prefix + '_ExtractGAN_G.pth'
        G_save_path = os.path.join(self.save_dir, G_save_filename)
        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            torch.save(self.G.module.cpu().state_dict(), G_save_path)
            self.G.cuda(self.gpu_ids[0])
        else:
            torch.save(self.G.cpu().state_dict(), G_save_path)

        D_save_filename = prefix + '_ExtractGAN_D.pth' 
        D_save_path = os.path.join(self.save_dir, D_save_filename)
        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            torch.save(self.D.module.cpu().state_dict(), D_save_path)
            self.D.cuda(self.gpu_ids[0])
        else:
            torch.save(self.D.cpu().state_dict(), D_save_path)
    
    # load models from the disk
    def load_networks(self, load_dir, prefix):
        G_load_filename = prefix + '_G.pth'
        G_load_path = os.path.join(load_dir, G_load_filename)
        print('loading the model from ' + G_load_path)
        G_state_dict = torch.load(G_load_path, map_location=self.device)
        if hasattr(G_state_dict, '_metadata'):
            del G_state_dict._metadata
        self.G.load_state_dict(G_state_dict)

        D_load_filename = prefix + '_D.pth'
        D_load_path = os.path.join(load_dir, D_load_filename)
        print('loading the model from ' + D_load_path)
        D_state_dict = torch.load(D_load_path, map_location=self.device)
        if hasattr(D_state_dict, '_metadata'):
            del D_state_dict._metadata
        self.D.load_state_dict(D_state_dict)


# Useful CODE!
# Do NOT Delete!

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
