import itertools
import os
import time
from copy import deepcopy

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
    def __init__(self, input_nc=3, same_size_nf = [32], downsampling_nf=[64], init_type='xavier', norm_layer=nn.InstanceNorm2d, padding_type='zero'):
        super(Encoder, self).__init__()

        self.same_size_nf = same_size_nf
        self.downsampling_nf = downsampling_nf

        padding_layer = nn.ReflectionPad2d if padding_type == 'reflect' else nn.ZeroPad2d
        use_bias = norm_layer == nn.InstanceNorm2d
        
        model = []


        last_nf = input_nc
        
        # same-size layers
        if same_size_nf:
            for nf in same_size_nf:
                model += [
                    padding_layer(1),
                    nn.Conv2d(last_nf, nf, kernel_size=3, padding=0, bias=use_bias),
                    norm_layer(nf),
                    nn.LeakyReLU(0.2, True),
                    ]
                last_nf = nf

        # downsampling layers
        for nf in downsampling_nf:
            model += [
                padding_layer(1),
                nn.Conv2d(last_nf, nf, stride=2, kernel_size=3, padding=0, bias=use_bias),
                norm_layer(nf),
                nn.LeakyReLU(0.2, True),
                ]
            last_nf = nf

        self.model = nn.Sequential(*model)
        
        weights_init_func = lambda m : self.weights_init_func(m, init_type, gain=0.02)
        self.apply(weights_init_func)
        print('Encoder weights initialized using %s.' % init_type)

    def forward(self, input):
        return self.model(input)


class Decoder(BaseModule):
    def __init__(self, output_nc=3, same_size_nf=[32], upsampling_nf=[64], init_type='xavier', norm_layer=nn.InstanceNorm2d, padding_type='zero'):
        super(Decoder, self).__init__()

        self.same_size_nf = same_size_nf
        self.upsampling_nf = upsampling_nf

        # padding_layer = nn.ReflectionPad2d if padding_type == 'reflect' else nn.ZeroPad2d
        use_bias = norm_layer == nn.InstanceNorm2d
        
        model = []

        # upsampling layers
        last_next_nf = output_nc if not same_size_nf else same_size_nf[0]
        for i, nf in enumerate(upsampling_nf):
            next_nf = last_next_nf if i == len(upsampling_nf) - 1 else upsampling_nf[i + 1]
            model += [
                # padding_layer(1),
                nn.ConvTranspose2d(nf, next_nf, stride=2, kernel_size=3, padding=1, output_padding=1, bias=use_bias),
                norm_layer(nf),
                ]
            model += [nn.LeakyReLU(0.2, True)]


        # same-size layers
        if same_size_nf:
            for i, nf in enumerate(same_size_nf):
                next_nf = output_nc if i == len(same_size_nf) - 1 else same_size_nf[i + 1]
                model += [
                    # padding_layer(1),
                    nn.ConvTranspose2d(nf, next_nf, kernel_size=3, padding=1, bias=use_bias),
                    norm_layer(nf),
                    ]
                model += [nn.LeakyReLU(0.2, True)]
                    
        model = model[:-1]
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
            nn.LeakyReLU(0.2, True),
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
        out = F.leaky_relu(x + self.conv_block(x),0.2, inplace=True)
        return out
    
    
class StyleExtractor(BaseModule):
    def __init__(self, feature_extractor=None, input_size=224, n_kernel_channels=64, init_type='xavier', same_size_nf=[], 
                 downsampling_nf=[64, 64], n_hidden=1024):
        super(StyleExtractor, self).__init__()

        self.nkc = n_kernel_channels
        self.feature_extractor = Encoder(same_size_nf=same_size_nf, downsampling_nf=downsampling_nf)
        # self.register_buffer('vgg_mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        # self.register_buffer('vgg_std', torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        
        f_size = lambda x : int((x - 1)/2) + 1
        feature_extractor_size = input_size
        for _ in downsampling_nf:
            feature_extractor_size = f_size(feature_extractor_size)

        
        
        self.representor = nn.Sequential(
            nn.Linear(downsampling_nf[-1] * feature_extractor_size * feature_extractor_size, n_hidden),
            nn.LeakyReLU(0.2, True),
            # nn.ReLU(True),
            # nn.Dropout(),
            # nn.Linear(4096, 4096),
            # nn.ReLU(True),
            # nn.Dropout()
        )

        # self.conv_kernel_gen_1 = nn.Sequential(nn.Linear(n_hidden, n_kernel_channels*n_kernel_channels*3*3), nn.Tanh())
        # self.conv_kernel_gen_2 = nn.Sequential(nn.Linear(n_hidden, n_kernel_channels*n_kernel_channels*3*3), nn.Tanh())
        # self.conv_kernel_gen_3 = nn.Sequential(nn.Linear(n_hidden, n_kernel_channels*n_kernel_channels*3*3), nn.Tanh())
        # self.conv_kernel_gen_4 = nn.Sequential(nn.Linear(n_hidden, n_kernel_channels*n_kernel_channels*3*3), nn.Tanh())

        self.conv_kernel_gen_1 = nn.Linear(n_hidden, n_kernel_channels*n_kernel_channels*3*3)
        self.conv_kernel_gen_2 = nn.Linear(n_hidden, n_kernel_channels*n_kernel_channels*3*3)
        # self.conv_kernel_gen_3 = nn.Linear(n_hidden, n_kernel_channels*n_kernel_channels*3*3)
        # self.conv_kernel_gen_4 = nn.Linear(n_hidden, n_kernel_channels*n_kernel_channels*3*3)

        weights_init_func = lambda m : self.weights_init_func(m, init_type, gain=0.02)
        # for module in self.children():
            # if module is not self.feature_encoder:
            #     module.apply(weights_init_func)   
        self.apply(weights_init_func)       
        print('StyleExtractor weights initialized using %s.' % init_type)
        

    def forward(self, x):
        conv_kernels = []
        features = self.feature_extractor(x)
        batch_size  = features.size(0)
        features = features.view(batch_size, -1)
        deep_features = self.representor(features)
        # conv_kernels.append([k for k in self.conv_kernel_gen_1(deep_features).view(batch_size, self.nkc, -1).softmax(dim=2).view(batch_size, self.nkc, self.nkc, 3, 3)])
        # conv_kernels.append([k for k in self.conv_kernel_gen_2(deep_features).view(batch_size, self.nkc, -1).softmax(dim=2).view(batch_size, self.nkc, self.nkc, 3, 3)])
        # conv_kernels.append([k for k in self.conv_kernel_gen_3(deep_features).view(batch_size, self.nkc, -1).softmax(dim=2).view(batch_size, self.nkc, self.nkc, 3, 3)])
        # conv_kernels.append([k for k in self.conv_kernel_gen_4(deep_features).view(batch_size, self.nkc, -1).softmax(dim=2).view(batch_size, self.nkc, self.nkc, 3, 3)])
        conv_kernels.append([k for k in self.conv_kernel_gen_1(deep_features).view(batch_size, self.nkc, self.nkc, 3, 3)])
        conv_kernels.append([k for k in self.conv_kernel_gen_2(deep_features).view(batch_size, self.nkc, self.nkc, 3, 3)])
        # conv_kernels.append([k for k in self.conv_kernel_gen_3(deep_features).view(batch_size, self.nkc, self.nkc, 3, 3)])
        # conv_kernels.append([k for k in self.conv_kernel_gen_4(deep_features).view(batch_size, self.nkc, self.nkc, 3, 3)])

        return conv_kernels
                                  

class StyleWhitener(BaseModule):
    def __init__(self, n_blocks=1, dim=64, init_type='xavier', padding_type='zero', norm_layer = nn.InstanceNorm2d, use_dropout=False):
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
    def __init__(self, dim=64):
        super(KernelSpecifiedResnetBlock, self).__init__()
        self.norm_layer = nn.InstanceNorm2d(dim)

    def forward(self, x_batch_tensor, kernel1, kernel2):
        conv1_batch = [F.conv2d(x.unsqueeze(0), kernel1[i], padding=1) for i, x in enumerate(x_batch_tensor)]
        conv1_batch_normed = self.norm_layer(torch.cat(conv1_batch, dim=0))
        conv1_relu_batch = [F.leaky_relu(conv1, 0.2, inplace=True) for conv1 in conv1_batch_normed]
        conv2_batch = [F.conv2d(conv1_relu.unsqueeze(0), kernel2[i], padding=1) for i, conv1_relu in enumerate(conv1_relu_batch)]
        conv2_batch_normed = self.norm_layer(torch.cat(conv2_batch, dim=0))
        out = F.leaky_relu(x_batch_tensor + conv2_batch_normed, 0.2, inplace=True)
        # out = F.leaky_relu(conv2_batch_normed, 0.2, inplace=True)
        return out

    
class Stylizer(nn.Module):
    def __init__(self):
        super(Stylizer, self).__init__()
        self.ksr_block1 = KernelSpecifiedResnetBlock()
        # self.ksr_block2 = KernelSpecifiedResnetBlock()
    
    def forward(self, whitened, conv_kernels):
        # tmp = self.ksr_block1(whitened, conv_kernels[0], conv_kernels[1])
        # return self.ksr_block2(tmp, conv_kernels[2], conv_kernels[3])
        return self.ksr_block1(whitened, conv_kernels[0], conv_kernels[1])


class Generator(nn.Module):
    def __init__(self, vgg=None, same_size_nf=[32], upsampling_nf=[64], downsampling_nf=[64],
                 n_hidden=1024, init_type='xavier', norm_layer=nn.InstanceNorm2d, padding_type='zero'):
        super(Generator, self).__init__()
        self.encoder = Encoder(same_size_nf=same_size_nf, downsampling_nf=downsampling_nf)
        self.decoder = Decoder(same_size_nf=same_size_nf, upsampling_nf=upsampling_nf)
        self.style_extractor = StyleExtractor(same_size_nf=same_size_nf, downsampling_nf=downsampling_nf, n_hidden=n_hidden)
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
    def __init__(self, vgg=None, init_type='xavier', n_hidden=1024, additional_layer=True):
        super(Discriminator, self).__init__()

        self.vgg16 = vgg
        self.register_buffer('vgg_mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('vgg_std', torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

        classifier = []
        classifier += [
            nn.Linear(512 * 7 * 7 * 2, n_hidden),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(),
        ]
        if additional_layer:
            classifier += [
                nn.Linear(n_hidden, n_hidden),
                nn.LeakyReLU(0.2, True),
                nn.Dropout(), 
            ]
        classifier += [
            nn.Linear(n_hidden, 1),
            nn.Sigmoid()
        ]

        self.classifier = nn.Sequential(*classifier)

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


class EmbeddindDiscriminator(BaseModule):
    def __init__(self, feature_extractor=None, input_size=224, init_type='xavier', same_size_nf=[], 
                 downsampling_nf=[64, 64], n_hidden=2048, n_embedding=1024, additional_layer=False):
        super(EmbeddindDiscriminator, self).__init__()

        self.feature_extractor = Encoder(same_size_nf=same_size_nf, downsampling_nf=downsampling_nf)
        f_size = lambda x : int((x - 1)/2) + 1
        feature_extractor_size = input_size
        for _ in downsampling_nf:
            feature_extractor_size = f_size(feature_extractor_size)

        embeder = []
        embeder += [
            nn.Linear(downsampling_nf[-1] * feature_extractor_size * feature_extractor_size, n_hidden),
            nn.LeakyReLU(0.2, True),
            # nn.Dropout(),
        ]
        if additional_layer:
            embeder += [
                nn.Linear(n_hidden, n_hidden),
                nn.LeakyReLU(0.2, True),
                # nn.ReLU(True),
                # nn.Dropout(), 
            ]
        embeder += [
            nn.Linear(n_hidden, n_embedding),
        ]

        self.embeder = nn.Sequential(*embeder)

        weights_init_func = lambda m : self.weights_init_func(m, init_type, gain=0.02)
        self.apply(weights_init_func)      
        print('EmbeddindDiscriminator weights initialized using %s.' % init_type)

    def forward(self, img):
        feature = self.feature_extractor(img).view(img.size(0), -1)
        embedding = self.embeder(feature)
        return embedding
    

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=False, same_style=1.0, diff_style=0.0, coefficient=1e4):
        super(GANLoss, self).__init__()
        self.coefficient = coefficient
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
        return self.loss(input, target_tensor)*self.coefficient


class StyleLoss(nn.Module):

    def __init__(self):
        super(StyleLoss, self).__init__()
        self.targets = []
        self.target = None
        # self.target = gram_matrix(target_feature).detach()
        self.mode = 'learn'

    def gram_matrix(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)

    def forward(self, input):
        G = self.gram_matrix(input)
        if self.mode == 'loss':
            self.loss = self.weight * F.mse_loss(G, self.target)
        elif self.mode == 'learn':
            self.target = G.detach()
        return input

# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self, device):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).to(device)
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std




class LossNetwork(nn.Module):

    def __init__(self, device):
        super(LossNetwork, self).__init__()
        cnn = deepcopy(torchvision.models.vgg16(pretrained=True).features).to(device).eval()
        normalization = Normalization(device)
        # just in order to have an iterable access to or list of content/syle
        # losses
        # content_losses = []
        style_losses = []

        # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
        # to put in modules that are supposed to be activated sequentially
        model = nn.Sequential(normalization)

        # desired depth layers to compute style/content losses :
        # content_layers = ['conv_9']
        # content_weight = {
        #     'conv_9': 1
        # }
        style_layers = [ 'conv_2', 'conv_4', 'conv_6', 'conv_9']
        style_weight = {
            'conv_2': 1,
            'conv_4': 1,
            'conv_6': 1,
            'conv_9': 1,
        }

        i = 0  # increment every time we see a conv
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                # The in-place version doesn't play very nicely with the ContentLoss
                # and StyleLoss we insert below. So we replace with out-of-place
                # ones here.
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            # if name in content_layers:
            #     # add content loss:
            #     # target_feature = model(content_img).detach()
            #     content_loss = ContentLoss()
            #     content_loss.weight = content_weight[name]
            #     model.add_module("content_loss_{}".format(i), content_loss)
            #     content_losses.append(content_loss)

            if name in style_layers:
                # add style loss:
                # target_feature = model(style_img).detach()
                style_loss = StyleLoss()
                style_loss.weight = style_weight[name]
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        
        
        # now we trim off the layers after the last content and style losses
        for i in range(len(model) - 1, -1, -1):
            # if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            if isinstance(model[i], StyleLoss):
                break

        model = model[:(i + 1)]

        self.model = model
        self.style_losses = style_losses
        # self.content_losses = content_losses

    # def learn_content(self, input):
    #     for cl in self.content_losses:
    #         cl.mode = 'learn'
    #     for sl in self.style_losses:
    #         sl.mode = 'nop'
    #     self.model(input) # feed image to vgg19
    
    def learn_style(self, input):
        # for cl in self.content_losses:
        #     cl.mode = 'nop'
        for sl in self.style_losses: 
            sl.mode = 'learn'
        self.model(input) # feed image to vgg19

    def forward(self, input, style):
        # self.learn_content(content)
        self.learn_style(style)

        # for cl in self.content_losses:
        #     cl.mode = 'loss'
        for sl in self.style_losses:
            sl.mode = 'loss'
        self.model(input) # feed image to vgg19

        # content_loss = 0
        style_loss = 0

        # for cl in self.content_losses:
        #     content_loss += cl.loss
        for sl in self.style_losses:
            style_loss += sl.loss

        # return content_loss, style_loss
        return style_loss




class ExtractGANModel:
    def __init__(self, opt, same_size_nf=[32], upsampling_nf=[64], downsampling_nf=[64]):
        self.opt = opt
        self.isTrain = opt.isTrain
        self.name = opt.name
        self.save_dir = os.path.join(opt.checkpoints_dir, self.name)
        self.gpu_ids = opt.gpu_ids
        self.device = 'cuda:{}'.format(self.gpu_ids[0]) if self.gpu_ids else 'cpu'
        if self.device is not 'cpu':
            torch.backends.cudnn.benchmark = True

        # self.vgg16 = self.get_pretrained_vgg().to(self.device)
        # self.vgg16 = None
        self.G = Generator(same_size_nf=same_size_nf, n_hidden=opt.style_extractor_n_hidden,
                           upsampling_nf=upsampling_nf, downsampling_nf=downsampling_nf).to(self.device)

        # self.bcedloss = GANLoss(use_lsgan=False, 
        #                         coefficient=opt.gl_coefficient).to(self.device)

        if self.isTrain:
            # self.D = Discriminator(vgg=self.vgg16, n_hidden=opt.discriminator_n_hidden, 
                                #    additional_layer=not opt.no_D_additional_layer).to(self.device)
            # self.criterionGAN = GANLoss(use_lsgan=not opt.no_lsgan, 
            #                             coefficient=opt.gl_coefficient).to(self.device)

            # self.triplet_loss_G = nn.TripletMarginLoss(margin=1.0, p=2)
            # self.triplet_loss_D = nn.TripletMarginLoss(margin=1.0, p=2)

            self.style_loss_net = LossNetwork(self.device)


            self.mseloss = self.get_MSELoss()
            self.l1loss = self.get_L1Loss()

            # self.D = EmbeddindDiscriminator(input_size=opt.input_size, n_hidden=opt.discriminator_n_hidden, 
                                                    #   same_size_nf=same_size_nf, downsampling_nf=downsampling_nf).to(self.device)


            # Do NOT use torch.nn.L1Loss or torch.nn.MSELoss, they are quite bugful in PyTorch 0.4.
            if opt.cycle_loss == 'L1':
                # self.criterionCycle = torch.nn.L1Loss().to(self.device)
                self.criterionCycle = self.get_L1Loss()
            elif opt.cycle_loss == 'MSE':
                # self.criterionCycle = torch.nn.MSELoss().to(self.device)
                self.criterionCycle = self.get_MSELoss()
            else:
                raise NotImplementedError('cycle_loss [%s] is not implemented' % opt.cycle_loss)

            if opt.ae_loss == 'L1':
                # self.criterionAE = torch.nn.L1Loss().to(self.device)
                self.criterionAE = self.get_L1Loss()
            elif opt.ae_loss == 'MSE':
                # self.criterionAE = torch.nn.MSELoss().to(self.device)
                self.criterionAE = self.get_MSELoss()
            else:
                raise NotImplementedError('ae_loss [%s] is not implemented' % opt.ae_loss)

            self.optimizer_G = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            # self.optimizer_D = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D.parameters()),
            #                                     lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_AE = torch.optim.Adam(itertools.chain(self.G.encoder.parameters(), self.G.decoder.parameters()),
                                                lr=opt.lr_ae, betas=(opt.beta1, 0.999))
            # self.optimizer_AE = torch.optim.Adam(self.G.parameters(),
                                                # lr=opt.lr_ae, betas=(opt.beta1, 0.999))
                                                   
        print('New ExtractGAN model initialized!')

    def get_L1Loss(self):
        return lambda a, b : torch.mean(torch.abs(a-b))
    
    def get_MSELoss(self):
        return lambda a, b : torch.mean(torch.pow(a-b, 2))


    # def get_pretrained_vgg(self):
    #     cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

    #     layers = []
    #     in_channels = 3
    #     for v in cfg:
    #         if v == 'M':
    #             layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
    #         else:
    #             conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
    #             layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
    #             in_channels = v
    #     vgg = nn.Sequential(*layers)

    #     full_vgg = torchvision.models.vgg16_bn(pretrained=True)
    #     full_vgg_state_dict = full_vgg.state_dict()
    #     dict_new = vgg.state_dict().copy()
    #     new_list = list(vgg.state_dict().keys())
    #     trained_list = list(full_vgg_state_dict.keys())

    #     for i, _ in enumerate(full_vgg.parameters()):
    #         dict_new[new_list[i]] = full_vgg_state_dict[trained_list[i]]
        
    #     vgg.load_state_dict(dict_new)
    #     print('VGG parameters loaded.')

    #     for p in vgg.parameters():
    #         p.requires_grad = False

    #     return vgg

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
        # for p in self.vgg16.parameters():
        #     p.requires_grad = False

    # def set_requires_grad_AE(self, requires_grad: bool):
    #     for p in self.D.parameters():
    #         p.requires_grad = False
    #     for p in self.G.parameters():
    #         p.requires_grad = False
    #     for p in self.G.encoder.parameters():
    #         p.requies_grad = requires_grad
    #     for p in self.G.decoder.parameters():
    #         p.requies_grad = requires_grad        

    def forward(self):
        self.stylized_img = self.G(self.ori_img, self.style_img)
        # self.stylized_img = self.G(self.ori_img, self.ori_img)
        self.rec_img = self.G(self.stylized_img, self.style_ori_img)
        # self.rec_img = self.G(self.stylized_img, self.ori_img)


    def forward_AE(self):
        self.ae_img = self.G.decoder(self.G.encoder(self.ori_img))

    def backward_G(self):
        # self.loss_G_gen = self.criterionGAN(self.D(self.stylized_img, self.style_ref_img), True)
        # self.loss_G_gen = self.triplet_loss_G(self.D(self.stylized_img), self.D(self.style_img), self.D(self.ori_img))
        # self.loss_G_gen = torch.tensor(0.0).cuda()

        self.loss_G_gen = self.opt.style_weight*(self.style_loss_net(self.stylized_img, self.style_img) + self.style_loss_net(self.rec_img, self.ori_img))

        # self.loss_G_gen = self.style_loss_net(self.stylized_img, self.style_img)

        # self.loss_G_gen = self.l1loss(self.stylized_img, self.ori_img)
        self.loss_cycle = self.criterionCycle(self.rec_img, self.ori_img)
        self.loss_G = self.loss_G_gen + self.loss_cycle
        # self.loss_G = self.loss_G_gen 
        # self.loss_G = self.loss_cycle
        
        self.loss_G.backward()

    # def backward_D(self):
    #     # self.loss_D_same = self.criterionGAN(self.D(self.style_img, self.style_ref_img), True)
    #     # self.loss_D_diff = self.criterionGAN(self.D(self.stylized_img.detach(), self.style_ref_img), False)
    #     # self.loss_D = (self.loss_D_same + self.loss_D_diff) * 0.5
    #     self.loss_D = self.triplet_loss_D(self.D(self.style_img), self.D(self.style_ref_img), self.D(self.stylized_img.detach()))
        
    #     self.loss_D.backward()

    def backward_AE(self):
        self.loss_AE = self.criterionAE(self.ae_img, self.ori_img)
        self.loss_AE.backward()

    def optimize_parameters(self):
        # self.set_requires_grad(self.G, True)
        self.forward()

        # self.set_requires_grad(self.D, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        # self.set_requires_grad(self.D, True)
        # self.optimizer_D.zero_grad()
        # self.backward_D()
        # self.optimizer_D.step()
    
    def optimize_parameters_AE(self):
        # self.set_requires_grad_AE(True) wrong!
        self.set_requires_grad(self.G, True)

        self.forward_AE()

        self.optimizer_AE.zero_grad()
        self.backward_AE()
        self.optimizer_AE.step()

    # def optimize_parameters_D(self):
    #     self.set_requires_grad(self.D, True)
    #     self.optimizer_D.zero_grad()
    #     same = self.D(self.ori_img, self.style_ori_img)
    #     diff = self.D(self.ori_img, self.style_img)
    #     self.loss_D_same = self.criterionGAN(same, True)
    #     self.loss_D_diff = self.criterionGAN(diff, False)
    #     self.loss_D = (self.loss_D_same + self.loss_D_diff) * 0.5
    #     self.loss_D.backward()
    #     self.optimizer_D.step()

    # def optimize_parameters_D(self):
    #     self.set_requires_grad(self.D, True)
    #     self.optimizer_D.zero_grad()
    #     # same = self.D(self.ori_img, self.style_ori_img)
    #     # diff = self.D(self.ori_img, self.style_img)
    #     # self.loss_D_same = self.criterionGAN(same, True)
    #     # self.loss_D_diff = self.criterionGAN(diff, False)
    #     self.loss_D = self.triplet_loss_D(self.D(self.ori_img), self.D(self.style_ori_img), self.D(self.style_img))
    #     # self.loss_D.backward()
    #     # self.optimizer_D.step()

    def train(self, mode=True):
        self.G.train(mode)
        # self.D.train(mode)
        # self.vgg16.eval()

    def eval(self):
        self.G.eval()
        # self.D.eval()

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
            torch.save(self.G.state_dict(), G_save_path)
            self.G.cuda(self.gpu_ids[0])
        else:
            torch.save(self.G.cpu().state_dict(), G_save_path)

        # D_save_filename = prefix + '_ExtractGAN_D.pth' 
        # D_save_path = os.path.join(self.save_dir, D_save_filename)
        # if len(self.gpu_ids) > 0 and torch.cuda.is_available():
        #     torch.save(self.D.state_dict(), D_save_path)
        #     self.D.cuda(self.gpu_ids[0])
        # else:
        #     torch.save(self.D.cpu().state_dict(), D_save_path)
    
    # load models from the disk
    def load_networks(self, load_dir, prefix):
        G_load_filename = prefix
        G_load_path = os.path.join(load_dir, G_load_filename)
        print('loading the model from ' + G_load_path)
        G_state_dict = torch.load(G_load_path, map_location=self.device)
        if hasattr(G_state_dict, '_metadata'):
            del G_state_dict._metadata
        self.G.load_state_dict(G_state_dict)
    # def load_networks(self, load_dir, prefix):
    #     G_load_filename = prefix + '_G.pth'
    #     G_load_path = os.path.join(load_dir, G_load_filename)
    #     print('loading the model from ' + G_load_path)
    #     G_state_dict = torch.load(G_load_path, map_location=self.device)
    #     if hasattr(G_state_dict, '_metadata'):
    #         del G_state_dict._metadata
    #     self.G.load_state_dict(G_state_dict)

        # D_load_filename = prefix + '_D.pth'
        # D_load_path = os.path.join(load_dir, D_load_filename)
        # print('loading the model from ' + D_load_path)
        # D_state_dict = torch.load(D_load_path, map_location=self.device)
        # if hasattr(D_state_dict, '_metadata'):
        #     del D_state_dict._metadata
        # self.D.load_state_dict(D_state_dict)


    # load D from the disk
    # def load_D(self, load_dir, name):
    #     D_load_filename = name
    #     D_load_path = os.path.join(load_dir, D_load_filename)
    #     print('loading the model from ' + D_load_path)
    #     D_state_dict = torch.load(D_load_path, map_location=self.device)
    #     if hasattr(D_state_dict, '_metadata'):
    #         del D_state_dict._metadata
    #     self.D.load_state_dict(D_state_dict)

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
