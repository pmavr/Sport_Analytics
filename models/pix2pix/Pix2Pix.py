
import torch
import torch.nn as nn
import functools
from torch.optim import lr_scheduler
from torch.autograd import Variable

from models.pix2pix.ImagePool import ImagePool


def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        nn.init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal(m.weight.data, 1.0, 0.02)
        nn.init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal(m.weight.data, 1.0, 0.02)
        nn.init.constant(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer):
    niter = 100
    niter_decay = 100
    epoch_count = 1

    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch + 1 + epoch_count - niter) / float(niter_decay + 1)
        return lr_l
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    return scheduler


class Pix2Pix(nn.Module):

    def __init__(self, is_train=True):
        super().__init__()
        self.is_train = is_train
        self.input_shape = (256, 256, 3)
        self.lambda_a = 100.

        self.generator = self._define_generator(
            input_nc=3, output_nc=1, ngf=64, norm='batch', use_dropout=True, init_type='normal')

        self.discriminator = self._define_discriminator(
            input_nc=4, ndf=64, n_layers_D=3, norm='batch', use_sigmoid=True, init_type='normal')

        self.fake_AB_pool = ImagePool(pool_size=0)


        print('---------- Networks initialized -------------')
        self.print_network(self.generator)
        if self.is_train:
            self.print_network(self.discriminator)
        print('-----------------------------------------------')

    def _define_generator(self, input_nc, output_nc, ngf, norm='batch', use_dropout=False, init_type='normal'):
        norm_layer = get_norm_layer(norm_type=norm)

        netG = UnetGenerator(
            input_nc, output_nc, num_downs=8, ngf=ngf,
            norm_layer=norm_layer,
            use_dropout=use_dropout)
        init_weights(netG, init_type=init_type)
        return netG

    def _define_discriminator(self, input_nc, ndf, n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal'):
        norm_layer = get_norm_layer(norm_type=norm)

        netD = NLayerDiscriminator(
            input_nc, ndf, n_layers=n_layers_D,
            norm_layer=norm_layer, use_sigmoid=use_sigmoid)
        init_weights(netD, init_type=init_type)
        return netD

    @staticmethod
    def print_network(net):
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        print(net)
        print('Total number of parameters: %d' % num_params)

    def forward(self, input_, output_):
        real_input = Variable(input_)
        fake_output = self.generator(real_input)
        real_output = Variable(output_)
        return real_output, fake_output

    def backward_discriminator(self, real_input, real_output, fake_output, gan_criterion):
        # Fake
        fake_AB = self.fake_AB_pool.query(torch.cat((real_input, fake_output), 1).data)
        pred_fake = self.discriminator(fake_AB.detach())
        loss_discriminator_fake = gan_criterion(input=pred_fake, target_is_real=False)

        # Real
        real_AB = torch.cat((real_input, real_output), 1)
        pred_real = self.discriminator(real_AB)
        loss_discriminator_real = gan_criterion(input=pred_real, target_is_real=True)

        # Combined loss
        loss_discriminator = (loss_discriminator_fake + loss_discriminator_real) * 0.5
        loss_discriminator.backward()

        return loss_discriminator_real, loss_discriminator_fake

    def backward_generator(self, real_input, real_output, fake_output, gan_criterion, l1_criterion):
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((real_input, fake_output), 1)
        pred_fake = self.discriminator(fake_AB)
        loss_generator_gan = gan_criterion(pred_fake, True)

        # Second, G(A) = B
        loss_generator_l1 = l1_criterion(fake_output, real_output) * self.lambda_a

        loss_generator = loss_generator_gan + loss_generator_l1
        loss_generator.backward()

        return loss_generator_gan, loss_generator_l1

    def infer(self, input_):
        real_input = Variable(input_)
        fake_output = self.generator(real_input)
        return fake_output

    @staticmethod
    def load_model(filename, model=None,
                   gen_optimizer=None, discr_optimizer=None,
                   gen_sched=None, discr_sched=None,
                   history=None):
        """Load trained model along with its optimizer and training, plottable history."""
        model_components = torch.load(filename)
        if model:
            model.load_state_dict(model_components['model'])
        if gen_optimizer:
            gen_optimizer.load_state_dict(model_components['generator_opt_func'])
        if gen_sched:
            gen_sched.load_state_dict(model_components['generator_sched_func'])
        if discr_optimizer:
            discr_optimizer.load_state_dict(model_components['discriminator_opt_func'])
        if discr_sched:
            discr_sched.load_state_dict(model_components['discriminator_sched_func'])
        if history:
            history = model_components['history']
        return model, gen_optimizer, discr_optimizer, gen_sched, discr_sched, history


class UnetGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)
