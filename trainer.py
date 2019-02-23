#!/usr/bin/python
# -*- coding: utf-8 -*-
import cv2
import os
import time
import torch
import datetime
import numpy as np
import shutil
import math

import torch.nn as nn
from io import BytesIO
from torch.autograd import Variable
from torchvision.utils import save_image

from sagan_models import Generator, Discriminator
from utils import *


class Trainer(object):

    def __init__(self, data_loader, config):

        # Data loader

        self.data_loader = data_loader

        # exact model and loss

        self.model = config.model
        self.adv_loss = config.adv_loss

        # Model hyper-parameters

        self.imsize = config.imsize
        self.g_num = config.g_num
        self.z_dim = config.z_dim
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.parallel = config.parallel

        self.lambda_gp = config.lambda_gp
        self.total_step = config.total_step
        self.d_iters = config.d_iters
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.lr_decay = config.lr_decay
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.pretrained_model = config.pretrained_model

        self.dataset = config.dataset
        self.use_tensorboard = config.use_tensorboard
        self.image_path = config.image_path
        self.log_path = config.log_path
        self.model_save_path = config.model_save_path
        self.sample_path = config.sample_path
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.version = config.version

        # Path

        self.log_path = os.path.join(config.log_path, self.version)
        self.sample_path = os.path.join(config.sample_path,
                self.version)
        self.model_save_path = os.path.join(config.model_save_path,
                self.version)

        self.build_model()

        if self.use_tensorboard:
            self.build_tensorboard()

        # Start with trained model

        if self.pretrained_model:
            self.load_pretrained_model()

    def train(self):

        # Data iterator

        data_iter = iter(self.data_loader)
        step_per_epoch = len(self.data_loader)
        model_save_step = int(self.model_save_step * step_per_epoch)

        # Fixed input for debugging

        fixed_z = tensor2var(torch.normal(0, torch.ones([self.batch_size, self.z_dim])*3))

        # Start with trained model

        if self.pretrained_model:
            start = self.pretrained_model + 1
        else:
            start = 0

        # Start time

        start_time = time.time()
        i = 0
        for step in range(start, self.total_step):

            # ================== Train D ================== #

            self.D.train()
            self.G.train()

            try:
                (real_images, _) = next(data_iter)
            except:
                data_iter = iter(self.data_loader)
                (real_images, _) = next(data_iter)

            # Compute loss with real images
            # dr1, dr2, df1, df2, gf1, gf2 are attention scores

            real_images = tensor2var(real_images)
            d_out_real = self.D(real_images)
            if self.adv_loss == 'wgan-gp':
                d_loss_real = -torch.mean(d_out_real)
            elif self.adv_loss == 'hinge':
                d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()

            # apply Gumbel Softmax
            z = tensor2var(torch.normal(0, torch.ones([real_images.size(0), self.z_dim])*3))
            # (fake_images, gf1, gf2) = self.G(z)
            (fake_images, gf2) = self.G(z)

            if i < 1:
                print('***** Result Image size now *****')
                print(fake_images.size())
                # print(gf1.size())
                print(gf2.size())
            i = i + 1

            d_out_fake = self.D(fake_images)

            if self.adv_loss == 'wgan-gp':
                d_loss_fake = d_out_fake.mean()
            elif self.adv_loss == 'hinge':
                d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()

            # Backward + Optimize

            d_loss = d_loss_real + d_loss_fake
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            if self.adv_loss == 'wgan-gp':

                # Compute gradient penalty

                alpha = torch.rand(real_images.size(0), 1, 1,
                                   1).cuda().expand_as(real_images)
                interpolated = Variable(alpha * real_images.data + (1
                        - alpha) * fake_images.data, requires_grad=True)
                out = self.D(interpolated)

                grad = torch.autograd.grad(
                    outputs=out,
                    inputs=interpolated,
                    grad_outputs=torch.ones(out.size()).cuda(),
                    retain_graph=True,
                    create_graph=True,
                    only_inputs=True,
                    )[0]

                grad = grad.view(grad.size(0), -1)
                grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
                d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)

                # Backward + Optimize

                d_loss = self.lambda_gp * d_loss_gp

                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

            # ================== Train G and gumbel ================== #
            # Create random noise
            z = tensor2var(torch.normal(0, torch.ones([real_images.size(0), self.z_dim])*3))
            # (fake_images, _, _) = self.G(z)
            (fake_images, _) = self.G(z)

            # Compute loss with fake images

            g_out_fake = self.D(fake_images)  # batch x n
            if self.adv_loss == 'wgan-gp':
                g_loss_fake = -g_out_fake.mean()
            elif self.adv_loss == 'hinge':
                g_loss_fake = -g_out_fake.mean()

            self.reset_grad()
            g_loss_fake.backward()
            self.g_optimizer.step()

            # Print out log info

            if (step + 1) % self.log_step == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                print('Elapsed [{}], G_step [{}/{}], D_step[{}/{}], d_out_real: {:.4f}, ave_gamma_l4: {:.4f}'.format(
                    elapsed,
                    step + 1,
                    self.total_step,
                    step + 1,
                    self.total_step,
                    d_loss_real.data[0],
                    self.G.module.attn2.gamma.mean().data[0],
                    ))

        # (1) Log values of the losses (scalars)

                info = {
                    'd_loss_real': d_loss_real.data[0],
                    'd_loss_fake': d_loss_fake.data[0],
                    'd_loss': d_loss.data[0],
                    'g_loss_fake': g_loss_fake.data[0],
                    # 'ave_gamma_l3': self.G.module.attn1.gamma.mean().data[0],
                    'ave_gamma_l4': self.G.module.attn2.gamma.mean().data[0],
                    }

                for (tag, value) in info.items():
                    self.logger.scalar_summary(tag, value, step + 1)

            # Sample images / Save and log

            if (step + 1) % self.sample_step == 0:

                 # (2) Log values and gradients of the parameters (histogram)

                for (net, name) in zip([self.G, self.D], ['G_', 'D_']):
                    for (tag, value) in net.named_parameters():
                        tag = name + tag.replace('.', '/')
                        self.logger.histo_summary(tag,
                                value.data.cpu().numpy(), step + 1)

                # (3) Log the tensorboard

                info = \
                    {'fake_images': (fake_images.view(fake_images.size())[:
                     16, :, :, :]).data.cpu().numpy(),
                     'real_images': (real_images.view(real_images.size())[:
                     16, :, :, :]).data.cpu().numpy()}


                # (fake_images, at1, at2) = self.G(fixed_z)
                (fake_images, at2) = self.G(fixed_z)
                if (step + 1) % (self.sample_step * 10) == 0:
                    save_image(denorm(fake_images.data),
                            os.path.join(self.sample_path,
                            '{}_fake.png'.format(step + 1)))

                # print('***** Fake Image size now *****')
                # print('fake_images ', fake_images.size())
                # print('at2 ', at2.size())   # B * N * N
                # at2_4d = at2.view(*(at2.size()[0], at2.size()[1], int(math.sqrt(at2.size()[2])), int(math.sqrt(at2.size()[2])))) # W * N * W * H
                # # print('at2_4d ', at2_4d.size())
                # at2_mean = at2_4d.mean(dim=1,keepdim=False) # B * W * H
                # print('at2_mean ', at2_mean.size())

                # print('***** start create activation map *****')
                # attn_list = []
                # for i in range(at2.size()[0]):
                #     # print('fake_images size: ',fake_images[i].size())
                #     # print('at2 mean size', at2_mean[i].size())
                #
                #     f = BytesIO()
                #     img = np.uint8(np.zeros(at2_mean[i,:,:].size()[0],at2_mean[i,:,:].size()[1],3))
                #     a = np.uint8(at2_mean[i,:,:].mul(255).data.cpu().numpy())
                #     # print('image: ', img.shape)
                #     # print('a shape: ',a.shape)
                #
                #     # im_image = img.reshape(img.shape[1],img.shape[2],img.shape[0])
                #     im_image = img
                #     im_attn = cv2.applyColorMap(a, cv2.COLORMAP_JET)
                #
                #     img_with_heatmap = np.float32(im_attn) + np.float32(im_image)
                #     img_with_heatmap = img_with_heatmap / np.max(img_with_heatmap)
                #
                #     attn_np = np.uint8((255 * img_with_heatmap).reshape(img_with_heatmap.shape[2],img_with_heatmap.shape[0],img_with_heatmap.shape[1]))
                #     attn_torch = torch.from_numpy(attn_np)
                #     # print('final attn image size: ', attn_torch.size())
                #     attn_list.append(attn_torch.unsqueeze(0))
                #
                # attn_images = torch.cat(attn_list)
                # print('attn images list: ',attn_images.size())
                # info['attn_images'] = (attn_images.view(attn_images.size())[:16, :, :, :]).numpy()


                for (tag, image) in info.items():
                    self.logger.image_summary(tag, image, step + 1)

            if (step + 1) % model_save_step == 0:
                torch.save(self.G.state_dict(),
                           os.path.join(self.model_save_path,
                           '{}_G.pth'.format(step + 1)))
                torch.save(self.D.state_dict(),
                           os.path.join(self.model_save_path,
                           '{}_D.pth'.format(step + 1)))

    def build_model(self):

        self.G = Generator(self.batch_size, self.imsize, self.z_dim,
                           self.g_conv_dim).cuda()
        self.D = Discriminator(self.batch_size, self.imsize,
                               self.d_conv_dim).cuda()
        if self.parallel:
            self.G = nn.DataParallel(self.G)
            self.D = nn.DataParallel(self.D)

        # Loss and optimizer
        # self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])

        self.g_optimizer = torch.optim.Adam(filter(lambda p: \
                p.requires_grad, self.G.parameters()), self.g_lr,
                [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(filter(lambda p: \
                p.requires_grad, self.D.parameters()), self.d_lr,
                [self.beta1, self.beta2])

        self.c_loss = torch.nn.CrossEntropyLoss()

        # print networks

        print(self.G)
        print(self.D)

    def build_tensorboard(self):
        from logger import Logger
        #if os.path.exists(self.log_path):
        #    shutil.rmtree(self.log_path)
        #os.makedirs(self.log_path)
        self.logger = Logger(self.log_path)

    def load_pretrained_model(self):
        self.G.load_state_dict(torch.load(os.path.join(self.model_save_path,
                               '{}_G.pth'.format(self.pretrained_model))))
        self.D.load_state_dict(torch.load(os.path.join(self.model_save_path,
                               '{}_D.pth'.format(self.pretrained_model))))
        print('loaded trained models (step: {})..!'.format(self.pretrained_model))

    def reset_grad(self):
        self.d_optimizer.zero_grad()
        self.g_optimizer.zero_grad()

    def save_sample(self, data_iter):
        (real_images, _) = next(data_iter)
        save_image(denorm(real_images), os.path.join(self.sample_path,
                   'real.png'))

    def save_gradient_images(self, gradient, file_name):
        """
            Exports the original gradient image

        Args:
            gradient (np arr): Numpy array of the gradient with shape (3, 224, 224)
            file_name (str): File name to be exported
        """
        if not os.path.exists('attn2/results'):
            os.makedirs('attn2/results')
        # Normalize
        gradient = gradient - gradient.min()
        gradient /= gradient.max()
        # Save image
        path = os.path.join('attn2/results', file_name + '.jpg')
        im = gradient
        if isinstance(im, np.ndarray):
            if len(im.shape) == 2:
                im = np.expand_dims(im, axis=0)
            if im.shape[0] == 1:
                # Converting an image with depth = 1 to depth = 3, repeating the same values
                # For some reason PIL complains when I want to save channel image as jpg without
                # additional format in the .save()
                im = np.repeat(im, 3, axis=0)
                # Convert to values to range 1-255 and W,H, D
            if im.shape[0] == 3:
                im = im.transpose(1, 2, 0) * 255
            im = Image.fromarray(im.astype(np.uint8))
        im.save(path)
