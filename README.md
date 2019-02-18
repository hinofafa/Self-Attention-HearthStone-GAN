# Self-Attention GAN
**[Han Zhang, Ian Goodfellow, Dimitris Metaxas and Augustus Odena, "Self-Attention Generative Adversarial Networks." arXiv preprint arXiv:1805.08318 (2018)](https://arxiv.org/abs/1805.08318).**

## Meta overview
This repository provides a PyTorch implementation of [SAGAN](https://arxiv.org/abs/1805.08318). Both wgan-gp and wgan-hinge loss are ready, but note that wgan-gp is somehow not compatible with the spectral normalization. Remove all the spectral normalization at the model for the adoption of wgan-gp.

Self-attentions are applied to later two layers of both discriminator and generator.

<p align="center"><img width="100%" src="image/main_model.PNG" /></p>

## Current update status
* [ ] Supervised setting
* [x] Multi-gpu
* [x] Tensorboard loggings
* [x] **[20180608] updated the self-attention module. Thanks to my colleague [Cheonbok Park](https://github.com/cheonbok94)! see 'sagan_models.py' for the update. Should be efficient, and run on large sized images**
* [ ] Attention visualization
* [x] Unsupervised setting (use no label yet)
* [x] Applied: [Spectral Normalization](https://arxiv.org/abs/1802.05957), code from [here](https://github.com/christiancosgrove/pytorch-spectral-normalization-gan)
* [x] Implemented: self-attention module, two-timescale update rule (TTUR), wgan-hinge loss, wgan-gp loss

&nbsp;
&nbsp;

## Results

### Attention result on LSUN (epoch #8)
<p align="center"><img width="100%" src="image/sagan_attn.png" /></p>
Per-pixel attention result of SAGAN on LSUN church-outdoor dataset. It shows that unsupervised training of self-attention module still works, although it is not interpretable with the attention map itself. Better results with regard to the generated images will be added. These are the visualization of self-attention in generator layer3 and layer4, which are in the size of 16 x 16 and 32 x 32 respectively, each for 64 images. To visualize the per-pixel attentions, only a number of pixels are chosen, as shown on the leftmost and the rightmost numbers indicate.

### CelebA dataset (epoch on the left, still under training)
<p align="center"><img width="80%" src="image/sagan_celeb.png" /></p>

### LSUN church-outdoor dataset (epoch on the left, still under training)
<p align="center"><img width="70%" src="image/sagan_lsun.png" /></p>

## Prerequisites
* [Python 3.5+](https://www.continuum.io/downloads)
* [PyTorch 0.3.0](http://pytorch.org/)

&nbsp;

## Usage

#### 1. Clone the repository
```bash
$ git clone https://github.com/heykeetae/Self-Attention-GAN.git
$ cd Self-Attention-GAN
```

#### 2. Install datasets (CelebA or LSUN or Hearthstone)
```bash
$ bash download.sh CelebA (404 not found)
or
$ bash download.sh LSUN
```


#### 3. Train
##### (i) Train
```bash
$ python main.py --batch_size 64 --imsize 64 --dataset celeb --adv_loss hinge --version sagan_celeb
or
$ python main.py --batch_size 64 --imsize 64 --dataset lsun --adv_loss hinge --version sagan_lsun
```

Advanced training:
```bash
python main.py --batch_size 16 --imsize 128 --dataset hearthstone --adv_loss hinge --version sagan_hearth_ --num_workers 16 --use_tensorboard True --parallel True --log_path ./logs2 --model_save_path ./models2  --attn ./attn2 --sample_path ./samples2 --total_step 200000 --log_step 100
```
#### 4. Enjoy the results
```bash
$ cd samples/sagan_celeb
or
$ cd samples/sagan_lsun

```
Samples generated every 100 iterations are located. The rate of sampling could be controlled via --sample_step (ex, --sample_step 100).
