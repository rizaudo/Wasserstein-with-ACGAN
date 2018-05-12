import matplotlib
import chainer
import chainer.links as L
import chainer.functions as F
import chainer.training as training
import chainer.training.extensions as extensions
import chainer.optimizers as optimizers
from chainer import Variable
from pathlib import Path
# import chainer.computational_graph as cg
import os
from PIL import Image
import argparse
import numpy as np
# import cupy as cp
matplotlib.use('Agg')


# ここは元のGと違う。こっちはDCGAN使ってる
class Generator(chainer.Chain):
    def __init__(self, out_ch=3, n_hidden=100, label_num=None, bottom_width=4, ch=512, wscale=0.02, use_bn=True, distribution='normal'):
        super(Generator, self).__init__()
        self.use_bn = use_bn
        self.bottom_width = bottom_width
        self.hidden_activation = F.leaky_relu
        self.output_activation = F.tanh
        self.ch = ch
        self.n_hidden = n_hidden
        self.label_num = label_num
        if distribution not in ['normal', 'uniform']:
            raise ValueError('unknown z distribution: %s' % self.distribution)
        self.distribution = distribution

        with self.init_scope():
            w = chainer.initializers.Normal(wscale)
            self.l0 = L.Linear(self.n_hidden, bottom_width * bottom_width * ch,
                               initialW=w)
            self.dc1 = L.Deconvolution2D(ch, ch // 2, 4, 2, 1, initialW=w)
            self.dc2 = L.Deconvolution2D(ch // 2, ch // 4, 4, 2, 1, initialW=w)
            self.dc3 = L.Deconvolution2D(ch // 4, ch // 8, 4, 2, 1, initialW=w)
            self.dc4 = L.Deconvolution2D(ch // 8, out_ch, 3, 1, 1, initialW=w)
            if self.use_bn:
                self.bn0 = L.BatchNormalization(bottom_width * bottom_width * ch)
                self.bn1 = L.BatchNormalization(ch // 2)
                self.bn2 = L.BatchNormalization(ch // 4)
                self.bn3 = L.BatchNormalization(ch // 8)

    def make_input_z(self, batchsize):
        # size = 512
        if self.distribution == 'normal':
            return np.random.randn(batchsize, self.n_hidden, 1, 1).astype(np.float32)
        elif self.distribution == 'uniform':
            return np.random.uniform(-1, 1, (batchsize, self.n_hidden, 1, 1)).astype(np.float32)

        else:
            raise ValueError('unknown z distribution: %s' % self.distribution)

    def make_input_z_with_label(self, batchsize: int, labelbatch: np.array):
        # labelbatch is 1d array
        # onehot representation
        xp = self.xp
        targets = labelbatch.reshape(-1)
        onehot = xp.eye(self.label_num)[targets]
        onehot = onehot.reshape(batchsize, self.label_num, 1, 1)

        if self.distribution == 'normal':
            nikome = xp.random.randn(batchsize, self.n_hidden - self.label_num, 1, 1)
            return xp.concatenate((onehot, nikome), axis=1).astype(np.float32)

        elif self.distribution == 'uniform':
            nikome = xp.random.uniform(-1, 1, (batchsize, self.n_hidden - self.label_num, 1, 1))
            return xp.concatenate((onehot, nikome), axis=1).astype(np.float32)

        else:
            raise ValueError('unknown z distribution: %s' % self.distribution)

    def __call__(self, z):
        if not self.use_bn:
            h = F.reshape(self.hidden_activation(self.l0(z)),
                          (len(z), self.ch, self.bottom_width, self.bottom_width))
            h = self.hidden_activation(self.dc1(h))
            h = self.hidden_activation(self.dc2(h))
            h = self.hidden_activation(self.dc3(h))
            x = self.output_activation(self.dc4(h))
        else:
            h = F.reshape(self.hidden_activation(self.bn0(self.l0(z))),
                          (len(z), self.ch, self.bottom_width, self.bottom_width))
            h = self.hidden_activation(self.bn1(self.dc1(h)))
            h = self.hidden_activation(self.bn2(self.dc2(h)))
            h = self.hidden_activation(self.bn3(self.dc3(h)))
            x = self.output_activation(self.dc4(h))
        return x


# Gに同じく変えてる
class Critic(chainer.Chain):
    def __init__(self, label_num=0, batch_num=None, ch=512, bottom_width=4, wscale=0.02, output_dim=1):
        super(Critic, self).__init__()
        self.label_num = label_num
        self.batch_num = batch_num
        with self.init_scope():
            w = chainer.initializers.Normal(wscale)
            self.l0 = L.Linear(self.label_num + 3 * 32 * 32, 3 * 32 * 32, initialW=w)
            self.c0 = L.Convolution2D(3, ch // 8, 3, 1, 1, initialW=w)
            self.c1 = L.Convolution2D(ch // 8, ch // 4, 4, 2, 1, initialW=w)
            self.c1_0 = L.Convolution2D(ch // 4, ch // 4, 3, 1, 1, initialW=w)
            self.c2 = L.Convolution2D(ch // 4, ch // 2, 4, 2, 1, initialW=w)
            self.c2_0 = L.Convolution2D(ch // 2, ch // 2, 3, 1, 1, initialW=w)
            self.c3 = L.Convolution2D(ch // 2, ch // 1, 4, 2, 1, initialW=w)
            self.c3_0 = L.Convolution2D(ch // 1, ch // 1, 3, 1, 1, initialW=w)
            self.l4 = L.Linear(bottom_width * bottom_width * ch, output_dim, initialW=w)

    def concat_label(self, x, label):
        xp = self.xp
        y = label.reshape(-1)
        y = xp.eye(self.label_num)[y]
        y = y.reshape((self.batch_num, self.label_num, 1, 1))
        y = y * xp.ones((self.batch_num, self.label_num, 32, 32))
        return xp.concatenate((x, y), axis=1).astype(np.float32)

    def new_concat_label(self, x, label):
        xp = self.xp
        y = label.reshape(-1)
        y = xp.eye(self.label_num)[y]
        y = y.reshape((self.batch_num, self.label_num))
        return xp.concatenate((x, y), axis=1).astype(np.float32)

    def __call__(self, x: Variable, y: Variable):
        hoge = F.reshape(x, (self.batch_num, -1))
        hoge = self.new_concat_label(hoge.data, y.data)
        hoge = self.l0(hoge)
        hoge = F.reshape(hoge, (self.batch_num, 3, 32, 32))
        self.h0 = F.leaky_relu(self.c0(hoge))
        self.h1 = F.leaky_relu(self.c1(self.h0))
        self.h2 = F.leaky_relu(self.c1_0(self.h1))
        self.h3 = F.leaky_relu(self.c2(self.h2))
        self.h4 = F.leaky_relu(self.c2_0(self.h3))
        self.h5 = F.leaky_relu(self.c3(self.h4))
        self.h6 = F.leaky_relu(self.c3_0(self.h5))
        return self.l4(self.h6)


class NonLabelCritic(chainer.Chain):
    def __init__(self, init_ch=3, ch=512, bottom_width=4, wscale=0.02, output_dim=1):
        super(NonLabelCritic, self).__init__()
        with self.init_scope():
            w = chainer.initializers.Normal(wscale)
            self.c0 = L.Convolution2D(init_ch, ch // 8, 3, 1, 1, initialW=w)
            self.c1 = L.Convolution2D(ch // 8, ch // 4, 4, 2, 1, initialW=w)
            self.c1_0 = L.Convolution2D(ch // 4, ch // 4, 3, 1, 1, initialW=w)
            self.c2 = L.Convolution2D(ch // 4, ch // 2, 4, 2, 1, initialW=w)
            self.c2_0 = L.Convolution2D(ch // 2, ch // 2, 3, 1, 1, initialW=w)
            self.c3 = L.Convolution2D(ch // 2, ch // 1, 4, 2, 1, initialW=w)
            self.c3_0 = L.Convolution2D(ch // 1, ch // 1, 3, 1, 1, initialW=w)
            self.l4 = L.Linear(bottom_width * bottom_width * ch, output_dim, initialW=w)

    def __call__(self, x: Variable):
        self.h0 = F.leaky_relu(self.c0(x))
        self.h1 = F.leaky_relu(self.c1(self.h0))
        self.h2 = F.leaky_relu(self.c1_0(self.h1))
        self.h3 = F.leaky_relu(self.c2(self.h2))
        self.h4 = F.leaky_relu(self.c2_0(self.h3))
        self.h5 = F.leaky_relu(self.c3(self.h4))
        self.h6 = F.leaky_relu(self.c3_0(self.h5))
        return self.l4(self.h6)


class Classfier(chainer.Chain):
    def __init__(self, init_ch=3, y_dim=10):
        super(Classfier, self).__init__()
        self.y_dim = y_dim
        size = 32
        with self.init_scope():
            self.c0 = L.Convolution2D(init_ch, size, ksize=5, stride=1)
            self.c1 = L.Convolution2D(size, size * 2, ksize=5, stride=2)
            self.l0 = L.Linear(None, out_size=1024)
            self.l1 = L.Linear(1024, out_size=y_dim)

    def __call__(self, x):
        cat = F.relu(self.c0(x))
        cat = F.max_pooling_2d(cat, ksize=2, stride=2)
        cat = F.relu(self.c1(cat))
        cat = F.max_pooling_2d(cat, ksize=2, stride=2)
        # cat = F.flatten(cat)
        cat = F.relu(self.l0(cat))
        cat = self.l1(cat)

        return cat


class LayerMLP(chainer.Chain):
    def __init__(self, batchsize, n_hidden=100,  label_num=10, distribution='uniform', wscale=0.02):
        super(LayerMLP, self).__init__()
        self.n_hidden = n_hidden
        self.label_num = label_num
        self.distribution = distribution
        self.batchsize = batchsize
        with self.init_scope():
            w = chainer.initializers.Normal(wscale)
            self.l0 = L.Linear(None, 256, initialW=w)
            self.l1 = L.Linear(256, 512, initialW=w)
            self.l2 = L.Linear(512, 256, initialW=w)
            self.l3 = L.Linear(256, 28 * 28, initialW=w)

    def make_input_z_with_label(self, batchsize: int, labelbatch: np.array):
        # labelbatch is 1d array
        # onehot representation
        xp = self.xp
        targets = labelbatch.reshape(-1)
        onehot = xp.eye(self.label_num)[targets]
        onehot = onehot.reshape(batchsize, self.label_num, 1, 1)

        if self.distribution == 'normal':
            nikome = xp.random.randn(batchsize, self.n_hidden - self.label_num, 1, 1)
            return xp.concatenate((onehot, nikome), axis=1).astype(np.float32)

        elif self.distribution == 'uniform':
            nikome = xp.random.uniform(-1, 1, (batchsize, self.n_hidden - self.label_num, 1, 1))
            return xp.concatenate((onehot, nikome), axis=1).astype(np.float32)

        else:
            raise ValueError('unknown z distribution: %s' % self.distribution)

    def __call__(self, x):
        self.x = F.relu(self.l0(x))
        self.x = F.relu(self.l1(self.x))
        self.x = F.relu(self.l2(self.x))
        self.x = self.l3(self.x)
        return self.x


class LayerMLPCritic(chainer.Chain):
    def __init__(self, wscale=0.02):
        super(LayerMLPCritic, self).__init__()
        with self.init_scope():
            w = chainer.initializers.Normal(wscale)

            self.l0 = L.Linear(None, 256, initialW=w)
            self.l1 = L.Linear(256, 512, initialW=w)
            self.l2 = L.Linear(512, 256, initialW=w)
            self.l3 = L.Linear(256, 1, initialW=w)

    def __call__(self, x):
        self.x = F.relu(self.l0(x))
        self.x = F.relu(self.l1(self.x))
        self.x = F.relu(self.l2(self.x))
        self.x = self.l3(self.x)
        return self.x


class LayerMLPClassifier(chainer.Chain):
    def __init__(self, label_num=10, wscale=0.02):
        super(LayerMLPClassifier, self).__init__()
        with self.init_scope():
            w = chainer.initializers.Normal(wscale)
            self.l0 = L.Linear(None, 256, initialW=w)
            self.l1 = L.Linear(256, 512, initialW=w)
            self.l2 = L.Linear(512, 256, initialW=w)
            self.l3 = L.Linear(256, label_num, initialW=w)

    def __call__(self, x):
        self.x = F.relu(self.l0(x))
        self.x = F.relu(self.l1(self.x))
        self.x = F.relu(self.l2(self.x))
        self.x = self.l3(self.x)
        return self.x


class Updater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.generator, self.critic, self.classifier = kwargs.pop('models')
        self.n_critic = kwargs.pop('n_critic')
        self.n_classify = kwargs.pop('n_classify')
        self.l = 10
        super(Updater, self).__init__(*args, **kwargs)

    def update_core(self):
        # TODO: support n_Classfier <- いる？
        # TIPS: in case of experiments, set n_critic as 5 is best result.
        gen_optimizer = self.get_optimizer('gen')
        critic_optimizer = self.get_optimizer('critic')
        clfr_optimizer = self.get_optimizer('classfier')
        xp = self.generator.xp

        for i in range(self.n_critic):
            # grab data
            batch = self.get_iterator('main').next()
            batchsize = len(batch)
            batch = self.converter(batch, self.device)
            real_data, real_label = batch
            real_label = Variable(real_label)
            real_data = Variable(real_data) / 255.



            # TODO: cWGANってuniformで良いんだっけ...?
            z = Variable(xp.asarray(self.generator.make_input_z_with_label(batchsize, real_label.data)))

            # Generator
            gen_data = self.generator(z)
            gen_data = gen_data.reshape(batchsize, 1, 28, 28)

            # Critic(Discrimintor)
            critic_real = self.critic(real_data)
            critic_fake = self.critic(gen_data)

            # Classifier
            # classifier_real = self.classifier(real_data)
            # classifier_fake = self.classifier(gen_data)

            # Loss
            ## Critic Loss
            # print(critic_fake.shape, critic_real.shape, gen_data.shape, real_data.shape)
            # critic_loss = F.mean(critic_fake - critic_real)

            e = xp.random.uniform(0., 1., (batchsize, 1, 1, 1)).astype(np.float32)
            x_hat = e * real_data + (1 - e) * gen_data  # recreate Variable

            loss_gan = F.average(critic_fake - critic_real)
            # x_hat.backward(retain_grad=True, enable_double_backprop=True)
            grad, = chainer.grad([self.critic(x_hat)], [x_hat],
                                 enable_double_backprop=True)
            grad = F.sqrt(F.batch_l2_norm_squared(grad))

            loss_grad = self.l * F.mean_absolute_error(grad,
                                                       xp.ones_like(grad.data))

            critic_loss = loss_gan + loss_grad

            self.critic.cleargrads()
            critic_loss.backward()
            critic_optimizer.update()
            chainer.report({'critic_loss': critic_loss})
            chainer.report({'loss_grad': loss_grad})
            chainer.report({'loss_gan': loss_gan})

        batch = self.get_iterator('main').next()
        batchsize = len(batch)
        batch = self.converter(batch, self.device)
        real_data, real_label = batch
        real_label = Variable(real_label)
        real_data = Variable(real_data) / 255.

        z = Variable(xp.asarray(self.generator.make_input_z_with_label(batchsize, real_label.data)))

        # Generator
        gen_data = self.generator(z)

        # Critic(Discrimintor)
        critic_fake = self.critic(gen_data)

        # Classifier
        classifier_real = self.classifier(real_data)
        classifier_fake = self.classifier(gen_data)

        ## Categorical Loss
        c_f_loss = F.softmax_cross_entropy(classifier_fake, real_label)
        c_r_loss = F.softmax_cross_entropy(classifier_real, real_label)
        c_loss = (c_r_loss + c_f_loss) / 2

        self.classifier.cleargrads()
        c_loss.backward()
        clfr_optimizer.update()
        chainer.report({'c_r_loss': c_r_loss})
        chainer.report({'c_loss': c_loss})

        #  Generator Loss
        gen_loss = F.average(-critic_fake)

        self.generator.cleargrads()
        gen_loss.backward()
        gen_optimizer.update()
        chainer.report({'gen_loss': gen_loss})

        self.classifier.cleargrads()
        c_f_loss.backward()
        gen_optimizer.update()
        chainer.report({'c_f_loss': c_f_loss})


def out_generated_image(gen: Generator, dis: Critic, label_num: int, rows: int, cols: int, dst: str):
    @chainer.training.make_extension()
    def make_image(trainer):
        n_images = rows * cols
        labels = np.random.choice(label_num, n_images)
        xp = chainer.cuda.get_array_module(gen)
        z = Variable(gen.make_input_z_with_label(n_images, labels))
        x = gen(z)
        x = chainer.cuda.to_cpu(x.data)
        x = x.reshape((n_images, 1, 28, 28))

        x = np.asarray(np.clip(x * 255, 0.0, 255.0), dtype=np.uint8)
        _, ch, H, W = x.shape
        x = x.reshape((rows, cols, ch, H, W))
        x = x.transpose(0, 3, 1, 4, 2)
        x = x.reshape((rows * H, cols * W, ch))

        preview_dir = '{}/preview'.format(dst)
        preview_path = preview_dir + '/image{:0>8}.png'.format(trainer.updater.iteration)
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)

        x = np.squeeze(x, axis=2)
        Image.fromarray(x, mode='L').save(preview_path)
    return make_image


def main():
    # We Divided categorical loss calc and Wasserstein loss calc.
    parser = argparse.ArgumentParser(description='Condtitional WGAN in Chainer')
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='Integer of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=1,
                        help='Integer of Epochs')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory of output result')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='Gpu number')
    parser.add_argument('--resume', '-r', default='',
                        help='start training from snapshot')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed of z at visualization stage')
    parser.add_argument('--imagedir', '-dir', default=None,
                        help="Directory of image dir")
    parser.add_argument('--ncritic', '-nc', default=5,
                        help='n_critic')
    parser.add_argument('--clamp', default=0.01,
                        help='bound of weight clipping in critic')
    parser.add_argument('--debug', default=False,
                        help='chainer debug mode')
    parser.add_argument('--distribution', '-dist', default='uniform',
                        help='noise z sampling distribution')
    parser.add_argument('--mnist', action='store_true')
    args = parser.parse_args()

    print(args.mnist)
    if args.debug:
        chainer.set_debug(True)

    # model setup
    if args.mnist is True:
        generator = LayerMLP(batchsize=args.batchsize, n_hidden=110, label_num=10)
        critic = LayerMLPCritic()
        classifier = LayerMLPClassifier()
    else:
        generator = Generator(n_hidden=110, label_num=10, distribution=args.distribution)
        critic = NonLabelCritic()
        classifier = Classfier(y_dim=10)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        generator.to_gpu()
        critic.to_gpu()
        classifier.to_gpu()

    # optimizer setup
    # optimizer is need quite tough decision.
    def make_optimizer(model, alpha=0.0002, beta1=0.5, beta2=0.9):
        optimizer = optimizers.Adam(alpha=alpha, beta1=beta1, beta2=beta2)
        optimizer.setup(model)
        # ACGAN Does not using WeightDecay or something like it
        # optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001), 'hook_dec')
        return optimizer

    def make_wgan_gp_optimizer(model, alpha=1e-4, beta1=0.5, beta2=0.9):
        optimizer = optimizers.Adam(alpha=alpha, beta1=beta1, beta2=beta2)
        optimizer.setup(model)

        # donot use WeightDecay
        return optimizer

    def make_optimizer_SGD(model, lr=0.01):
        optimizer = optimizers.SGD(lr)
        optimizer.setup(model)

        return optimizer

    def make_optimizer_RMS(model, lr=0.0002):
        optimizer = optimizers.RMSprop(lr)
        optimizer.setup(model)

        return optimizer

    # we follow WGAN as CWGAN does.
    opt_gen = make_wgan_gp_optimizer(generator)
    opt_critic = make_wgan_gp_optimizer(critic)
    # following ACGAN Parameter
    opt_classifier = make_optimizer(classifier, alpha=2e-4, beta1=0.5, beta2=0.999)

    # dataset setup
    if args.imagedir is None:
        # if imagedir not given, use cifar-10 or mnist
        if args.mnist is True:
            train, _ = chainer.datasets.get_mnist(withlabel=True, ndim=3, scale=255.)
        else:
            train, _ = chainer.datasets.get_cifar10(withlabel=True, scale=255.)
    else:
        #  TODO: change from ImageDataset to Labeled Dataset.
        p = Path(args.imagedir)
        img_tuple = {}
        label_num = 0
        datapathstore = {}

        labeldirs = [x for x in p.iterdir() if x.is_dir()]
        for label in labeldirs:
            img_tuple[label.name] = label_num
            datapathstore[label.name] = map(lambda x: (x,label_num), [str for str in label.iterdir() if str.is_file()])

        datalist = []
        for str in datapathstore.items():
            datalist = datalist + str

        train = chainer.datasets.ImageDataset(datalist)
        # train *= 1. / 255.

    # *er setup
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)

    updater = Updater(
        models=(generator, critic, classifier),
        iterator=train_iter,
        optimizer={
            'gen': opt_gen, 'critic': opt_critic, 'classfier': opt_classifier
        },
        n_critic=args.ncritic,
        n_classify=args.ncritic,
        device=args.gpu
    )
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # extentions
    snapshot_interval = (1000, 'iteration')
    display_interval = (100, 'iteration')

    trainer.extend(
        extensions.snapshot(filename='snapshot_iter_{.updater.iteration}.npz'),
        trigger=snapshot_interval
    )
    trainer.extend(
        extensions.snapshot_object(
            generator, 'gen_iter_{.updater.iteration}.npz'),
        trigger=snapshot_interval
    )
    trainer.extend(
        extensions.snapshot_object(
            critic, 'critic_iter_{.updater.iteration}.npz'),
        trigger=snapshot_interval
    )
    trainer.extend(
        extensions.snapshot_object(
            classifier, 'classifier_iter_{.updater.iteration}.npz'
        ),
        trigger=snapshot_interval
    )
    trainer.extend(extensions.LogReport(trigger=display_interval))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'gen_loss', 'critic_loss', 'c_f_loss', 'c_r_loss', 'c_loss']
        ),
        trigger=display_interval
    )
    trainer.extend(extensions.ProgressBar(update_interval=20))
    # trainer.extend(
    #     extensions.dump_graph(args.out)
    # )
    trainer.extend(
        extensions.PlotReport(['gen_loss'],
                              'iteration', file_name='genloss.png', trigger=snapshot_interval)
    )
    trainer.extend(
        extensions.PlotReport(['critic_loss', 'loss_gan', 'loss_grad'],
                              'iteration', file_name='criticloss.png', trigger=snapshot_interval)
    )
    trainer.extend(
        extensions.PlotReport(['c_f_loss', 'c_r_loss', 'c_loss'],
                              'iteration', file_name='loss.png', trigger=snapshot_interval)
    )
    trainer.extend(
        out_generated_image(generator, critic, 10, 10, 10, dst=args.out),
        trigger=display_interval
    )

    # FLY TO THE FUTURE!
    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()

if __name__ ==  '__main__':
    main()