#!/usr/bin/env python
"""CEVAE model on IHDP
"""
from __future__ import absolute_import
from __future__ import division

import edward as ed
import tensorflow as tf

import pickle

from edward.models import Bernoulli, Normal
from progressbar import ETA, Bar, Percentage, ProgressBar

from datasets import IHDP, SYNData, SYNDataFrontBack
from evaluation import Evaluator
import numpy as np
import time
from scipy.stats import sem

import matplotlib.pyplot as plt

from utils import fc_net, get_y0_y1
from argparse import ArgumentParser






def iteration(reps=100,noise1=1,noise2=1):
    parser = ArgumentParser()
    parser.add_argument('-reps', type=int, default=10)
    parser.add_argument('-earl', type=int, default=5)
    parser.add_argument('-lr', type=float, default=0.0005)
    parser.add_argument('-opt', choices=['adam', 'adamax'], default='adam')
    parser.add_argument('-epochs', type=int, default=50)
    parser.add_argument('-print_every', type=int, default=10)

    args = parser.parse_args()
    args.reps = reps
    args.true_post = True

    dataset = SYNDataFrontBack(replications=args.reps,noiseX1=noise1,noiseX2=noise2)
    # dimx = 25
    dimx = 1
    scores = np.zeros((args.reps, 3))
    scores_test = np.zeros((args.reps, 3))

    M = None  # batch size during training
    d = 20  # latent dimension
    d = 100

    lamba = 1e-4  # weight decay
    nh, h = 3, 200  # number and size of hidden layers
    # nh, h = 2, 1000  # number and size of hidden layers

    for i, (train, valid, test, contfeats, binfeats) in enumerate(dataset.get_train_valid_test()):
        print '\nReplication {}/{}'.format(i + 1, args.reps)
        (xtr, ttr, ytr), (y_cftr, mu0tr, mu1tr) = train
        (xva, tva, yva), (y_cfva, mu0va, mu1va) = valid
        (xte, tte, yte), (y_cfte, mu0te, mu1te) = test
        evaluator_test = Evaluator(yte, tte, y_cf=y_cfte, mu0=mu0te, mu1=mu1te)

        # reorder features with binary first and continuous after
        perm = binfeats + contfeats
        xtr, xva, xte = xtr[:,:, perm], xva[:,:, perm], xte[:,:, perm]

        xalltr, talltr, yalltr = np.concatenate([xtr, xva], axis=0), np.concatenate([ttr, tva], axis=0), np.concatenate([ytr, yva], axis=0)

        evaluator_train = Evaluator(yalltr, talltr, y_cf=np.concatenate([y_cftr, y_cfva], axis=0),
                                    mu0=np.concatenate([mu0tr, mu0va], axis=0), mu1=np.concatenate([mu1tr, mu1va], axis=0))

        # zero mean, unit variance for y during training
        ym, ys = np.mean(ytr), np.std(ytr)
        ytr, yva = (ytr - ym) / ys, (yva - ym) / ys

        best_logpvalid = - np.inf

        with tf.Graph().as_default():
            sess = tf.InteractiveSession()

            ed.set_seed(1)
            np.random.seed(1)
            tf.set_random_seed(1)

            x_ph_bin = tf.placeholder(tf.float32, [M,2, len(binfeats)], name='x_bin')  # binary inputs
            x_ph_cont = tf.placeholder(tf.float32, [M,2,len(contfeats)], name='x_cont')  # continuous inputs
            t_ph = tf.placeholder(tf.float32, [M, 1])
            y_ph = tf.placeholder(tf.float32, [M, 1])

            x_ph = tf.concat([x_ph_bin, x_ph_cont], 2)
            activation = tf.nn.elu

            # CEVAE model (decoder)
            # p(z)
            z = Normal(loc=tf.zeros([tf.shape(x_ph)[0], 2*d]), scale=tf.ones([tf.shape(x_ph)[0], 2*d]))


            # p(x|z)
            hx = fc_net(z, (nh - 1) * [h], [], 'px_z_shared', lamba=lamba, activation=activation)
            logits = fc_net(hx, [h], [[2*len(binfeats), None]], 'px_z_bin'.format(i + 1), lamba=lamba, activation=activation)
            # x1 = Bernoulli(logits=tf.expand_dims(logits,-1), dtype=tf.float32, name='bernoulli_px_z')
            x1 = Bernoulli(logits=tf.concat((tf.expand_dims(logits,1),tf.expand_dims(logits,1)),axis=1), dtype=tf.float32, name='bernoulli_px_z')

            mu_before, sigma_before = fc_net(hx, [h], [[len(contfeats), None], [len(contfeats), tf.nn.softplus]], 'px_z_before_cont', lamba=lamba,
                               activation=activation)
            mu_after, sigma_after = fc_net(hx, [h], [[len(contfeats), None], [len(contfeats), tf.nn.softplus]], 'px_z_after_cont', lamba=lamba,
                               activation=activation)

            mu = tf.concat((mu_before,mu_after),axis=1)
            mu = tf.expand_dims(mu,dim=2)
            sigma = tf.concat((sigma_before,sigma_after),axis=1)
            sigma = tf.expand_dims(sigma,dim=2)

            x2 = Normal(loc=mu, scale=sigma, name='gaussian_px_z')

            # p(t|z)
            logits = fc_net(z, [h], [[1, None]], 'pt_z', lamba=lamba, activation=activation)
            t = Bernoulli(logits=logits, dtype=tf.float32)

            # p(y|t,z)
            mu2_t0 = fc_net(z, nh * [h], [[1, None]], 'py_t0z', lamba=lamba, activation=activation)
            mu2_t1 = fc_net(z, nh * [h], [[1, None]], 'py_t1z', lamba=lamba, activation=activation)
            y = Normal(loc=t * mu2_t1 + (1. - t) * mu2_t0, scale=tf.ones_like(mu2_t0))

            # CEVAE variational approximation (encoder) Before
            # q(t|x)
            x_ph_before = x_ph[:,0]
            x_ph_after = x_ph[:,1]
            logits_t_before = fc_net(x_ph_before, [d], [[1, None]], 'qt_before', lamba=lamba, activation=activation)
            qt_before = Bernoulli(logits=logits_t_before, dtype=tf.float32)
            # q(y|x,t)
            hqy_before = fc_net(x_ph_before, (nh - 1) * [h], [], 'qy_xt_shared_before', lamba=lamba, activation=activation)
            mu_qy_t0_before = fc_net(hqy_before, [h], [[1, None]], 'qy_xt0_before', lamba=lamba, activation=activation)
            mu_qy_t1_before = fc_net(hqy_before, [h], [[1, None]], 'qy_xt1_before', lamba=lamba, activation=activation)
            qy_before = Normal(loc=qt_before * mu_qy_t1_before + (1. - qt_before) * mu_qy_t0_before, scale=tf.ones_like(mu_qy_t0_before))
            # q(z|x,t,y)
            inpt2_before = tf.concat([x_ph_before, qy_before], 1)
            hqz_before = fc_net(inpt2_before, (nh - 1) * [h], [], 'qz_xty_shared_before', lamba=lamba, activation=activation)
            muq_t0_before, sigmaq_t0_before = fc_net(hqz_before, [h], [[d, None], [d, tf.nn.softplus]], 'qz_xt0_before', lamba=lamba,
                                       activation=activation)
            muq_t1_before, sigmaq_t1_before = fc_net(hqz_before, [h], [[d, None], [d, tf.nn.softplus]], 'qz_xt1_before', lamba=lamba,
                                       activation=activation)

            # CEVAE variational approximation (encoder) After
            # q(t|x)
            logits_t_after = fc_net(x_ph_after, [d], [[1, None]], 'qt_after', lamba=lamba, activation=activation)
            qt_after = Bernoulli(logits=logits_t_after, dtype=tf.float32)
            # q(y|x,t)
            hqy_after = fc_net(x_ph_after, (nh - 1) * [h], [], 'qy_xt_shared_after', lamba=lamba, activation=activation)
            mu_qy_t0_after = fc_net(hqy_after, [h], [[1, None]], 'qy_xt0_after', lamba=lamba, activation=activation)
            mu_qy_t1_after = fc_net(hqy_after, [h], [[1, None]], 'qy_xt1_after', lamba=lamba, activation=activation)
            qy_after = Normal(loc=qt_after * mu_qy_t1_after + (1. - qt_after) * mu_qy_t0_after, scale=tf.ones_like(mu_qy_t0_after))
            # q(z|x,t,y)
            inpt2_after = tf.concat([x_ph_after, qy_after], 1)
            hqz_after = fc_net(inpt2_after, (nh - 1) * [h], [], 'qz_xty_shared_after', lamba=lamba, activation=activation)
            muq_t0_after, sigmaq_t0_after = fc_net(hqz_after, [h], [[d, None], [d, tf.nn.softplus]], 'qz_xt0_after', lamba=lamba,
                                       activation=activation)
            muq_t1_after, sigmaq_t1_after = fc_net(hqz_after, [h], [[d, None], [d, tf.nn.softplus]], 'qz_xt1_after', lamba=lamba,
                                       activation=activation)

            qloc = tf.concat((qt_before * muq_t1_before + (1. - qt_before) * muq_t0_before,
                              qt_after * muq_t1_after + (1. - qt_after) * muq_t0_after), axis=1)
            qscale = tf.concat((qt_before * sigmaq_t1_before + (1. - qt_before) * sigmaq_t0_before,
                               qt_after * sigmaq_t1_after + (1. - qt_after) * sigmaq_t0_after), axis=1)
            qz = Normal(loc=qloc, scale=qscale)
            # Create data dictionary for edward
            data = {x1: x_ph_bin, x2: x_ph_cont, y: y_ph, qt_before: t_ph, qt_after: t_ph, t: t_ph, qy_before: y_ph, qy_after: y_ph}

            # sample posterior predictive for p(y|z,t)
            y_post = ed.copy(y, {z: qz, t: t_ph}, scope='y_post')
            # crude approximation of the above
            y_post_mean = ed.copy(y, {z: qz.mean(), t: t_ph}, scope='y_post_mean')
            # construct a deterministic version (i.e. use the mean of the approximate posterior) of the lower bound
            # for early stopping according to a validation set
            y_post_eval = ed.copy(y, {z: qz.mean(), qt_before: t_ph, qy_before: y_ph, qt_after: t_ph, qy_after: y_ph, t: t_ph}, scope='y_post_eval')
            x1_post_eval = ed.copy(x1, {z: qz.mean(),qt_before: t_ph, qy_before: y_ph, qt_after: t_ph, qy_after: y_ph}, scope='x1_post_eval')
            x2_post_eval = ed.copy(x2, {z: qz.mean(), qt_before: t_ph, qy_before: y_ph, qt_after: t_ph, qy_after: y_ph,}, scope='x2_post_eval')
            t_post_eval = ed.copy(t, {z: qz.mean(), qt_before: t_ph, qy_before: y_ph, qt_after: t_ph, qy_after: y_ph,}, scope='t_post_eval')
            logp_valid = tf.reduce_mean(tf.reduce_sum(y_post_eval.log_prob(y_ph) + t_post_eval.log_prob(t_ph), axis=1) +
                                        tf.reduce_sum(x1_post_eval.log_prob(x_ph_bin), axis=[1,2]) +
                                        tf.reduce_sum(x2_post_eval.log_prob(x_ph_cont),axis=[1,2]) +
                                        tf.reduce_sum(z.log_prob(qz.mean()) - qz.log_prob(qz.mean()), axis=1))

            inference = ed.KLqp({z: qz}, data)
            optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
            inference.initialize(optimizer=optimizer)


            saver = tf.train.Saver(tf.contrib.slim.get_variables())
            tf.global_variables_initializer().run()

            n_epoch, n_iter_per_epoch, idx = args.epochs, 10 * int(xtr.shape[0] / 100), np.arange(xtr.shape[0])

            # dictionaries needed for evaluation
            tr0, tr1 = np.zeros((xalltr.shape[0], 1)), np.ones((xalltr.shape[0], 1))
            tr0t, tr1t = np.zeros((xte.shape[0], 1)), np.ones((xte.shape[0], 1))
            f1 = {x_ph_bin: xalltr[:,:, 0:len(binfeats)], x_ph_cont: xalltr[:,:, len(binfeats):], t_ph: tr1}
            f0 = {x_ph_bin: xalltr[:,:, 0:len(binfeats)], x_ph_cont: xalltr[:,:, len(binfeats):], t_ph: tr0}
            f1t = {x_ph_bin: xte[:,:, 0:len(binfeats)], x_ph_cont: xte[:,:, len(binfeats):], t_ph: tr1t}
            f0t = {x_ph_bin: xte[:,:, 0:len(binfeats)], x_ph_cont: xte[:,:, len(binfeats):], t_ph: tr0t}

            for epoch in range(n_epoch):
                avg_loss = 0.0

                t0 = time.time()
                widgets = ["epoch #%d|" % epoch, Percentage(), Bar(), ETA()]
                pbar = ProgressBar(max_value=n_iter_per_epoch, widgets=widgets)
                pbar.start()
                np.random.shuffle(idx)
                for j in range(n_iter_per_epoch):
                    pbar.update(j)
                    batch = np.random.choice(idx, 100)
                    x_train, y_train, t_train = xtr[batch][:,:], ytr[batch], ttr[batch]
                    info_dict = inference.update(feed_dict={x_ph_bin: x_train[:,:, 0:len(binfeats)],
                                                            x_ph_cont: x_train[:,:,len(binfeats):],
                                                            t_ph: t_train, y_ph: y_train})
                    avg_loss += info_dict['loss']

                avg_loss = avg_loss / n_iter_per_epoch
                avg_loss = avg_loss / 100

                if epoch % args.earl == 0 or epoch == (n_epoch - 1):
                    logpvalid = sess.run(logp_valid, feed_dict={x_ph_bin: xva[:,:, 0:len(binfeats)], x_ph_cont: xva[:,:,len(binfeats):],
                                                                t_ph: tva, y_ph: yva})
                    if logpvalid >= best_logpvalid:
                        print 'Improved validation bound, old: {:0.3f}, new: {:0.3f}'.format(best_logpvalid, logpvalid)
                        best_logpvalid = logpvalid
                        saver.save(sess, 'models/m6-ihdp')

                if epoch % args.print_every == 0:
                    y0, y1 = get_y0_y1(sess, y_post, f0, f1, shape=yalltr.shape, L=1)
                    y0, y1 = y0 * ys + ym, y1 * ys + ym
                    score_train = evaluator_train.calc_stats(y1, y0)
                    rmses_train = evaluator_train.y_errors(y0, y1)

                    y0, y1 = get_y0_y1(sess, y_post, f0t, f1t, shape=yte.shape, L=1)
                    y0, y1 = y0 * ys + ym, y1 * ys + ym
                    score_test = evaluator_test.calc_stats(y1, y0)

                    print "Epoch: {}/{}, log p(x) >= {:0.3f}, ite_tr: {:0.3f}, ate_tr: {:0.3f}, pehe_tr: {:0.3f}, " \
                          "rmse_f_tr: {:0.3f}, rmse_cf_tr: {:0.3f}, ite_te: {:0.3f}, ate_te: {:0.3f}, pehe_te: {:0.3f}, " \
                          "dt: {:0.3f}".format(epoch + 1, n_epoch, avg_loss, score_train[0], score_train[1], score_train[2],
                                               rmses_train[0], rmses_train[1], score_test[0], score_test[1], score_test[2],
                                               time.time() - t0)

            saver.restore(sess, 'models/m6-ihdp')
            y0, y1 = get_y0_y1(sess, y_post, f0, f1, shape=yalltr.shape, L=100)
            y0, y1 = y0 * ys + ym, y1 * ys + ym
            score = evaluator_train.calc_stats(y1, y0)
            scores[i, :] = score

            y0t, y1t = get_y0_y1(sess, y_post, f0t, f1t, shape=yte.shape, L=100)
            y0t, y1t = y0t * ys + ym, y1t * ys + ym
            score_test = evaluator_test.calc_stats(y1t, y0t)
            scores_test[i, :] = score_test

            print 'Replication: {}/{}, tr_ite: {:0.3f}, tr_ate: {:0.3f}, tr_pehe: {:0.3f}' \
                  ', te_ite: {:0.3f}, te_ate: {:0.3f}, te_pehe: {:0.3f}'.format(i + 1, args.reps,
                                                                                score[0], score[1], score[2],
                                                                                score_test[0], score_test[1], score_test[2])
            sess.close()

    print 'CEVAE model total scores'
    means, stds = np.mean(scores, axis=0), sem(scores, axis=0)
    means_train, stds_train = means, stds
    print 'train ITE: {:.3f}+-{:.3f}, train ATE: {:.3f}+-{:.3f}, train PEHE: {:.3f}+-{:.3f}' \
          ''.format(means[0], stds[0], means[1], stds[1], means[2], stds[2])


    means, stds = np.mean(scores_test, axis=0), sem(scores_test, axis=0)
    means_test, stds_test = means, stds
    print 'test ITE: {:.3f}+-{:.3f}, test ATE: {:.3f}+-{:.3f}, test PEHE: {:.3f}+-{:.3f}' \
          ''.format(means[0], stds[0], means[1], stds[1], means[2], stds[2])

    return means_train[1], stds_train[1], means_test[1], stds_test[1]



noises = [1,3,5,10,25,50,100]

means_train = np.zeros(len(noises))
stds_train = np.zeros(means_train.shape)
means_test = np.zeros(means_train.shape)
stds_test = np.zeros(means_train.shape)
for i,noise in enumerate(noises):
    print('\n\n\n\n\n\n\n---------------------------------------------------------------\nCurrent noise level is: noise = {}\n---------------------------------------------------------------\n\n\n\n\n\n\n'.format(noise))
    means_train[i], stds_train[i], means_test[i], stds_test[i] = iteration(reps=50,noise1=noise)


data = {}
data['noises'] = noises
data['mean_train'] = means_train
data['std_train'] = stds_train
data['mean_test'] = means_test
data['std_test'] = stds_test


with open("simulation_data_dualCEVAE_pre_noise_lr.pickle","wb") as f:
    pickle.dump(data,f)

noises = [1, 3, 5, 10, 25, 50, 100]

means_train = np.zeros(len(noises))
stds_train = np.zeros(means_train.shape)
means_test = np.zeros(means_train.shape)
stds_test = np.zeros(means_train.shape)
for i, noise in enumerate(noises):
    print(
        '\n\n\n\n\n\n\n---------------------------------------------------------------\nCurrent noise level is: noise = {}\n---------------------------------------------------------------\n\n\n\n\n\n\n'.format(
            noise))
    means_train[i], stds_train[i], means_test[i], stds_test[i] = iteration(reps=50, noise2=noise)

data = {}
data['noises'] = noises
data['mean_train'] = means_train
data['std_train'] = stds_train
data['mean_test'] = means_test
data['std_test'] = stds_test

with open("simulation_data_dualCEVAE_post_noise_lr.pickle", "wb") as f:
    pickle.dump(data, f)

#
noises1 = [1,3,5,10,25,50,100]
noises2 = noises1
noisesX, noisesY = np.meshgrid(noises1, noises2)
means_train = np.zeros(noisesX.shape)
stds_train = np.zeros(noisesX.shape)
means_test = np.zeros(noisesX.shape)
stds_test = np.zeros(noisesX.shape)
for i,noise1 in enumerate(noises1):
    for j,noise2 in enumerate(noises2):
        print('\n\n\n\n\n\n\n---------------------------------------------------------------\nCurrent noise levels are: noise1 = {} , noises2 = {}\n---------------------------------------------------------------\n\n\n\n\n\n\n'.format(noise1,noise2))
        means_train[i,j], stds_train[i,j], means_test[i,j], stds_test[i,j]  = iteration(reps=50,noise1=noise1,noise2=noise2)
    data = {}
    data['noises_1_before'] = noises1
    data['noises_2_after'] = noises2
    data['noisesX'] = noisesX
    data['noisesY'] = noisesY
    data['mean_train'] = means_train
    data['std_train'] = stds_train
    data['mean_test'] = means_test
    data['std_test'] = stds_test

    with open("simulation_data_{}.pickle".format(i), "wb") as f:
        pickle.dump(data, f)
#

data = {}
data['noises_1_before'] = noises1
data['noises_2_after'] = noises2
data['noisesX'] = noisesX
data['noisesY'] = noisesY
data['mean_train'] = means_train
data['std_train'] = stds_train
data['mean_test'] = means_test
data['std_test'] = stds_test


with open("simulation_data.pickle","wb") as f:
    pickle.dump(data,f)


plt.errorbar(noises1,means_test[:,0],yerr=stds_test[:,0], linestyle='None', fmt='o')
plt.xlabel('Noise Coefficient')
plt.ylabel('ATE estimation Error')
plt.title('ATE error vs. Pre-treatment Noise')
plt.savefig('Noise1.png')

plt.errorbar(noises1,means_test[0,:],yerr=stds_test[0,:], linestyle='None', fmt='o')
plt.xlabel('Noise Coefficient')
plt.ylabel('ATE estimation Error')
plt.title('ATE error vs. Post-treatment Noise')
plt.savefig('Noise2.png')

plt.errorbar(noises1,np.diag(means_test),yerr=np.diag(stds_test), linestyle='None', fmt='o')
plt.xlabel('Noise Coefficient')
plt.ylabel('ATE estimation Error')
plt.title('ATE error vs. Post-treatment Noise')
plt.savefig('NoiseDiagonal.png')

plt.contourf(noisesX,noisesY,means_test)
plt.title('Post-Treatment vs. Pre-Treatment noise')
plt.xlabel('Pre-Treatment Noise Level')
plt.ylabel('Post-Treatment Noise Level')
plt.savefig('Contour.png')



plt.errorbar(noises,means,yerr=stds, linestyle='None', fmt='o')
plt.xlabel('Noise Coefficient')
plt.ylabel('ATE estimation Error')
plt.title('ATE error vs. Post-treatment Noise')
plt.savefig('Noise2.png')