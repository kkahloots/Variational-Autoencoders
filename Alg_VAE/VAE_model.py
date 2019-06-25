"""VAE_model.py: Tensorflow model for the VAutoencoder"""
__author__      = "Khalid M. Kahloot"
__copyright__   = "Copyright 2019, Only for professionals"


import sys
sys.path.append('..')
import gc
import glob

from PIL import Image as PILImage
from IPython.display import display, Image
import os

from base.base_model import BaseModel
import tensorflow as tf
import numpy as np
import dask.array as da

from .VAE_graph import VAEGraph
from .VAECNN_graph import VAECNNGraph

from utils.logger import Logger
from utils.early_stopping import EarlyStopping
from tqdm import tqdm_notebook as tqdm
import sys
from collections import defaultdict

import utils.utils as utils
import utils.constants as const

from dask_ml.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
import hdbscan
from dask_ml.metrics import accuracy_score
from utils.plots import plot_dataset, plot_dataset3d, plot_samples, merge, resize_gif

class VAEModel(BaseModel):
    def __init__(self, network_params, act_out=tf.nn.softplus, sigma=0.001,
                 transfer_fct= tf.nn.relu, learning_rate=1e-5,
                 kinit=tf.contrib.layers.xavier_initializer(), batch_size=32,
                 dropout=0.2, batch_norm=True, epochs=200, checkpoint_dir='',
                 summary_dir='', result_dir='', restore=False, plot=False, clustering=False, colab=False, model_type=const.VAE):

        BaseModel.__init__(self, checkpoint_dir, summary_dir, result_dir)
        self.summary_dir = summary_dir
        self.result_dir = result_dir
        self.batch_size = batch_size
        self.dropout = dropout
        self.sigma = sigma

        self.epochs = epochs
        self.w_file = result_dir + '/w_file'
        self.restore = restore
        self.plot = plot
        self.colab = colab
        self.clustering = clustering

        if self.plot:
            self.w_space_files = list()
            self.w_space3d_files = list()
            self.recons_files = list()

        # Creating computational graph for train and test
        self.graph = tf.Graph()
        with self.graph.as_default():
            if(model_type == const.VAE):
                self.model_graph = VAEGraph(network_params=network_params, act_out=act_out,sigma=sigma,
                                            transfer_fct=transfer_fct, learning_rate=learning_rate, kinit=kinit,
                                            batch_size=batch_size, reuse=False)
            if(model_type == const.VAECNN):
                self.model_graph = VAECNNGraph(network_params=network_params, act_out=act_out,sigma=sigma,
                                               transfer_fct=transfer_fct, learning_rate=learning_rate, kinit=kinit,
                                                batch_size=batch_size, reuse=False)

            self.model_graph.build_graph()
            self.trainable_count = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])


    def train_epoch(self, session,logger, data_train):
        loop = tqdm(range(data_train.num_batches(self.batch_size)))
        vae_losses = []
        ae_losses = []
        kl_losses = []
        recons = []
        L2_loss = []

        for _ in loop:
            batch_x = next(data_train.next_batch(self.batch_size))
            vae_loss_curr, ae_loss_curr, recons_curr, kl_loss_curr, L2_loss_curr = self.model_graph.partial_fit(session, batch_x)
            vae_losses.append(vae_loss_curr)
            ae_losses.append(ae_loss_curr)
            recons.append(recons_curr)
            kl_losses.append(kl_loss_curr)
            L2_loss.append(L2_loss_curr)

        vae_tr = np.mean(vae_losses)
        ae_tr = np.mean(ae_losses)
        recons_tr = np.mean(recons)
        kl_tr = np.mean(kl_losses)
        L2_loss = np.mean(L2_loss)

        cur_it = self.model_graph.global_step_tensor.eval(session)
        summaries_dict = {
            'vae_loss': vae_tr,
            'ae_loss': ae_tr,
            'recons_loss': recons_tr,
            'kl_loss': kl_tr,
            'L2_loss': L2_loss
        }

        logger.summarize(cur_it, summaries_dict=summaries_dict)

        return vae_tr, ae_tr, recons_tr, kl_tr, L2_loss

    def valid_epoch(self, session, logger, data_valid):
        # COMPUTE VALID LOSS
        loop = tqdm(range(data_valid.num_batches(self.batch_size)))
        vae_losses = []
        ae_losses = []
        kl_losses = []
        recons = []
        L2_loss = []

        for _ in loop:
            batch_x = next(data_valid.next_batch(self.batch_size))
            vae_loss_curr, ae_loss_curr, recons_curr, kl_loss_curr, L2_loss_curr = self.model_graph.evaluate(session, batch_x)

            vae_losses.append(vae_loss_curr)
            ae_losses.append(ae_loss_curr)
            recons.append(recons_curr)
            kl_losses.append(kl_loss_curr)
            L2_loss.append(L2_loss_curr)

        vae_val = np.mean(vae_losses)
        ae_val = np.mean(ae_losses)
        recons_val = np.mean(recons)
        kl_val = np.mean(kl_losses)
        L2_loss = np.mean(L2_loss)

        cur_it = self.model_graph.global_step_tensor.eval(session)
        summaries_dict = {
            'vae_loss': vae_val,
            'ae_loss': ae_val,
            'recons_loss': recons_val,
            'kl_loss': kl_val,
            'L2_loss': L2_loss
        }

        logger.summarize(cur_it, summaries_dict=summaries_dict)

        return vae_val, ae_val, recons_val, kl_val, L2_loss

    def decay_fn(self):
        return self.model_graph.decay_lr(self.session)

    def train(self, data_train, data_valid, enable_es=1):

        with tf.Session(graph=self.graph) as session:
            tf.set_random_seed(1234)

            logger = Logger(session, self.summary_dir)
            # here you initialize the tensorflow saver that will be used in saving the checkpoints.
            # max_to_keep: defaults to keeping the 5 most recent checkpoints of your model
            saver = tf.train.Saver()
            self.session = session
            early_stopping = EarlyStopping(name='total loss', decay_fn=self.decay_fn)

            if(self.restore and self.load(session, saver) ):
                num_epochs_trained = self.model_graph.cur_epoch_tensor.eval(session)
                print('EPOCHS trained: ', num_epochs_trained)
            else:
                print('Initizalizing Variables ...')
                tf.global_variables_initializer().run()


            if(self.model_graph.cur_epoch_tensor.eval(session) ==  self.epochs):
                return

            for cur_epoch in range(self.model_graph.cur_epoch_tensor.eval(session), self.epochs + 1, 1):

                print('EPOCH: ', cur_epoch)
                self.current_epoch = cur_epoch

                vae_tr, ae_tr, recons_tr, kl_tr, L2_loss_tr = self.train_epoch(session, logger, data_train)
                if da.isnan(vae_tr):
                    print ('Encountered NaN, stopping training. Please check the learning_rate settings and the momentum.')
                    print('AE loss: ', ae_tr)
                    print('Recons loss: ', recons_tr)
                    print('kl tr: ', kl_tr)
                    print('L2 loss: ', L2_loss_tr)

                    sys.exit()

                vae_val, ae_val, recons_val, kl_val, L2_loss_val = self.valid_epoch(session, logger, data_valid)

                print('TRAIN | VAE Loss: ', vae_tr, 'AE Loss: ', ae_tr, ' | Recons: ', recons_tr)
                print('      | KL-div: ', kl_tr, ' | L2_loss: ', L2_loss_tr)

                print('VALID | VAE Loss: ', vae_val, 'AE Loss: ', ae_val, ' | Recons: ', recons_val)
                print('      | KL-div: ', kl_val, ' | L2_loss: ', L2_loss_val)


                if (cur_epoch==1) or ((cur_epoch % const.SAVE_EPOCH == 0) and ((cur_epoch!=0))):
                    self.save(session, saver, self.model_graph.global_step_tensor.eval(session))
                    if self.plot:
                        self.generate_samples(data_train, session, cur_epoch)

                    if self.clustering:
                        self.generate_clusters(logger, cur_epoch, data_train, data_valid)

                session.run(self.model_graph.increment_cur_epoch_tensor)

                #Early stopping
                if(enable_es==1 and early_stopping.stop(vae_val)):
                    print('Early Stopping!')
                    break

                if cur_epoch % 50 == 0:
                    if self.colab:
                        self.push_colab()

            self.save(session, saver, self.model_graph.global_step_tensor.eval(session))

            if self.colab:
                self.push_colab()
        return


    def test (self, data):
        with tf.Session(graph=self.graph) as session:
            tf.set_random_seed(1234)

            logger = Logger(session, self.summary_dir)
            # here you initialize the tensorflow saver that will be used in saving the checkpoints.
            # max_to_keep: defaults to keeping the 5 most recent checkpoints of your model
            saver = tf.train.Saver()
            self.session = session
            try:
                num_epochs_trained=self.model_graph.cur_epoch_tensor.eval(session)
            except:
                if (self.load(session, saver)):
                    num_epochs_trained = self.model_graph.cur_epoch_tensor.eval(session)
                    print('EPOCHS trained: ', num_epochs_trained)

                vae_val, ae_val, recons_val, kl_val, L2_loss_val = self.valid_epoch(session, logger, data)

                print('VALID | VAE Loss: ', vae_val, 'AE Loss: ', ae_val, ' | Recons: ', recons_val)
                print('      | KL-div: ', kl_val, ' | L2_loss: ', L2_loss_val)

                session.run(self.model_graph.increment_cur_epoch_tensor)

            if self.plot:
                self.generate_samples(data, session, 0)

        return

    def evaluate(self, data_valid):

        with tf.Session(graph=self.graph) as session:
            tf.set_random_seed(1234)

            logger = Logger(session, self.summary_dir)
            # here you initialize the tensorflow saver that will be used in saving the checkpoints.
            # max_to_keep: defaults to keeping the 5 most recent checkpoints of your model
            saver = tf.train.Saver()
            self.session = session
            try:
                num_epochs_trained=self.model_graph.cur_epoch_tensor.eval(session)
            except:
                if (self.load(session, saver)):
                    num_epochs_trained = self.model_graph.cur_epoch_tensor.eval(session)
                    print('EPOCHS trained: ', num_epochs_trained)

            if(num_epochs_trained ==  self.epochs):
                return

            for cur_epoch in range(num_epochs_trained, self.epochs + 1, 1):

                print('EPOCH: ', cur_epoch)
                self.current_epoch = cur_epoch

                vae_val, ae_val, recons_val, kl_val, L2_loss_val = self.valid_epoch(session, logger, data_valid)

                print('VALID | VAE Loss: ', vae_val, 'AE Loss: ', ae_val, ' | Recons: ', recons_val)
                print('      | KL-div: ', kl_val, ' | L2_loss: ', L2_loss_val)

                session.run(self.model_graph.increment_cur_epoch_tensor)

            if self.plot:
                self.generate_samples(data_valid, session, cur_epoch)

        return


    '''  
    ------------------------------------------------------------------------------
                                         MODEL OPERATIONS
    ------------------------------------------------------------------------------ 
    '''
    def reconst(self, inputs):
        return self.decode(self.encode(inputs))

    def encode(self, inputs):
        return self.batch_function(self.model_graph.encode, inputs)

    def decode(self, w):
        return self.batch_function(self.model_graph.decode, w)

    def interpolate(self, input1, input2):

        z1 = self.encode(input1)
        z2 = self.encode(input2)

        decodes = defaultdict(list)
        for idx, ratio in enumerate(np.linspace(0, 1, 10)):
            decode = dict()
            z = np.stack([self.slerp(ratio, r1, r2) for r1, r2 in zip(z1, z2)])
            z_decode = self.decode(z)

            for i in range(z_decode.shape[0]):
                try:
                    decode[i] = [z_decode[i].compute()]
                except:
                    decode[i] = [z_decode[i]]

            for i in range(z_decode.shape[0]):
                decodes[i] = decodes[i] + decode[i]

        imgs = []

        for idx in decodes:
            l = []

            l += [input1[idx:idx + 1][0]]
            l += decodes[idx]
            l += [input2[idx:idx + 1][0]]

            imgs.append(l)
        del decodes

        return imgs


    def reconst_loss(self, inputs):
        return self.batch_function(self.model_graph.reconst_loss, inputs)

    def slerp(self, val, low, high):
        """Code from https://github.com/soumith/dcgan.torch/issues/14"""
        omega = da.arccos(da.clip(da.dot(low / da.linalg.norm(low), high.transpose() / da.linalg.norm(high)), -1, 1))
        so = da.sin(omega)

        #l1 = lambda low, high, val: (1.0-val) * low + val * high
        #l2 = lambda low, high, val, so, omega: np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high
        if so == 0:
            return (1.0-val) * low + val * high # L'Hopital's rule/LERP
        return da.sin((1.0 - val) * omega) / so * low + da.sin(val * omega) / so * high

    '''
    ------------------------------------------------------------------------------
                                         GENERATE SAMPLES
    ------------------------------------------------------------------------------
    '''

    def animate(self):
        if not hasattr(self, 'w_space_files') or len(self.recons_files)==0:
            print('No images were generated during trainning!')

            path = self.summary_dir
            st = path+'/{} samples generation in epoch'.format(self.summary_dir.split('/')[-1:][0])
            self.recons_files = [f for f in glob.glob(path + "**/*.jpg", recursive=True) if f.startswith(st)]
            self.recons_files = list(map(lambda f: f.split('/')[-1], self.recons_files))

            self.recons_files.sort(key=utils.natural_keys)
            self.recons_files = list(map(lambda f: path+'/'+f , self.recons_files))

            st = path+'/{} W space in epoch'.format(self.summary_dir.split('/')[-1:][0])
            self.w_space_files = [f for f in glob.glob(path + "**/*.jpg", recursive=True) if f.startswith(st)]
            self.w_space_files = list(map(lambda f: f.split('/')[-1], self.w_space_files))

            self.w_space_files.sort(key=utils.natural_keys)
            self.w_space_files = list(map(lambda f: path + '/' + f, self.w_space_files))

            st = path + '/{} W space 3d in epoch'.format(self.summary_dir.split('/')[-1:][0])
            self.w_space3d_files = [f for f in glob.glob(path + "**/*.jpg", recursive=True) if f.startswith(st)]
            self.w_space3d_files = list(map(lambda f: f.split('/')[-1], self.w_space3d_files))

            self.w_space3d_files.sort(key=utils.natural_keys)
            self.w_space3d_files = list(map(lambda f: path + '/' + f, self.w_space3d_files))

            if len(self.recons_files)==0:
                print('No previous images found!')
                return None

        path = self.summary_dir
        st = path+'/{} samples generation in epoch'.format(self.summary_dir.split('/')[-1:][0])
        images = [PILImage.open(fn) for fn in self.recons_files]
        images[0].save(st+'_res_animate.gif', save_all=True, append_images=images[1:], duration=len(images)*60, loop=0xffff)
        with open(st+'_res_animate.gif','rb') as f:
            img1 = Image(data=f.read(), format='gif')

        st = path+'/{} W space in epoch'.format(self.summary_dir.split('/')[-1:][0])
        images = [PILImage.open(fn) for fn in self.w_space_files]
        images[0].save(st+'_animate.gif', save_all=True, append_images=images[1:], duration=len(images)*60, loop=0xffff, dpi=70)
        resize_gif(path=st+'_animate.gif', save_as=st+'_res_animate.gif', resize_to=(900,450))
        with open(st+'_res_animate.gif','rb') as f:
            img2 = Image(data=f.read(), format='gif')

        st = path+'/{} W space 3d in epoch'.format(self.summary_dir.split('/')[-1:][0])
        images = [PILImage.open(fn) for fn in self.w_space3d_files]
        images[0].save(st+'_animate.gif', save_all=True, append_images=images[1:], duration=len(images)*60, loop=0xffff, dpi=70)
        resize_gif(path=st+'_animate.gif', save_as=st+'_res_animate.gif', resize_to=(900,450))
        with open(st+'_res_animate.gif','rb') as f:
            img3 = Image(data=f.read(), format='gif')

        return img1, img2, img3

    def do_clustering(self, x, alg='kmeans'):
        w = self.encode(x)
        if alg=='kmeans':
            clustering = MiniBatchKMeans(verbose=True)
            y_pred = clustering.fit_predict(w)
        else:
            clustering = hdbscan.HDBSCAN(min_cluster_size=50, gen_min_span_tree=False)
            clustering = clustering.fit(w)
            y_pred = clustering.labels_
        del clustering, w
        return y_pred

    def generate_clusters(self, logger, cur_it, data_tr, data_val):
        # Generating W space
        print('Generating W space ...')

        cluster_tr = self.do_clustering(data_tr.x, alg='kmeans')
        cluster_val = self.do_clustering(data_val.x, alg='kmeans')

        accuracy_tr = accuracy_score(data_tr.labels, cluster_tr)
        accuracy_val = accuracy_score(data_val.labels, cluster_val)

        summaries_dict = {
            'kmeans_cluster_acc': accuracy_tr
        }
        logger.summarize(cur_it, summarizer='train', summaries_dict=summaries_dict)

        summaries_dict = {
            'kmeans_cluster_acc': accuracy_val
        }
        logger.summarize(cur_it, summarizer='test', summaries_dict=summaries_dict)

        print('TRAIN | kmeans Clustering Acc: ', accuracy_tr)
        print('VALID | kmeans Clustering Acc: ', accuracy_val)

        cluster_tr = self.do_clustering(data_tr.x, alg='hdbscan')
        cluster_val = self.do_clustering(data_val.x, alg='hdbscan')

        accuracy_tr = accuracy_score(data_tr.labels, cluster_tr)
        accuracy_val = accuracy_score(data_val.labels, cluster_val)

        summaries_dict = {
            'hdbscan_cluster_acc': accuracy_tr
        }
        logger.summarize(cur_it, summarizer='train', summaries_dict=summaries_dict)

        summaries_dict = {
            'hdbscan_cluster_acc': accuracy_val
        }
        logger.summarize(cur_it, summarizer='test', summaries_dict=summaries_dict)

        print('TRAIN | hdbscan Clustering Acc: ', accuracy_tr)
        print('VALID | hdbscan Clustering Acc: ', accuracy_val)

        del cluster_tr, cluster_val, accuracy_tr, accuracy_val
        gc.collect()


    def generate_samples(self, data, session, cur_epoch=''):
        # Generating W space
        print('Generating W space ...')
        w_en = self.encode(data.x)

        pca = PCA(n_components=2)
        W_pca = pca.fit_transform(w_en)
        print('W space dimensions: {}'.format(W_pca.shape))
        print('Ploting W space ...')
        w_space = self.summary_dir + '/{} W space in epoch {}.jpg'.format(self.summary_dir.split('/')[-1:][0], cur_epoch)
        self.w_space_files.append(w_space)
        print(w_space)
        plot_dataset(W_pca.compute(), y=data.labels, save=w_space)

        pca = PCA(n_components=3)
        W_pca = pca.fit_transform(w_en)
        print('W space dimensions: {}'.format(W_pca.shape))
        print('Ploting W space ...')
        w_space = self.summary_dir + '/{} W space 3d in epoch {}.jpg'.format(self.summary_dir.split('/')[-1:][0],
                                                                           cur_epoch)
        self.w_space3d_files.append(w_space)
        plot_dataset3d(W_pca.compute(), y=data.labels, save=w_space)

        del W_pca, w_en
        gc.collect()

        # Generating Samples
        print('Generating Samples ...')

        x_recons_l = self.reconst(data.samples)
        recons_file = self.summary_dir+'/{} samples generation in epoch {}.jpg'.format(self.summary_dir.split('/')[-1:][0], cur_epoch)
        self.recons_files.append(recons_file)
        #plot_samples(x_recons_l, scale=10, save=recons_file)
        
        del x_recons_l
        gc.collect()
        
    ''' 
    ------------------------------------------------------------------------------
                                         MODEL FUNCTIONS
    ------------------------------------------------------------------------------ 
    '''    
    def batch_function(self, func, p1):                      
        with tf.Session(graph=self.graph) as session:
            saver = tf.train.Saver()
            if(self.load(session, saver)):
                num_epochs_trained = self.model_graph.cur_epoch_tensor.eval(session)
                print('EPOCHS trained: ', num_epochs_trained)
            else:
                return

            output_l = list()
            
            start=0
            end= self.batch_size
            
            with tqdm(range(p1.shape[0]//self.batch_size)) as pbar:
                while end < p1.shape[0]:
                    output = func(session, p1[start:end])    
                    output = np.array(output)
                    output = output.reshape([output.shape[0] * output.shape[1]] + list(output.shape[2:]))
                    output_l.append(output)
        
                    start=end
                    end +=self.batch_size
                    pbar.update(1)
                else:
                        
                    x1 = p1[start:]
                    xsize = len(x1)
                    p1t = da.zeros([self.batch_size - xsize] + list(x1.shape[1:]))
                    
                    output = func(session, np.concatenate((x1, p1t), axis=0))
                    output = np.array(output)
                    output = output.reshape([output.shape[0] * output.shape[1]] + list(output.shape[2:]))[0:xsize]
                
                    output_l.append(output)
                    
                    pbar.update(1)
        
        try:
            return da.vstack(output_l)
        except:
            output_l = list(map(lambda l: l.reshape(-1,1), output_l
                   )
                    )
        return da.vstack(output_l)
