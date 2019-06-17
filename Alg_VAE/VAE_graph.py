"""VAE_graph.py: Tensorflow Graph for the VAutoencoder"""
__author__      = "Khalid M. Kahloot"
__copyright__   = "Copyright 2019, Only for professionals"

from base.base_graph import BaseGraph
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import utils.constants as const

from networks.dense_net import DenseNet

'''
This is the Main VAEGraph.
'''
class VAEGraph(BaseGraph):
    def __init__(self, network_params, act_out=tf.nn.softplus, sigma=0.001,
                 transfer_fct=tf.nn.relu, learning_rate=1e-4,
                 kinit=tf.contrib.layers.xavier_initializer(), batch_size=32,
                 reuse=None, dropout=0.2):
        
        super().__init__(learning_rate)
        
        self.width = network_params['input_width']
        self.height = network_params['input_height']
        self.nchannel = network_params['input_nchannels']
        
        self.hidden_dim = network_params['hidden_dim']
        self.latent_dim = network_params.latent_dim    
        self.num_layers = network_params['num_layers'] # Num of Layers in P(x|z)
        self.l2 = network_params.l2
        self.dropout = dropout
        self.sigma = sigma
        self.learning_rate = learning_rate
        
        self.act_out = act_out # Actfunc for NN modeling
        
        self.x_flat_dim = self.width * self.height * self.nchannel
        
        self.transfer_fct = transfer_fct
        self.kinit = kinit
        self.bias_init = tf.constant_initializer(0.0)
        self.batch_size = batch_size

        self.reuse = reuse

    def build_graph(self):
        self.create_inputs()
        self.create_graph()
        self.create_loss_optimizer()
    
    def create_inputs(self):
        with tf.variable_scope('inputs', reuse=self.reuse):
            self.x_batch = tf.placeholder(tf.float32, [self.batch_size, self.width, self.height, self.nchannel], name='x_batch')
            self.x_batch_flat = tf.reshape(self.x_batch , [-1,self.x_flat_dim])
            
            self.w_batch = tf.placeholder(tf.float32, [self.batch_size, self.latent_dim], name='w_batch')
            self.lr = tf.placeholder_with_default(self.learning_rate, shape=None, name='lr')

    ''' 
    ------------------------------------------------------------------------------
                                     GRAPH FUNCTIONS
    ------------------------------------------------------------------------------ 
    '''    
        
    def create_graph(self):
        print('\n[*] Defining encoder...')
        
        with tf.variable_scope('encoder_mean', reuse=self.reuse):
            Qz_x_mean = self.create_encoder(input_=self.x_batch_flat,
                            hidden_dim=self.hidden_dim, 
                            output_dim=self.latent_dim, 
                            num_layers=self.num_layers, 
                            transfer_fct=self.transfer_fct,
                            act_out=None, 
                            reuse=self.reuse, 
                            kinit=self.kinit,
                            bias_init=self.bias_init,
                            drop_rate=self.dropout)
        
            self.encoder_mean = Qz_x_mean.output

        with tf.variable_scope('encoder_var', reuse=self.reuse):
            Qz_x_var = self.create_encoder(input_=self.x_batch_flat,
                                            hidden_dim=self.hidden_dim,
                                            output_dim=self.latent_dim,
                                            num_layers=self.num_layers,
                                            transfer_fct=self.transfer_fct,
                                            act_out=tf.nn.softplus,
                                            reuse=self.reuse,
                                            kinit=self.kinit,
                                            bias_init=self.bias_init,
                                            drop_rate=self.dropout)

            self.encoder_var  = Qz_x_var.output

            print('\n[*] Reparameterization trick...')
            self.encoder_logvar = tf.log(self.encoder_var + const.epsilon)
            eps = tf.random_normal((self.batch_size, self.latent_dim), 0, 1, dtype=tf.float32)
            self.latent = tf.add(self.encoder_mean, tf.multiply(tf.sqrt(self.encoder_var), eps))

            self.w_batch = self.latent
            
        print('\n[*] Defining decoder...')
        with tf.variable_scope('decoder_mean', reuse=self.reuse):
            Px_w_mean = self.create_decoder(input_=self.w_batch,
                                            hidden_dim=self.hidden_dim,
                                            output_dim=self.x_flat_dim,
                                            num_layers=self.num_layers,
                                            transfer_fct=self.transfer_fct,
                                            act_out=tf.nn.sigmoid,
                                            reuse=self.reuse,
                                            kinit=self.kinit,
                                            bias_init=self.bias_init,
                                            drop_rate=self.dropout)
        
            self.decoder_mean_flat = Px_w_mean.output
            ##self.decoder_x_flat = Px_w_mean.output
        eps = tf.random_normal(tf.shape(self.decoder_mean_flat), 0, 1, dtype=tf.float32)
        self.decoder_x_flat = tf.add(self.decoder_mean_flat, tf.multiply(tf.sqrt(self.sigma), eps))
        self.decoder_x = tf.reshape(self.decoder_x_flat , [-1,self.width, self.height, self.nchannel])


    '''  
    ------------------------------------------------------------------------------
                                     ENCODER-DECODER
    ------------------------------------------------------------------------------ 
    '''          
    def create_encoder(self, input_,hidden_dim,output_dim,num_layers,transfer_fct, \
                       act_out,reuse,kinit,bias_init, drop_rate):
        latent_ = DenseNet(input_=input_,
                            hidden_dim=hidden_dim, 
                            output_dim=output_dim, 
                            num_layers=num_layers, 
                            transfer_fct=transfer_fct,
                            act_out=act_out, 
                            reuse=reuse, 
                            kinit=kinit,
                            bias_init=bias_init,
                            drop_rate=drop_rate)        
        return latent_
    
    def create_decoder(self,input_,hidden_dim,output_dim,num_layers,transfer_fct, \
                       act_out,reuse,kinit,bias_init, drop_rate):
        recons_ = DenseNet(input_=input_,
                            hidden_dim=hidden_dim, 
                            output_dim=output_dim, 
                            num_layers=num_layers, 
                            transfer_fct=transfer_fct,
                            act_out=act_out, 
                            reuse=reuse, 
                            kinit=kinit,
                            bias_init=bias_init,
                            drop_rate=drop_rate)        
        return recons_

    '''  
    ------------------------------------------------------------------------------
                                     LOSSES
    ------------------------------------------------------------------------------ 
    '''
    
    def create_loss_optimizer(self):
        print('[*] Defining Loss Functions and Optimizer...')
        with tf.name_scope('reconstruct'):
            self.reconstruction =   self.get_ell(self.x_batch_flat, self.decoder_mean_flat)
            #self.reconstruction = (0.5 / self.sigma) * self.get_ell(self.x_batch_flat, self.decoder_x_flat)
            #self.reconstruction = -(0.5/self.sigma) * tf.reduce_sum(tf.square(self.decoder_x_flat- self.x_batch_flat), 1)
        self.loss_reconstruction_m = tf.reduce_mean(self.reconstruction)

        with tf.variable_scope("L2_loss", reuse=self.reuse):
            tv = tf.trainable_variables()
            self.L2_loss = tf.reduce_sum([ tf.nn.l2_loss(v) for v in tv ])
        
        with tf.variable_scope("ae_loss", reuse=self.reuse):
            self.ae_loss = tf.reduce_mean(self.reconstruction)  + self.l2*self.L2_loss # shape = [None,]

        with tf.variable_scope("kl_loss", reuse=self.reuse):
            self.kl_loss = self.get_kl(self.encoder_mean, self.encoder_logvar)
        self.kl_loss_m = tf.reduce_mean(self.kl_loss)

        with tf.variable_scope("vae_loss", reuse=self.reuse):
            self.vae_loss = self.ae_loss  + self.kl_loss_m

        with tf.variable_scope("optimizer" ,reuse=self.reuse):
            self.optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_step = self.optimizer.minimize(self.vae_loss, global_step=self.global_step_tensor)

    ## ------------------- LOSS: EXPECTED LOWER BOUND ----------------------
    def get_ell(self, x, x_recons):
        """
        Returns the expected log-likelihood of the lower bound.
        For this we use a bernouilli LL.
        """
        # p(x|w)
        return - tf.reduce_sum((x) * tf.log(x_recons + const.epsilon) +
                               (1 - x) * tf.log(1 - x_recons + const.epsilon), 1)
                               

    def get_kl(self, mu, log_var):
        """
        d_kl(q(z|x)||p(z)) returns the KL-divergence between the prior p and the variational posterior q.
        :return: KL divergence between q and p
        """
        # Formula: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        return - 0.5 * tf.reduce_sum( 1.0 + 2.0 * log_var - tf.square(mu) - tf.exp(2.0 * log_var), 1)

    '''  
    ------------------------------------------------------------------------------
                                     FIT & EVALUATE TENSORS
    ------------------------------------------------------------------------------ 
    '''
    
    def partial_fit(self, session, x):
        tensors = [self.train_step, self.vae_loss, self.ae_loss, self.loss_reconstruction_m, self.kl_loss_m, self.L2_loss]
        feed_dict = {self.x_batch: x}
        _, vae_loss, ae_loss, recons, kl_loss, L2_loss  = session.run(tensors, feed_dict=feed_dict)

        return vae_loss, ae_loss, recons, kl_loss, L2_loss
    
    def evaluate(self, session, x):
        tensors = [self.vae_loss, self.ae_loss, self.loss_reconstruction_m, self.kl_loss_m, self.L2_loss]
        feed_dict = {self.x_batch: x}
        vae_loss, ae_loss, recons, kl_loss, L2_loss  = session.run(tensors, feed_dict=feed_dict)
        return vae_loss, ae_loss, recons, kl_loss, L2_loss

    '''  
    ------------------------------------------------------------------------------
                                     GENERATE LATENT and RECONSTRUCT
    ------------------------------------------------------------------------------ 
    '''
        
    def reconst_loss(self, session, x):
        tensors= [self.reconstruction] 
        feed = {self.x_batch: x}  
        return session.run(tensors, feed_dict=feed) 

    def decay_lr(self, session):
        self.lr = tf.multiply(0.1, self.lr)
        nlr = session.run(self.lr)

        if nlr > const.min_lr:
            print('decaying learning rate ... ')

            tensors = [self.lr]
            feed_dict = {self.lr: nlr}
            nlr = session.run(tensors, feed_dict=feed_dict)[0]
            nlr = session.run(self.lr)
            nlr = round(nlr, 8)
            print('new learning rate: {}'.format(nlr))

        
    '''  
    ------------------------------------------------------------------------------
                                         GRAPH OPERATIONS
    ------------------------------------------------------------------------------ 
    '''    
              
    def encode(self, session, inputs):
        tensors = [self.latent]
        feed_dict = {self.x_batch: inputs}
        return session.run(tensors, feed_dict=feed_dict)
        
    def decode(self, session, w):
        tensors = [self.decoder_x]
        feed_dict = {self.latent: w}
        return session.run(tensors, feed_dict=feed_dict) 
    
