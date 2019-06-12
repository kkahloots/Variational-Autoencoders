""" VAECNN_graph.py: Tensorflow CNN Graph for the VAutoencoder """
__author__      = "Khalid M. Kahloot"
__copyright__   = "Copyright 2019, Only for professionals"

from .VAE_graph import VAEGraph
import tensorflow as tf
import numpy as np

from networks.conv_net import ConvNet3
from networks.deconv_net import DeconvNet3

class VAECNNGraph(VAEGraph):
    def __init__(self, network_params, act_out=tf.nn.softplus,
                 transfer_fct=tf.nn.relu, learning_rate=1e-4,
                 kinit=tf.contrib.layers.xavier_initializer(), batch_size=32,
                 reuse=None, dropout=0.2):
        
        super().__init__(network_params, act_out,
                         transfer_fct, learning_rate,
                         kinit, batch_size,
                         reuse, dropout)
        
    def build_graph(self):
        self.create_inputs()
        self.create_graph()
        self.create_loss_optimizer()

    '''  ------------------------------------------------------------------------------
                                     GRAPH FUNCTIONS
    ------------------------------------------------------------------------------ '''    

    def create_encoder(self, input_,hidden_dim,output_dim,num_layers,transfer_fct, \
                       act_out,reuse,kinit,bias_init, drop_rate):
        latent_ = ConvNet3(input_=input_,
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
    
    def create_decoder(self,input_, hidden_dim, output_dim, num_layers, \
                       transfer_fct, act_out,reuse,kinit,bias_init, drop_rate):
        recons_ = DeconvNet3(input_=input_,
                            num_layers=num_layers,
                            hidden_dim=hidden_dim,
                            output_dim=output_dim,
                            width=self.width, 
                            height=self.height, 
                            nchannels=self.nchannel,
                            transfer_fct=transfer_fct,
                            act_out=act_out, 
                            reuse=reuse, 
                            kinit=kinit,
                            bias_init=bias_init,
                            drop_rate=drop_rate)  
        return recons_
 

     
        
        

