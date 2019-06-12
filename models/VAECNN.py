"""AECNN.py:  CNN Autoencoder model"""
__author__      = "Khalid M. Kahloot"
__copyright__   = "Copyright 2019, Only for professionals"

import os
import tensorflow as tf
import numpy as np
import sys
sys.path.append('..')

from .AE_BASE import AE_BASE
import utils.utils as utils
import utils.constants as const

class VAECNN(AE_BASE):
    def __init__(self, *argz, **kwrds):
        super(VAECNN, self).__init__(*argz, **kwrds)
        self.config.model_name = 'VAECNN'
        self.config.model_type = const.VAECNN
        self.setup_logging()
        
    def _build(self):
        '''  ---------------------------------------------------------------------
                            COMPUTATION GRAPH (Build the model)
        ---------------------------------------------------------------------- '''
        from Alg_VAE.VAE_model import VAEModel
        self.model = VAEModel(self.network_params, act_out=utils.softplus_bias,
                             transfer_fct=tf.nn.relu, learning_rate=self.config.l_rate,
                             kinit=tf.contrib.layers.xavier_initializer(),
                             batch_size=self.config.batch_size, dropout=self.config.dropout, batch_norm=self.config.batch_norm,
                             epochs=self.config.epochs, checkpoint_dir=self.config.checkpoint_dir,
                             summary_dir=self.config.summary_dir, result_dir=self.config.results_dir,
                             restore=self.flags.restore, plot=self.flags.plot, model_type=self.config.model_type)
        print('building AECNN Model...')
        print('\nNumber of trainable paramters', self.model.trainable_count)
    
    def animate(self):
        return self.model.animate()

    '''  
    ------------------------------------------------------------------------------
                                         MODEL OPERATIONS
    ------------------------------------------------------------------------------ 
    '''    
              
    def encode(self, inputs):
        '''  ------------------------------------------------------------------------------
                                         DATA PROCESSING
        ------------------------------------------------------------------------------ '''           
        inputs = utils.prepare_dataset(inputs) 
        return self.model.encode(inputs)
        
    def decode(self, w):
        return self.model.decode(w)
     
    def interpolate(self, input1, input2):
        input1 = utils.prepare_dataset(input1)
        input2 = utils.prepare_dataset(input2)         
        return self.model.interpolate(input1, input2)

    def reconst_loss(self, inputs):
        '''  ------------------------------------------------------------------------------
                                         DATA PROCESSING
        ------------------------------------------------------------------------------ '''           
        inputs = utils.prepare_dataset(inputs) 
        return self.model.reconst_loss(inputs)        