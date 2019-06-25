#    +>AD`c+W4J/nJ*^L
#    autoencoders.project

__author__      = "Khalid M. Kahloot"
__copyright__   = "Copyright 2019, Only for professionals"

""" 
------------------------------------------------------------------------------
AE_BASE.py:  Autoencoder argument process
------------------------------------------------------------------------------
"""

import os
import sys
sys.path.append('..')

import utils.utils as utils
import utils.constants as const


class AE_BASE():
    '''  ------------------------------------------------------------------------------
                                         SET ARGUMENTS
        ---------------------------------------------------------------------------------- '''
    
    def __init__(self, dataset_name, alpha=1, beta=0.6, gamma=1, sigma=0.001, l2=1e-6, \
                       latent_dim=10, hidden_dim=100, num_layers=3,epochs=100, batch_size=32, \
                       checkpoint_dir = 'checkpoints', summary_dir='summary', result_dir='result', log_dir='log', \
                       dropout=0.2, batch_norm=True, l_rate=1e-05, restore=False, plot=False, \
                       clustering=False, colab=False, colabpath='',revision_hash=''):
        args=dict()
        args['model_type']=0
        args['model_name']='AE'
        args['dataset_name']=dataset_name
        args['revision_hash'] = revision_hash
        args['alpha']=alpha
        args['beta']=beta
        args['gamma']=gamma
        args['sigma']=sigma
        args['l2']=l2
        
        args['latent_dim']=latent_dim

        args['hidden_dim']=hidden_dim
        args['num_layers']=num_layers
        args['epochs']=epochs
        args['batch_size']=batch_size
        args['dropout']=dropout
        args['batch_norm']=batch_norm
        args['l_rate']=float(l_rate)
        args['train']=True if restore==False else False
  
        args['plot']=plot
        args['clustering'] = clustering
        args['restore']=restore
        args['early_stopping']=1

        args['checkpoint_dir'] = checkpoint_dir
        args['summary_dir'] = summary_dir
        args['result_dir'] = result_dir
        args['log_dir'] = log_dir
        args['colab'] = colab
        self.colabpath = colabpath

        self.config = utils.Config(args)
        self.isBuilt = False;

    def setup_logging(self):        
        self.experiments_root_dir = 'experiments'
        self.config.model_name = const.get_model_name(self.config.model_name, self.config)

        self.config.summary_dir = os.path.join(self.experiments_root_dir+"/"+ self.config.model_name+"_"+self.config.revision_hash+"/",self.config.log_dir)
        self.config.checkpoint_dir = os.path.join(self.experiments_root_dir+"/"+self.config.model_name+"_"+self.config.revision_hash+"/", self.config.checkpoint_dir)
        self.config.results_dir = os.path.join(self.experiments_root_dir+"/"+self.config.model_name+"_"+self.config.revision_hash+"/", self.config.result_dir)

        #Flags
        flags_list = ['train', 'restore', 'plot', 'clustering', 'early_stopping', 'colab','revision_hash']
        self.flags = utils.Config({ your_key: self.config.__dict__[your_key] for your_key in flags_list})
        
        # create the experiments dirs
        utils.create_dirs([self.config.summary_dir, self.config.checkpoint_dir, self.config.results_dir])
        utils.save_args(self.config.__dict__, self.config.summary_dir)

    def build(self,X,y=None):
        print('\n Processing data...')
        self.data_train, self.data_valid = utils.process_data(X, y)

        print('\n building a model...')
        self.buildModel()

        self.isBuilt=True

    def fit(self, X=None, y=None):
        '''  ------------------------------------------------------------------------------
                                         DATA PROCESSING
        ------------------------------------------------------------------------------ '''
        rebuild = False;
        if(not self.isBuilt):
            self.data_train, self.data_valid = utils.process_data(X, y)
            rebuild=True
        elif(not X is None or not y is None):
            print('\n Processing data...')
            new_data_train, new_data_valid = utils.process_data(X, y)
            print(type(self.data_train))
            print(type(new_data_train))
            if(self.data_train.shape[1:]!=new_data_train.shape[1:] or self.data_valid.shape[1:]!=new_data_valid.shape[1:]):
                print('\n Data shape changed, building a new model...')
                self.data_train, self.data_valid = new_data_train, new_data_valid
                rebuild=True
        else: # possibly different data that has the same shape, no rebuild required
            self.data_train, self.data_valid = new_data_train, new_data_valid

        if(rebuild):
            self.buildModel();
            self.isBuilt=True;
        '''  -------------------------------------------------------------------------------
                        GOOGLE COLAB 
        ------------------------------------------------------------------------------------- '''
        if self.flags.colab:
            self.push_colab()
            self.model.push_colab = self.push_colab



        '''  -------------------------------------------------------------------------------
                        TRAIN THE MODEL
        ------------------------------------------------------------------------------------- '''


        print('\n training a model...')
        self.model.train(self.data_train, self.data_valid, enable_es=self.flags.early_stopping)

    def test(self,X):
        if(not self.isBuilt):
            print("Connot evaluate an unbuilt model")
            return;
        data= utils.process_data_nosplit(X);
        self.model.test(data)

    def buildModel(self):
        '''  ------------------------------------------------------------------------------
                                     SET NETWORK PARAMS
        ------------------------------------------------------------------------------ '''        

        network_params_dict = dict()
        network_params_dict['input_height'] = self.data_train.height
        network_params_dict['input_width'] = self.data_train.width
        network_params_dict['input_nchannels'] = self.data_train.num_channels
        network_params_dict['train_size'] = self.data_train._ndata
        
        network_params_dict['hidden_dim'] =  self.config.hidden_dim
        network_params_dict['latent_dim'] =  self.config.latent_dim
        network_params_dict['l2'] =  self.config.l2
        network_params_dict['num_layers'] =  self.config.num_layers
        self.network_params = utils.Config(network_params_dict)

        self._build()    
     
    def _build(self):
        pass

    def zipdir(self, path, ziph):
        # ziph is zipfile handle
        for root, dirs, files in os.walk(path):
            for file in files:
                ziph.write(os.path.join(root, file))

    def zipExperiments(self):
        import zipfile as zf
        zipf = zf.ZipFile(self.config.model_name+self.config.revision_hash+'.zip', 'w', zf.ZIP_DEFLATED)
        self.zipdir(self.experiments_root_dir+'/', zipf)
        zipf.close()

    def push_colab(self):
        self.zipExperiments()
        self.colab2google()

    def colab2google(self):
        from google.colab import auth
        from googleapiclient.http import MediaFileUpload
        from googleapiclient.discovery import build


        file_name = self.config.model_name+'.zip'
        print('zip experiments {} ...'.format(file_name))
        file_path = './'+ file_name

        auth.authenticate_user()
        drive_service = build('drive', 'v3')

        print('uploading to google drive ...')
        file_metadata = {
            'name': file_name,
            'mimeType': 'application/octet-stream',
            'parents': [self.colabpath]
        }
        media = MediaFileUpload(file_path,
                                mimetype='application/octet-stream',
                                resumable=True)
        created = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        print('File ID: {}'.format(created.get('id')))