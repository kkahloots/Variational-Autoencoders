
AE = 0
AECNN = 1


VAE = 2
VAECNN = 3

# Stopping tolerance
tol = 1e-8
min_lr = 1e-8
epsilon = 1e-8
SAVE_EPOCH=20

def get_model_name(model, config):
    if model in ['AE', 'AECNN']:
        return get_model_name_AE(model, config)
    elif model in ['VAE', 'VAECNN']:
        return get_model_name_AE(model, config) + '_' \
               + 'sigma' + str(config.sigma).replace('.','')
        
def get_model_name_AE(model, config):
    model_name = model + '_' \
                 + config.dataset_name+ '_'  \
                 + 'latent_dim' + str(config.latent_dim) + '_' \
                 + 'h_dim' + str(config.hidden_dim)  + '_' \
                 + 'h_nl' + str(config.num_layers)
    return model_name
