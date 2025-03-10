class Config(object):
    def __init__(self):

        # cnn model configs
        self.cnn_input_channels_1 = 3
        self.cnn_output_channels_1 = 64

        self.cnn_input_channels_2 = 64
        self.cnn_output_channels_2 = 128

        self.cnn_input_channels_3 = 128
        self.cnn_output_channels_3 = 256
        
        # mlp model configs
        self.mlp_input_channels = 3
        self.original_seq_len = 1101
        self.hidden_channels = 128
        self.mlp_dropout = 0.2
        self.mlp_final_output = 256


        # channels for TC module.
        self.final_out_channels = 256
        self.num_classes = 1  # which means prediction or regression.
        self.dropout = 0.35
        self.features_len = 63
        
        # training configs
        self.num_epoch = 150

        # optimizer parameters
        self.optimizer = 'adam'
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-4

        # data parameters
        self.drop_last = True
        self.k_fold = 5
        self.batch_size = 64

        self.Context_Cont = Context_Cont_configs()
        self.TC = TC()
        self.augmentation = augmentations()



class augmentations(object):
    def __init__(self):

        self.jitter_ratio_weak = 0.4
        self.jitter_ratio_strong = 0.8



class Context_Cont_configs(object):
    def __init__(self):
        self.temperature = 0.2
        self.use_cosine_similarity = True


class TC(object):
    def __init__(self):
        self.hidden_dim = 64
        self.timesteps = 8
