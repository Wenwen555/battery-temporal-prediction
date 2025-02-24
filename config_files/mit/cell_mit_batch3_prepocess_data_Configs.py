class Config(object):
    def __init__(self):
        # mlp model configs
        self.original_input_channels = 4
        self.original_seq_len = 300
        self.input_channels = 256
        self.hidden_channels = 128
        self.output_channels = 2
        self.seqlen = 41
        # channels for TC module.
        self.final_out_channels = 256
        self.num_classes = 1  # which means prediction or regression.
        self.dropout = 0.35

        self.kernel_size = 25
        self.stride = 3
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
        self.batch_size = 128

        self.Context_Cont = Context_Cont_configs()
        self.TC = TC()
        self.augmentation = augmentations()
        self.window_len = 5  # 可以尝试修改此值, 但是此参数仅针对TC_tuned.



class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 1.5
        self.jitter_scale_ratio_weak = 0.8
        self.jitter_scale_ratio_strong = 1.2
        self.jitter_ratio = 2

        self.jitter_ratio_weak = 0.8
        self.jitter_ratio_strong = 1.4
        self.max_seg = 4
        # self.max_seg = 12



class Context_Cont_configs(object):
    def __init__(self):
        self.temperature = 0.2
        self.use_cosine_similarity = True


class TC(object):
    def __init__(self):
        self.hidden_dim = 64
        self.timesteps = 40
