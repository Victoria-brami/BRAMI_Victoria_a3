import numpy as np


class TrainDatasetConfig(object):
    """ Configuration of the training routine (params passed to the Dataset and DataLoader"""

    def __init__(self):
        self.data = "/gpfswork/rech/rnt/uuj49ar/bird_dataset"

        self.sigma = 7  # internal, changing is likely to break code or accuracy
        self.path_thickness = 1  # internal, changing is likely to break code or accuracy
        self.batch_size = 64  # please account for optuna's n_jobs
        self.num_workers = 4
        self.num_data = 1082


class ValDatasetConfig(object):
    """" Configuration of the validation routine"""

    def __init__(self):

        self.data = "/gpfswork/rech/rnt/uuj49ar/bird_dataset"

        train_batch_size = TrainDatasetConfig().batch_size
        train_num_workers = TrainDatasetConfig().num_workers
        self.batch_size = train_num_workers * (train_batch_size // (train_num_workers * 2))
        self.num_workers = train_num_workers



class NetConfig(object):
    """ Configuration of the network"""

    def __init__(self):
        self.resume_training = False
        batch_size = TrainDatasetConfig().batch_size
        self.init_chkp = '../experiment/optuna_training/run_B'+str(batch_size)+'.pth'
        self.max_channels = 64
        self.min_linear_layers = 0
        self.max_linear_layers = 3
        self.min_linear_unit_size = 8
        self.max_linear_unit_size = 256  # can be overriden for the input size of layer taking the flattenened image


class ExecutionConfig(object):
    """ Configuration of the training loop"""

    def __init__(self):
        self.epochs = 5 # 5
        self.chkp_folder = '../experiment/optuna_training/'
        self.gpus = 1 #1
        self.num_validation_sanity_steps = 0


class OptunaConfig(object):  # put None in suggest to use the default value
    """ Configuration of the Optuna study: what to optimise and by what means"""

    def __init__(self):
        # Computations for HyperBand configuration
        n_iters = int(TrainDatasetConfig().num_data * ExecutionConfig().epochs / TrainDatasetConfig().batch_size)
        reduction_factor = int(round(np.exp(np.log(n_iters) / 4)))  # for 5 brackets (see Optuna doc)

        self.n_jobs = 1  # number of parallel optimisations
        self.n_iters = n_iters
        self.reduction_factor = reduction_factor
        self.timeout = 24*3600  # 2*3600       # seconds, if None is in both limits, use CTRL+C to stop
        self.n_trials = 50  # 500          # will stop whenever the time or number of trials is reached
        self.pruner = 'Hyperband'  # options: Hyperband, Median, anything else -> no pruner

        self.suggest_optimiser = None  # ['SGD', 'Adam']default is hardcoded to Adam
        self.default_optimiser = 'Adam'

        self.suggest_learning_rate = [1e-5, 1e-2]
        self.default_learning_rate = 0.001

        self.suggest_weight_decay = None #[0, 1e-5]
        # self.default_weight_decay = 0.00936239234038259
        self.default_weight_decay = 0.00

        self.suggest_net_architecture = None # [0,4]
        self.default_net_architecture = 0
