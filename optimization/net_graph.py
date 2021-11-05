import torch.nn as nn
from torch.nn import Linear, Dropout
import optuna
from typing import Union
from module_graph import ModuleGraph
from modules import Flatten, PlaceHolder


from net_model import NetConfig

cfg = NetConfig()
NB_JOINTS = 14
INPUT_SIZE = [1, NB_JOINTS, 3 + 3 * NB_JOINTS]
OUTPUTS_DIM = 3 * NB_JOINTS


def suggest_graph(trial: Union[optuna.trial.Trial, optuna.trial.FrozenTrial]) -> ModuleGraph:
    """ Suggests a module graph in order to create a GraphNet model"""
    # if trial is a Trial object we are optimising, otherwise we are loading
    suggest = isinstance(trial, optuna.trial.Trial)
    min_linear_layers = cfg.min_linear_layers
    max_linear_layers = cfg.max_linear_layers
    min_linear_unit_size = cfg.min_linear_unit_size
    max_linear_unit_size = cfg.max_linear_unit_size

    graph = ModuleGraph(INPUT_SIZE)

    def suggest_linear_block(name, input_dim, activation):
        # adding the linear layer, only multiples of 8 to reduce the space and the size has to decrease
        output_dim = trial.suggest_int('{}_units'.format(name), min_linear_unit_size,
                                       8 * (min(input_dim, max_linear_unit_size) // 8), step=8) \
            if suggest else trial.params['{}_units'.format(name)]
        linear = Linear(in_features=input_dim, out_features=output_dim)
        graph.add_module("{}_linear".format(name), linear, [input_dim], [output_dim])
        # adding the dropout layer
        dropout = Dropout(linear_dropout)
        graph.add_module("{}_dropout".format(name), dropout, [output_dim], [output_dim])
        graph.add_module("{}_activation".format(name), activation, [output_dim], [output_dim])

    def suggest_activation(name):
        activations = {'relu': nn.ReLU(), 'gelu': nn.GELU(), 'prelu': nn.PReLU()}
        suggested = trial.suggest_categorical(name, list(activations.keys())) if suggest else trial.params[name]
        return activations[suggested]

    # We optimize the number of layers, hidden units in each layer and dropouts.
    if suggest:  # currently optimising the architecture, trial is a optuna.trial.Trial object
        n_linear_blocks = trial.suggest_int("n_linear_layers", min_linear_layers, max_linear_layers)
        linear_dropout = trial.suggest_float("linear_dropout", 0.0, 0.2)
    else:  # currently creating a model from a completed trial: optuna.trial.FrozenTrial object
        n_linear_blocks = trial.params['n_linear_layers']
        linear_dropout = trial.params['linear_dropout']


    # conv layers
    input_size = INPUT_SIZE
    input_dim = input_size[0] * input_size[1] * input_size[2]

    # Flatten layer
    output_size = [input_dim]
    graph.add_module('flatten', Flatten(), input_size, output_size)

    # linear layers
    linear_activation = suggest_activation('linear_activation')
    for layer_idx in range(n_linear_blocks):
        suggest_linear_block("linear_block{}".format(layer_idx), input_dim, linear_activation)
        input_dim = graph.modules[graph.last_name]['out'][0]  # here the size is [n]

    graph.add_module('last_layer', Linear(input_dim, OUTPUTS_DIM), [input_dim], [OUTPUTS_DIM])

    return graph
