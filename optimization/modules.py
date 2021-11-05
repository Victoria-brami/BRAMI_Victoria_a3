import torch
import torch.nn as nn
from torch.nn import Sequential


class GraphNet(nn.Module):
    """ Creates a torch Module from a graph"""

    def __init__(self, module_graph):
        super(GraphNet, self).__init__()
        module_graph.apply_attributes(self)

    def forward(self, data):
        return Sequential(*self.seq)(data)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, data):
        # print("data shape", data.shape)
        batch, channels, width = data.shape
        return data.view(batch, channels * width)


class PlaceHolder(nn.Module):
    """ Placeholder to put in the graph, should never actually be part of a torch module tree"""

    def __init__(self):
        super(PlaceHolder, self).__init__()

    def forward(self, data):
        return None


class ParallelBlock(nn.Module):
    """ Given a list of sequentials with the same input sizes and output sizes, creates a Module that runs the sequences
     separately and merges them together, with a sum if merge='sum', with a channel concatenation if merge='cat' """

    def __init__(self, sequentials, merge='sum'):
        super(ParallelBlock, self).__init__()
        self.seqs = sequentials
        assert merge in ['sum', 'cat'], 'Argument merge={} is not within the possibilities [sum, cat]'.format(merge)
        self.merge = merge
        for seq_idx in range(len(self.seqs)):
            setattr(self, 'seq{}'.format(seq_idx), self.seqs[seq_idx])

    def forward(self, data):
        datas = torch.stack([seq(data) for seq in self.seqs])  # shape (branches, batch, channels, width, width)
        if self.merge == 'sum':
            datas = torch.sum(datas, 0)
            return datas
        elif self.merge == 'cat':
            datas = torch.transpose(datas, 0, 1)  # shape (batch, branches, channels, width, width)
            datas = torch.flatten(datas, 1, 2)  # shape (batch, branches*channels, width, width)
            return datas


class SkipBlock(nn.Module):
    """ Implements a DenseNet or a ResNet block. to be used by graph.apply_attributes.\n
    Given a list of modules to apply in sequence and the skips received by each layer,
    provides to the modules the sum of their input skip tensors.
    reception_skips: dictionary. keys: names of receiving module, item: list of names of sending modules"""

    def __init__(self, modules, module_names, reception_skips):
        super(SkipBlock, self).__init__()
        self.modules = modules
        for module_idx in range(len(modules)):
            setattr(self, module_names[module_idx], modules[module_idx])
        self.reception_skips = reception_skips
        self.module_names = module_names

    def forward(self, data):
        values = {'BLOCK_INPUT': data}  # values: inputs (item) to receiving modules of idx (key)

        for module_idx, module in enumerate(self.modules):
            " adding the sent skip connections to the input of the current module"
            module_name = self.module_names[module_idx]
            if module_name in self.reception_skips:  # if this module receives skips
                for skips_to_receive in self.reception_skips[self.module_names[module_idx]]:
                    data = data + values[skips_to_receive]
            data = module(data)
            values[module_name] = data

        return data
