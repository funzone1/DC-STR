import torch
import torch.nn as nn
from dcstr import create_dcstr
import math
from timm.models.vision_transformer import VisionTransformer

class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.dctstr= create_dcstr(num_tokens=opt.num_class, model=opt.TransformerModel)

    def forward(self, input, text, is_train=True, seqlen=25):
        prediction = self.dctstr(input, seqlen=seqlen)
        return prediction