from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import logging
import torch.utils.model_zoo as model_zoo
from copy import deepcopy
from functools import partial
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models import create_model
_logger = logging.getLogger(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
__all__ = [
    'dcstr_tiny_patch16_224',
    'dcstr_small_patch16_224',
    'dcstr_base_patch16_224',
]

def create_dcstr(num_tokens, model=None, checkpoint_path=''):
    dcstr = create_model(
        model,
        pretrained=True,
        num_classes=num_tokens,
        checkpoint_path=checkpoint_path)
    dcstr.reset_classifier(num_classes=num_tokens)
    return dcstr

class dcstr(VisionTransformer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ngram2 = nn.Conv1d(in_channels=197, out_channels=197, kernel_size=3, padding=2, stride=1, dilation=2)
        self.ngram3 = nn.Conv1d(in_channels=197, out_channels=197, kernel_size=3, padding=3, stride=1, dilation=3)
        self.ngram4 = nn.Conv1d(in_channels=197, out_channels=197, kernel_size=3, padding=4, stride=1, dilation=4)
        self.ngram5 = nn.Conv1d(in_channels=197, out_channels=197, kernel_size=3, padding=5, stride=1, dilation=5)
        self.maxpool = nn.MaxPool1d(kernel_size=14, stride=7)

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        gram2_output = self.ngram2(x)
        gram3_output = self.ngram3(x)
        gram4_output = self.ngram4(x)
        gram5_output = self.ngram5(x)
        gram_output = gram2_output + gram3_output + gram4_output + gram5_output
        for blk in self.blocks:
            multi_features = blk(gram_output)
        x1 = self.pos_drop(x)
        for blk in self.blocks:
            x1 = blk(x1)
        single_features = self.norm(x1)
        final_features = single_features + multi_features
        return final_features

    def forward(self, x, seqlen=25):
        x = self.forward_features(x)
        x = self.maxpool(x.transpose(1, 2)).transpose(1, 2)
        x = x[:, :seqlen]
        b, s, e = x.size()
        x = x.reshape(b * s, e)
        x = self.head(x)
        x = x.view(b, s, self.num_classes)
        return x

def load_pretrained(model, cfg=None, num_classes=1000, in_chans=1, filter_fn=None, strict=True):
    if cfg is None:
        cfg = getattr(model, 'default_cfg')
    if cfg is None or 'url' not in cfg or not cfg['url']:
        _logger.warning("Pretrained model URL is invalid, using random initialization.")
        return
    state_dict = model_zoo.load_url(cfg['url'], progress=True, map_location='cpu')
    if "model" in state_dict.keys():
        state_dict = state_dict["model"]

    if filter_fn is not None:
        state_dict = filter_fn(state_dict)

    if in_chans == 1:
        conv1_name = cfg['first_conv']
        _logger.info('Converting first conv (%s) pretrained weights from 3 to 1 channel' % conv1_name)
        key = conv1_name + '.weight'
        if key in state_dict.keys():
            _logger.info('(%s) key found in state_dict' % key)
            conv1_weight = state_dict[conv1_name + '.weight']
        else:
            _logger.info('(%s) key NOT found in state_dict' % key)
            return
        conv1_type = conv1_weight.dtype
        conv1_weight = conv1_weight.float()
        O, I, J, K = conv1_weight.shape
        if I > 3:
            assert conv1_weight.shape[1] % 3 == 0
            conv1_weight = conv1_weight.reshape(O, I // 3, 3, J, K)
            conv1_weight = conv1_weight.sum(dim=2, keepdim=False)
        else:
            conv1_weight = conv1_weight.sum(dim=1, keepdim=True)
        conv1_weight = conv1_weight.to(conv1_type)
        state_dict[conv1_name + '.weight'] = conv1_weight

    classifier_name = cfg['classifier']
    if num_classes == 1000 and cfg['num_classes'] == 1001:
        classifier_weight = state_dict[classifier_name + '.weight']
        state_dict[classifier_name + '.weight'] = classifier_weight[1:]
        classifier_bias = state_dict[classifier_name + '.bias']
        state_dict[classifier_name + '.bias'] = classifier_bias[1:]
    elif num_classes != cfg['num_classes']:
        del state_dict[classifier_name + '.weight']
        del state_dict[classifier_name + '.bias']
        strict = False

    print("Loading pre-trained vision transformer weights from %s ..." % cfg['url'])
    model.load_state_dict(state_dict, strict=strict)

def _conv_filter(state_dict, patch_size=16):
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict

@register_model
def dcstr_tiny_patch16_224(pretrained=False, **kwargs):
    kwargs['in_chans'] = 1
    model = dcstr(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True, **kwargs)
    model.default_cfg = _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth'
    )
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 1), filter_fn=_conv_filter)
    return model

@register_model
def dcstr_small_patch16_224(pretrained=False, **kwargs):
    kwargs['in_chans'] = 1
    model = dcstr(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, **kwargs)
    model.default_cfg = _cfg(
        url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth"
    )
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 1), filter_fn=_conv_filter)
    return model

@register_model
def dcstr_base_patch16_224(pretrained=False, **kwargs):
    kwargs['in_chans'] = 1
    model = dcstr(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, **kwargs)
    model.default_cfg = _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth'
    )
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 1), filter_fn=_conv_filter)
    return model