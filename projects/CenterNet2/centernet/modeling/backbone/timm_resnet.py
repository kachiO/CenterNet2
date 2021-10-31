import timm
from timm.data import resolve_data_config
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec
from detectron2.layers import FrozenBatchNorm2d
from detectron2.modeling.backbone.fpn import FPN
from .fpn_p5 import LastLevelP6P7_P5
from .bifpn import BiFPN

TIMM_PIXEL_MEAN = [0.485, 0.456, 0.406]
TIMM_PIXEL_STD = [0.229, 0.224, 0.225]

@BACKBONE_REGISTRY.register()
class ResNetTIMM(Backbone):
    def __init__(self, cfg,):
        super().__init__()
        stages = {"res2": 1, "res3": 2, "res4": 3, "res5": 4}
        out_inds = (stages[f] for f in cfg.MODEL.RESNETS.OUT_FEATURES)

        model = timm.create_model(f'resnet{cfg.MODEL.RESNETS.DEPTH}', 
                                  pretrained=cfg.MODEL.RESNETS.PRETRAINED, 
                                  features_only=True, 
                                  out_indices=out_inds)
        model = FrozenBatchNorm2d.convert_frozen_batchnorm(model)
        config = resolve_data_config({}, model=model)
        self.pixel_mean = list(config['mean'])
        self.pixel_std = list(config['std'])

        self.layer_map = {'layer1': 'res2',
                          'layer2': 'res3',
                          'layer3': 'res4',
                          'layer4': 'res5'}

        self.encoder = model

    def forward(self, image):
        return self.model(image)

    def output_shape(self):
        return {self.layer_map[info['module']]: ShapeSpec(channels=info['num_chs'], stride=info['reduction']) 
                for info in self.encoder.feature_info[1:]}


@BACKBONE_REGISTRY.register()
def build_p67_timm_resnet_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = ResNetTIMM(cfg)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelP6P7_P5(out_channels, out_channels),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone

@BACKBONE_REGISTRY.register()
def build_p35_resnet_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = ResNetTIMM(cfg)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
                    bottom_up=bottom_up,
                    in_features=in_features,
                    out_channels=out_channels,
                    norm=cfg.MODEL.FPN.NORM,
                    top_block=None,
                    fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
                )
    return backbone


@BACKBONE_REGISTRY.register()
def build_resnet_bifpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = ResNetTIMM(cfg)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    backbone = BiFPN(
                     cfg=cfg,
                     bottom_up=bottom_up,
                     in_features=in_features,
                     out_channels=cfg.MODEL.BIFPN.OUT_CHANNELS,
                     norm=cfg.MODEL.BIFPN.NORM,
                     num_levels=cfg.MODEL.BIFPN.NUM_LEVELS,
                     num_bifpn=cfg.MODEL.BIFPN.NUM_BIFPN,
                     separable_conv=cfg.MODEL.BIFPN.SEPARABLE_CONV,
                )
    return backbone
