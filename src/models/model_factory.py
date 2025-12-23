from models.registry import BACKBONE
from models.registry import CLASSIFIER
from models.registry import LOSSES


model_dict = {
    'resnet50': 1024,
    'swin_t': 768,
    'swin_s': 768,
    'swin_b': 1024,
    'convnext_tiny': 768,
    'convnext_small': 768,
    'convnext_base': 1024,
    'convnext_large': 1536,
    'convnext_xlarge': 2048

}
def build_backbone(key, multi_scale=False):

    print("backbone keys")
    print(BACKBONE.keys())
    print("backbone values")
    print(BACKBONE.values())
    model = BACKBONE[key]()
    output_d = model_dict[key]

    return model, output_d


def build_classifier(key):

    return CLASSIFIER[key]


def build_loss(key):

    return LOSSES[key]
