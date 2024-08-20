import timm
from fastervit import create_model
import torch
import torch.nn as nn

def main():
    """
    for backbone in timm.list_models(pretrained=True):
        if "convnext" in backbone:
            print(backbone)
    """

    model = timm.create_model('faster_vit_0_any_res',
                              resolution=[256,256],
                          features_only=True,
                          pretrained=True,
                          
    )    
    #model.head=nn.Identity()
    import ipdb; ipdb.set_trace()
    features=model(torch.randn([1,3,256,256]))
    


if __name__=="__main__":
    main()