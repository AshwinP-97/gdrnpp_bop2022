import torch
import torch.nn as nn
import torch.nn.functional as F

class AverageLayer(nn.Module):
    def __init__(self,threshold):

        super(AverageLayer,self).__init__()
        self.threshold=threshold
    
    
    def forward(self,region,region_bias,pnp,num_regions):
        
        new_region_size=num_regions//self.threshold
        new_region=[]
        new_region_bias=[]
        
        for i in range(0,region.shape[0],num_regions+1):
        
            reshape=region[i+1:i+1+num_regions,:,:,:].view(new_region_size,self.threshold,region.shape[1],region.shape[2],region.shape[3]
            )
            average=reshape.mean(dim=1)
            new_region.append(torch.cat((region[i].unsqueeze(0),average),dim=0))    


            reshape_bias=region_bias[i+1:i+1+num_regions].view(new_region_size,self.threshold)
            average_bias=reshape_bias.mean(dim=1)
            new_region_bias.append(torch.cat((region_bias[i].unsqueeze(0),average_bias),dim=0))

        
        reshape=pnp[:,-num_regions:,:,:].view(pnp.shape[0],new_region_size,self.threshold,pnp.shape[2],pnp.shape[3])
        avg_pnp=reshape.mean(dim=2)
        new_pnp=torch.cat((pnp[:,:pnp.shape[1]-num_regions,:,:],avg_pnp),dim=1)

        

        return [torch.cat(new_region,dim=0),torch.cat(new_region_bias,dim=0),new_pnp,new_region_size]



def compute_new_region(cfg,weight,bias,pnp):
    layer=AverageLayer(cfg.ABLATE.THRESHOLD)
    
    num_classes=cfg.MODEL.POSE_NET.NUM_CLASSES
    num_regions=cfg.MODEL.POSE_NET.GEO_HEAD.NUM_REGIONS
    region_param=num_classes*(num_regions+1)
    region_weight=weight[(-region_param):,:,:,:]
    region_bias=bias[(-region_param):,]
    out=layer(region_weight,region_bias,pnp,num_regions)
    out[0]=torch.cat((weight[:weight.shape[0]-region_param,:,:,:],out[0]),dim=0)
    out[1]=torch.cat((bias[:bias.shape[0]-region_param],out[1]),dim=0)

    return out





if __name__=='__main__':
    
    input=torch.randn(520,256,1,1)
    bias=torch.randn(520,1)
    pnp=torch.randn(256,69,3,3)
    layer=AverageLayer(2)
    import ipdb;ipdb.set_trace()
    out=layer(input,bias,pnp,64)
    


