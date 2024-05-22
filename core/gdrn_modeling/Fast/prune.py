import torch 
from lib.torch_utils.layers.conv_module import ConvModule
import torch.nn as nn
import os
import os.path as osp
from Fast.Region_ablate import compute_new_region

def prune(cfg,model,args,output,threshold_geo=0,threshold_pnp=0):
    new_cfg=cfg
    new_cfg.FAST.MODEL_CFG.PRUNE=True
    new_cfg.FAST.THRESHOLD_GEO=threshold_geo
    new_cfg.FAST.THRESHOLD_PNP=threshold_pnp
    model=model.cpu()
    new_state=model.state_dict()
    
    geo_head_net=model.geo_head_net
    layer=0
    geo_head_net_name='geo_head_net'
    
    with torch.no_grad():
        for module in geo_head_net.features:
            
            if isinstance(module,nn.ConvTranspose2d):
                
                index=l1_norm_conv(module.weight.data.transpose(0,1),threshold_geo)
                new_cfg.FAST.MODEL_CFG.GEO_HEAD_feat=len(index)
                
            if isinstance(module,nn.GroupNorm):
                num_channels=len(index)
                num_groups=int(num_channels/8)
                new_cfg.FAST.MODEL_CFG.GEO_HEAD_num_groups=num_groups
                norm=True
            if isinstance(module,nn.UpsamplingBilinear2d) or isinstance(module,nn.GELU):
                layer+=1
                continue
            name=geo_head_net_name+".features.{layer}".format(layer=layer)
           
            if isinstance(module,ConvModule):
                
                name=geo_head_net_name+".features.{layer}".format(layer=layer)
                for conv_module in module.children():
                    
                    if isinstance(conv_module,nn.Conv2d):
                        conv_in_channels=len(index)
                        list=[".conv",".norm",".gn"]
                        index2=l1_norm_conv(conv_module.weight,threshold_geo)
                        conv_out_channels=len(index2)
                        norm=False
                        for i in range(len(list)):
                            if  i!=0:
                                norm=True
                                index2=None
                            new_state=update_state(new_state,name+list[i],index,layer,index2,norm)                                            
                    elif isinstance(conv_module,nn.GroupNorm):
                        
                        num_channels=len(index)
                        num_groups=int(num_channels/8)
                    else:

                        layer+=1
                        break
                continue
                
            
            new_state=update_state(new_state,name,index,layer)
            layer+=1
        
        
        in_channels=len(index)
        #index2=l1_norm_conv(geo_head_net.out_layer.weight,threshold_geo)
        index2=None
        name=geo_head_net_name+".out_layer"
        new_state=update_state(new_state,name,index,layer,index2)

        if cfg.ABLATE.FLAG:
            
            weight='geo_head_net.out_layer.weight'
            bias='geo_head_net.out_layer.bias'
            pnp='pnp_net.features.0.weight'
            regions=new_cfg.MODEL.POSE_NET.GEO_HEAD.NUM_REGIONS
            new_state[weight],new_state[bias],new_state[pnp],new_cfg.MODEL.POSE_NET.GEO_HEAD.NUM_REGIONS \
                                        =compute_new_region(cfg,new_state[weight],new_state[bias],new_state[pnp])
        #out_channels=len(index2)

        
        model_pnp=model.pnp_net
        pnp_net='pnp_net.'
        pnp_net_name=pnp_net+'features'
        layer=0
        
        
        for module in model_pnp.features:
            
            name=pnp_net_name+".{layer}".format(layer=layer)
            if  isinstance(module,nn.Conv2d):
                
                if layer==0:
                    index=l1_norm_conv_pnp(module.weight,threshold_pnp)
                    out_channels=len(index)
                    new_cfg.FAST.MODEL_CFG.PNP_NET_Input=out_channels
                    index2=None
                else:
                    in_channels=out_channels
                    index2=l1_norm_conv_pnp(module.weight,threshold_pnp)
                    out_channels=len(index2)
                    norm=False
                

            elif isinstance(module,nn.GroupNorm):
                num_channels=len(index)
                num_groups=int(num_channels/4)
                new_cfg.FAST.MODEL_CFG.PNP_NET_num_groups=num_groups
                norm=True
                
            else:
                layer+=1
                continue
            
            new_state=update_pnp_state(new_state,name,index,layer,index2,norm)
            layer+=1

        in_features=len(index)*8*8
        new_state[pnp_net+'fc1.weight']=new_state[pnp_net+'fc1.weight'][:,index*64]
        index=l1_norm_fc(model_pnp.fc1.weight,threshold_pnp)
        new_state[pnp_net+'fc1.weight']=new_state[pnp_net+'fc1.weight'][index,:]
        new_state[pnp_net+'fc1.bias']=new_state[pnp_net+'fc1.bias'][index]
        out_features=len(index)
        new_cfg.FAST.MODEL_CFG.PNP_NET_fc_out=out_features

        
        new_state[pnp_net+'fc2.weight']=new_state[pnp_net+'fc2.weight'][:,index]
        in_features=model_pnp.fc1.out_features
        index=l1_norm_fc(model_pnp.fc2.weight,threshold_pnp)
        new_state[pnp_net+'fc2.weight']=new_state[pnp_net+'fc2.weight'][index,:]
        new_state[pnp_net+'fc2.bias']=new_state[pnp_net+'fc2.bias'][index]
        out_features=len(index)
        new_cfg.FAST.MODEL_CFG.PNP_NET_fc2_out=out_features


        new_state[pnp_net+'fc_r.weight']=new_state[pnp_net+'fc_r.weight'][:,index]
        new_state[pnp_net+'fc_t.weight']=new_state[pnp_net+'fc_t.weight'][:,index]
        fc_r_in_features=model_pnp.fc2.out_features
        fc_t_in_features=model_pnp.fc2.out_features


        model_path=os.path.join(output,"prune_l1_geo{threshold}_pnp{threshold2}_region{threshold3}.pth"
                    .format(threshold=threshold_geo,threshold2=threshold_pnp,threshold3=new_cfg.MODEL.POSE_NET.GEO_HEAD.NUM_REGIONS))
        new_cfg.FAST.WEIGHTS=model_path
        model.to(torch.device('cuda'))
        if os.path.isdir(output):
            torch.save(new_state,model_path) 
        output_dir="/gdrnpp_bop2022/configs/gdrn/lmo_pbr"
        path = osp.join(output_dir, 'l1_geo{threshold}_pnp{threshold2}_region{threshold3}.py'
                            .format(threshold=threshold_geo,threshold2=threshold_pnp,threshold3=new_cfg.MODEL.POSE_NET.GEO_HEAD.NUM_REGIONS))
        new_cfg.FAST.THRESHOLD_GEO=threshold_geo
        new_cfg.FAST.THRESHOLD_PNP=threshold_pnp
        new_cfg.dump(path)
    
    return cfg

def l1_norm_conv(param,threshold):
    reshape=param.reshape(param.size(0),-1)
    l1_norm,index=torch.sort(torch.norm(reshape,p=1,dim=1))
    return index[:int(index.shape[0]-(8*threshold))].tolist()
def l1_norm_conv_pnp(param,threshold):
    reshape=param.reshape(param.size(0),-1)
    l1_norm,index=torch.sort(torch.norm(reshape,p=1,dim=1))
    return index[:int(index.shape[0]-(4*threshold))].tolist()
def l1_norm_fc(param,threshold):
    l1_norm,index=torch.sort(torch.norm(param,p=1,dim=1))
    return index[:int(index.shape[0]-(4*threshold))].tolist()


def update_state(state,name,index,layer,index2=None,norm=False,):
    
    key=name+".weight"
    if layer ==0 or layer>10:
        print("1")
        state[key]=state[key][:,index,:,:]

    elif norm == True or layer ==1:
        key2=name+".bias"
        state[key]=state[key][index]
        state[key2]=state[key2][index]
    elif index2!=None:
        print("2")
        state[key]=state[key][index2,:,:,:]
        state[key]=state[key][:,index,:,:]
        #if layer>10:
        #    key2=name+".bias"
        #    state[key2]=state[key2][index2]
    return state
def update_pnp_state(state,name,index,layer,index2=None,norm=False):
    
    key=name+".weight"
    if layer ==0:
        print("1")
        state[key]=state[key][index,:,:,:]

    elif norm == True or layer ==1:
        key2=name+".bias"
        state[key]=state[key][index]
        state[key2]=state[key2][index]
    elif index2!=None:
        print("2")
        state[key]=state[key][index2,:,:,:]
        state[key]=state[key][:,index,:,:]
    return state

"""
def prune(cfg,model,args,output,threshold=1):
    
    cfg.FAST.MODEL_CFG.PRUNE=True
    model=model.cpu()
    new_state=model.state_dict()
    
    geo_head_net=model.geo_head_net
    layer=0
    geo_head_net_name='geo_head_net'
    
    with torch.no_grad():
        for module in geo_head_net.features:
            
            if isinstance(module,nn.ConvTranspose2d):
                
                index=l1_norm_conv(module.weight.data.transpose(0,1),threshold)
                geo_head_net.features[layer]=nn.ConvTranspose2d(in_channels=module.in_channels,out_channels=len(index),kernel_size=module.kernel_size,stride=module.stride,padding=module.padding,output_padding=module.output_padding,bias=module.bias)
                cfg.FAST.MODEL_CFG.GEO_HEAD_feat=len(index)
                
            if isinstance(module,nn.GroupNorm):
                num_channels=len(index)
                num_groups=int(num_channels/8)
                cfg.FAST.MODEL_CFG.GEO_HEAD_num_groups=num_groups
                geo_head_net.features[layer]=nn.GroupNorm(num_groups=num_groups,num_channels=num_channels,eps=module.eps,affine=module.affine)
                norm=True
            if isinstance(module,nn.UpsamplingBilinear2d) or isinstance(module,nn.GELU):
                layer+=1
                continue
            name=geo_head_net_name+".features.{layer}".format(layer=layer)
            '''
            if isinstance(module,nn.Conv2d):
                
                name=geo_head_net_name+".out_layer."
                in_channels=len(index)
                index=l1_norm_conv(module.weight,threshold)
                out_channels=len(index)
                geo_head_net.features[layer]=nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=module.kernel_size,stride=module.stride)                      
            '''
            
            
            if isinstance(module,ConvModule):
                
                name=geo_head_net_name+".features.{layer}".format(layer=layer)
                for conv_module in module.children():
                    
                    if isinstance(conv_module,nn.Conv2d):
                        conv_in_channels=len(index)
                        list=[".conv",".norm",".gn"]
                        index2=l1_norm_conv(conv_module.weight,threshold)
                        conv_out_channels=len(index2)
                        norm=False
                        for i in range(len(list)):
                            if  i!=0:
                                norm=True
                                index2=None
                            new_state=update_state(new_state,name+list[i],index,layer,index2,norm)
                        
                        geo_head_net.features[layer].conv=nn.Conv2d(in_channels=conv_in_channels,out_channels=conv_out_channels,kernel_size=conv_module.kernel_size,stride=conv_module.stride,padding=conv_module.padding,bias=conv_module.bias)
                                            
                    elif isinstance(conv_module,nn.GroupNorm):
                        
                        num_channels=len(index)
                        num_groups=int(num_channels/8)
                        if i==0:
                            geo_head_net.features[layer].norm=nn.GroupNorm(num_channels=num_channels,num_groups=num_groups,eps=conv_module.eps,affine=conv_module.affine)
                            i+=1
                        else:
                            geo_head_net.features[layer].gn=nn.GroupNorm(num_channels=num_channels,num_groups=num_groups,eps=conv_module.eps,affine=conv_module.affine)
                            i=0

                    else:

                        layer+=1
                        break
                continue
                
            
            new_state=update_state(new_state,name,index,layer)
            layer+=1
        
        
        in_channels=len(index)
        #index2=l1_norm_conv(geo_head_net.out_layer.weight,threshold)
        index2=None
        name=geo_head_net_name+".out_layer"
        new_state=update_state(new_state,name,index,layer,index2)
        #out_channels=len(index2)
        geo_head_net.out_layer=nn.Conv2d(in_channels=in_channels,out_channels=geo_head_net.out_layer.out_channels,kernel_size=geo_head_net.out_layer.kernel_size,stride=geo_head_net.out_layer.stride)


    

    
    model_pnp=model.pnp_net
    pnp_net='pnp_net.'
    pnp_net_name=pnp_net+'features'
    layer=0
    
    
    for module in model_pnp.features:
        
        name=pnp_net_name+".{layer}".format(layer=layer)
        if  isinstance(module,nn.Conv2d):
            
            if layer==0:
                index=l1_norm_conv(module.weight,threshold)
                out_channels=len(index)
                cfg.FAST.MODEL_CFG.PNP_NET_Input=out_channels
                index2=None
                model_pnp.features[layer]=nn.Conv2d(in_channels=module.in_channels,out_channels=out_channels,kernel_size=module.kernel_size,stride=module.stride,padding=module.padding,bias=module.bias)
            else:
                in_channels=out_channels
                index2=l1_norm_conv(module.weight,threshold)
                out_channels=len(index2)
                norm=False
                model_pnp.features[layer]=nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=module.kernel_size,stride=module.stride,padding=module.padding,bias=module.bias)
            

        elif isinstance(module,nn.GroupNorm):
            num_channels=len(index)
            num_groups=int(num_channels/4)
            cfg.FAST.MODEL_CFG.PNP_NET_num_groups=num_groups
            norm=True
            model_pnp.features[layer]=nn.GroupNorm(num_channels=num_channels,num_groups=num_groups,eps=module.eps,affine=module.affine)

            
        else:
            layer+=1
            continue
        
        new_state=update_pnp_state(new_state,name,index,layer,index2,norm)
        layer+=1

    in_features=len(index)*8*8
    new_state[pnp_net+'fc1.weight']=new_state[pnp_net+'fc1.weight'][:,index*64]
    index=l1_norm_fc(model_pnp.fc1.weight,threshold)
    new_state[pnp_net+'fc1.weight']=new_state[pnp_net+'fc1.weight'][index,:]
    new_state[pnp_net+'fc1.bias']=new_state[pnp_net+'fc1.bias'][index]
    out_features=len(index)
    cfg.FAST.MODEL_CFG.PNP_NET_fc_out=out_features
    model_pnp.fc1=nn.Linear(in_features=in_features,out_features=out_features,bias=True)

    
    new_state[pnp_net+'fc2.weight']=new_state[pnp_net+'fc2.weight'][:,index]
    in_features=model_pnp.fc1.out_features
    index=l1_norm_fc(model_pnp.fc2.weight,threshold)
    new_state[pnp_net+'fc2.weight']=new_state[pnp_net+'fc2.weight'][index,:]
    new_state[pnp_net+'fc2.bias']=new_state[pnp_net+'fc2.bias'][index]
    out_features=len(index)
    cfg.FAST.MODEL_CFG.PNP_NET_fc2_out=out_features
    model_pnp.fc2=nn.Linear(in_features=in_features,out_features=out_features,bias=True)


    new_state[pnp_net+'fc_r.weight']=new_state[pnp_net+'fc_r.weight'][:,index]
    new_state[pnp_net+'fc_t.weight']=new_state[pnp_net+'fc_t.weight'][:,index]
    fc_r_in_features=model_pnp.fc2.out_features
    fc_t_in_features=model_pnp.fc2.out_features
    model_pnp.fc_r=nn.Linear(in_features=fc_r_in_features,out_features= model_pnp.fc_r.out_features,bias=True)
    model_pnp.fc_t=nn.Linear(in_features=fc_t_in_features,out_features=model_pnp.fc_t.out_features,bias=True)


    model_path=os.path.join(output,"prune_l1_{threshold}.pth".format(threshold=threshold))
    cfg.FAST.WEIGHTS=model_path
    model.to(torch.device('cuda'))
    model.load_state_dict(new_state)
    if os.path.isdir(output):
        torch.save(new_state,model_path) 
    output_dir="/gdrnpp_bop2022/configs/gdrn/lmo_pbr"
    path = osp.join(output_dir, 'l1_{threshold}.py'.format(threshold=threshold))
    cfg.dump(path)
    
return cfg
"""