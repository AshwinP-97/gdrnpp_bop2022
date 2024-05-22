from ptflops import get_model_complexity_info
import torch
from collections import OrderedDict
import csv
import os

keys=["THRESHOLD","GMACS_BACKBONE","PARAMETERS_BACKBONE","GMACS_GEO_HEAD","PARAMETERS_GEO_HEAD","GMACS_PNP_HEAD","PARAMETERS_PNP_HEAD"]
ordered_dict=OrderedDict((key,None) for key in keys)




def get_gflops(model,threshold,save=True):
    
    macs_backbone,params_backbone=get_model_complexity_info(model.backbone,(8,3,256,256),input_constructor=prepate_input_backbone,as_strings=False,print_per_layer_stat=False,verbose=True)
    macs_geo,params_geo=get_model_complexity_info(model.geo_head_net,(8,1024,8,8),input_constructor=prepate_input_ghead,as_strings=False,print_per_layer_stat=False,verbose=True)
    macs_pnp,params_pnp=get_model_complexity_info(model.pnp_net,((8,5,64,64),(8,64,64,64),(8,3)),input_constructor=prepate_input_pnp,as_strings=False,print_per_layer_stat=False,verbose=True)
    
    ordered_dict["THRESHOLD"]=threshold
    ordered_dict["GMACS_BACKBONE"]=macs_backbone/1e09
    ordered_dict["PARAMETERS_BACKBONE"]=params_backbone/1e06
    ordered_dict["GMACS_GEO_HEAD"]=macs_geo/1e09
    ordered_dict["PARAMETERS_GEO_HEAD"]=params_geo/1e06
    ordered_dict["GMACS_PNP_HEAD"]=macs_pnp/1e09
    ordered_dict["PARAMETERS_PNP_HEAD"]=params_pnp/1e06
    if save:
        export_gflops(ordered_dict)
    else:

        return ordered_dict    

def export_gflops(model_params):

    result=[model_params]
    output_file = '/gdrnpp_bop2022/core/gdrn_modeling/Fast/inspect.csv'
    #import ipdb; ipdb.set_trace()
        # Specify the headers
    headers=list(model_params.keys())
    
    file_exists = os.path.exists(output_file)
        # Append data to CSV file
    with open(output_file, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)

            # Check if the file is empty
        
        if not file_exists:
            writer.writeheader()
            # Write data rows
        for row in result:
            writer.writerow(row)

    print(f'Data has been appended to {output_file}')
                



def prepate_input_pnp(resolution):
    
    coor_feat=torch.cuda.FloatTensor(8,5,64,64)
    region=torch.cuda.FloatTensor(8,64,64,64)
    extents=torch.cuda.FloatTensor(8,3)
    
    return dict(coor_feat=coor_feat,region=region,extents=extents,mask_attention=None)
def prepate_input_ghead(resolution):
    input=torch.cuda.FloatTensor(8,1024,8,8)
    return dict(x=input)    
def prepate_input_backbone(resolution):
    input=torch.cuda.FloatTensor(8,3,256,256)
    return dict(x=input)











