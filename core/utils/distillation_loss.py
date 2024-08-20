import torch
import torch.nn as nn
from torch.nn import init
#from geomloss import SamplesLoss

class Normalized_distill_Loss(nn.Module):
    def __init__(self,loss_type,student_features,teacher_features,threshold=None):
        super(Normalized_distill_Loss, self).__init__()
        self.loss_type=loss_type
        if "MSE" in self.loss_type:
            self.loss = nn.MSELoss()
        """
        if "Sample" in self.loss_type:
            self.loss=  SamplesLoss("sinkhorn",blur=threshold)
        """
        if "KL" in self.loss_type:
            self.temperature=threshold
            self.loss=nn.KLDivLoss(reduction="batchmean")
        if student_features != teacher_features:
            self.conv = nn.Conv2d(student_features, teacher_features, kernel_size=1)
                    # Initialize the weights
            init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
            if self.conv.bias is not None:
                init.constant_(self.conv.bias, 0)
                
    def normalize(self, features):
        mean = features.mean(dim=[2, 3], keepdim=True)
        std = features.std(dim=[2, 3], keepdim=True)
        return (features - mean) / (std + 1e-5)  # Add a small constant to avoid division by zero
                
    def forward(self, student_features, teacher_features):
    	# Normalize feature maps
        

                # Align the number of channels using a 1x1 convolution if necessary
        if student_features.shape[1] != teacher_features.shape[1]:
            self.conv = self.conv.to(student_features.device)
            student_features = self.conv(student_features)

        student_features = self.normalize(student_features)
        teacher_features = self.normalize(teacher_features)
        
        if "MSE" in self.loss_type:
            
            return self.loss(student_features,teacher_features)
        if "Sample" in self.loss_type:
            
            batch_size,channels,height,width=teacher_features.shape
            student_features_flat=student_features.permute(0,2,3,1).reshape(batch_size,-1,channels)
            teacher_features_flat=teacher_features.permute(0,2,3,1).reshape(batch_size,-1,channels)
            loss=0
            for i in range(batch_size):
                loss+=self.loss(student_features_flat[i],teacher_features_flat[i])
            
            return loss/batch_size
        if "KL" in self.loss_type:
            
            batch_size,channels,height,width=teacher_features.shape
            student_features=student_features.view(batch_size,channels,-1)
            teacher_features=teacher_features.view(batch_size,channels,-1)

            student_softmax=torch.log_softmax(student_features/self.temperature,dim=-1)
            teacher_softmax=torch.softmax(teacher_features/self.temperature,dim=-1)
            return self.loss(student_softmax,teacher_softmax)*(self.temperature**2)

def custom_loss(student_features,teacher_features,student_mask=None,teacher_mask=None,epsilon=0.0001):
    
    ot_loss=SamplesLoss("sinkhorn",blur=epsilon)
    N_s,C_s,H_s,W_s=student_features.shape
    N_t,C_t,H_t,W_t=teacher_features.shape
    #student_features_flat=student_features.permute(0,2,3,1).reshape(N_s,-1,C_s)
    #teacher_features_flat=teacher_features.permute(0,2,3,1).reshape(N_t,-1,C_t)

    
    student_mean=student_features.mean(dim=[2,3],keepdim=True)
    teacher_mean=teacher_features.mean(dim=[2,3],keepdim=True)
    student_std=student_features.std(dim=[2,3],keepdim=True)
    teacher_std=teacher_features.std(dim=[2,3],keepdim=True)
        
    student_features=(student_features-student_mean)/(student_std+1e-5)
    teacher_features=(teacher_features-teacher_mean)/(teacher_std+1e-5)
    
    student_features_flat=student_features.reshape(N_s,C_s,-1)
    teacher_features_flat=teacher_features.reshape(N_t,C_t,-1)

    loss=0
    
    if student_mask is None and teacher_mask is None:
        for i in range(student_features.size(0)):
            loss+= ot_loss(student_features_flat[i],teacher_features_flat[i])
    else:
        for i in range(student_features.size(0)):
            loss+= ot_loss(student_features_flat[i],teacher_features_flat[i],student_mask[i],teacher_mask[i])

    return loss/student_features.size(0)


if __name__ == "__main__":
  teacher= torch.randn(1,2,2,2)
  student= torch.rand(1,1,2,2)
  
  loss1=Normalized_distill_Loss("KL",1,2,0.001)
  loss1(student,teacher)