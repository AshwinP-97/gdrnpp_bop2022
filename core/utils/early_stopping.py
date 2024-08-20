import numpy as np
import csv

class EarlyStopping_smooth:
    def __init__(self,patience=3,min_delta=1e-4,windows=5):
        self.patience=patience
        self.min_delta=min_delta
        self.windows=windows
        self.losses=[]
        self.counter=0
        self.best_loss=np.inf



    def __call__(self,current_loss):
        
        self.losses.append(current_loss)

        if len(self.losses)> self.windows:
            smoothed_loss=np.mean(self.losses[-self.windows:])
        else:
            smoothed_loss=current_loss
        
        if self.best_loss-smoothed_loss>self.min_delta:
            self.best_loss=smoothed_loss
            self.counter=0
        else:
            self.counter+=1
        
        if self.counter>=self.patience:
            return True
        
        return False
    

def main():
    with open("prog_error.csv","r") as f:
        reader=csv.reader(f)
        data=list(reader)
    losses=np.array(data,dtype=float)
    stop=EarlyStopping_smooth(patience=4,min_delta=1e-4,windows=5)
    for i in range (0,1000,50):
        import ipdb;ipdb.set_trace()
        print(stop(np.mean(losses[i:i+50])))
        print(i)
        



if __name__=="__main__":
    main()