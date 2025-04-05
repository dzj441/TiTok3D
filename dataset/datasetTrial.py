from data import DecordVideoDataset
import torch
import decord
import sys
import os
import time
sys.path.append(os.getcwd())
load_path = "datasets/UCF-101/test"
trainlist = "datasets/UCF-101/trainlist.txt"
testlist = "datasets/UCF-101/testlist.txt"

if __name__ == '__main__':
    time1 = time.time()
    dataset = DecordVideoDataset(load_path,testlist,fps=16,train=False,sequence_length=16,resolution=128)
    time2 = time.time()
    print(f"building dataset takes {time2 - time1} seconds")
    dataset.__getitem__(0)
    time3 = time.time()
    print(f"loading a video takes {time3 - time2} seconds")
    trainloader = torch.utils.data.DataLoader(dataset=dataset,batch_size=1,shuffle=True,pin_memory=False,num_workers=2)
    time4 = time.time()
    print(f"building dataloader takes {time4-time3} seconds")
    print(len(trainloader))
    print(next(iter(trainloader)))
    print(next(iter(trainloader)).get('video'))
    
    

