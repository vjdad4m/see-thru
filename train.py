import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SeeThruDataset(Dataset):
    def __init__(self):
        data = np.load('processed/dataset.npz')
        self.X = torch.tensor(data['arr_0']).to(device)
        self.Y = torch.tensor(data['arr_1']).to(device)
        print('loaded dataset with shapes', self.X.shape, self.Y.shape)
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx])

class SeeThruNet(nn.Module):
    def __init__(self):
        super(SeeThruNet, self).__init__()
        self.a1 = nn.Conv2d(1, 16, kernel_size=3, padding=0)
        self.a2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.a3 = nn.Conv2d(16, 32, kernel_size=3, stride=2)

        self.b1 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.b2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.b3 = nn.Conv2d(32, 64, kernel_size=3, stride=2)

        self.c1 = nn.Conv2d(64, 64, kernel_size=2, padding=1)
        self.c2 = nn.Conv2d(64, 64, kernel_size=2, padding=1)
        self.c3 = nn.Conv2d(64, 128, kernel_size=2, stride=2)

        self.d1 = nn.Conv2d(128, 128, kernel_size=1)
        self.d2 = nn.Conv2d(128, 128, kernel_size=1)
        self.d3 = nn.Conv2d(128, 128, kernel_size=1)

        self.e1 = nn.Linear(256, 64)
        self.e2 = nn.Linear(64, 26)
    
    def forward(self, x):
        x = F.relu(self.a1(x))
        x = F.relu(self.a2(x))
        x = F.relu(self.a3(x))
        x = F.max_pool2d(x, (2, 2))
        
        x = F.relu(self.b1(x))
        x = F.relu(self.b2(x))
        x = F.relu(self.b3(x))
        x = F.max_pool2d(x, (2, 2))

        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = F.relu(self.c3(x))

        x = F.relu(self.d1(x))
        x = F.relu(self.d2(x))
        x = F.relu(self.d3(x))

        x = x.view(-1, 256)
        x = self.e1(x)
        x = self.e2(x)

        return x

def SeeThruLoss(out, real):
    assert len(out) == len(real)
    loss = 0
    for i in range(len(out)//2):
        x1, y1 = out[i*2], out[i*2+1]
        x2, y2 = real[i*2], real[i*2+1]
        x1 = x1 * 640
        x2 = x2 * 640
        y1 = y1 * 480
        y2 = y2 * 480
        dist = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
        loss += dist
    return loss

if __name__ == '__main__':
    dataset = SeeThruDataset()
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    model = SeeThruNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    floss = SeeThruLoss

    model.train()
    min_loss = 9999999
    
    writer = SummaryWriter()

    for epoch in tqdm.trange(1000):
        all_loss = 0
        num_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            target = target.unsqueeze(-1)
            target = target.flatten(-3)
            
            data = data.unsqueeze(0)
            data = data.float()
            
            optimizer.zero_grad()
            output = model(data)
            loss = floss(output[0], target[0])
            loss.backward()
            optimizer.step()

            all_loss += loss.item()
            num_loss += 1
        
        writer.add_scalar("Loss/train", all_loss / num_loss, epoch)

        print('%3d: %f' % (epoch, all_loss / num_loss))
        if all_loss / num_loss < min_loss:
            min_loss = all_loss / num_loss
            torch.save(model.state_dict(), f'nets/seethru/seethru_{min_loss}.pth')