import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import tqdm
import random

from train import SeeThruLoss

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class PoseDataset(Dataset):
    def __init__(self):
        data = np.load('processed/dataset.npz')['arr_1']
        poses = []
        for p in data:
            # check if all keypoints exist
            if not [0, 0] in p:
                poses.append(p)
        self.poses = np.array(poses)

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, index):
        pose = self.poses[index]
        new_pose = pose.copy()
        # purge some keypoints
        k = random.randrange(1, 5)
        for i in range(k):
            new_pose[random.randrange(0, 13)] = [0, 0]
        return (torch.tensor(new_pose).to(device), torch.tensor(pose).to(device))

class PoseSupplementaryModel(nn.Module):
    def __init__(self):
        super(PoseSupplementaryModel, self).__init__()
        self.a1 = nn.Linear(26, 26 * 4)
        self.a2 = nn.Linear(26 * 4, 26 * 4 * 2)
        self.a3 = nn.Linear(26 * 4 * 2, 26)
        self.a4 = nn.Linear(26 * 2, 26)
    
    def forward(self, x):
        x_orig = x
        x = F.leaky_relu(self.a1(x), 0.2)
        x = F.leaky_relu(self.a2(x), 0.2)
        x = F.leaky_relu(self.a3(x), 0.2)
        x = torch.cat([x, x_orig], -1)
        x = self.a4(x)
        return x

def train():
    dataset = PoseDataset()
    print('dataset size is %d' % len(dataset))
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    model = PoseSupplementaryModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    floss = SeeThruLoss

    model.train()
    min_loss = 9999999

    writer = SummaryWriter()

    for epoch in tqdm.trange(10000):
        all_loss = 0
        num_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            target = target.unsqueeze(0).flatten()
            target = target.float()
            
            data = data.unsqueeze(0).flatten()
            data = data.float()
            
            optimizer.zero_grad()

            output = model(data)
            loss = floss(output, target)

            loss.backward()
            optimizer.step()

            all_loss += loss.item()
            num_loss += 1
        
        writer.add_scalar("Loss/train", all_loss / num_loss, epoch)

        if epoch % 25 == 0:
            print('%3d: %f' % (epoch, all_loss / num_loss))
        if all_loss / num_loss < min_loss:
            min_loss = all_loss / num_loss
            torch.save(model.state_dict(), f'nets/supplement/supplement_{min_loss}.pth')

class PoseSupplementer:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = PoseSupplementaryModel()
        vals = torch.load('nets/supplement.pth')
        self.model.load_state_dict(vals)
        self.model.to(self.device)

    def predict(self, pose_missing):
        pose_missing = torch.tensor(pose_missing).to(self.device).unsqueeze(0).flatten().float()
        with torch.no_grad():
            out = self.model(pose_missing)
        return out.cpu().numpy().reshape((13, 2)).astype(np.float32)
    
    def __call__(self, pose_missing):
        return self.predict(pose_missing)

if __name__ == '__main__':
    train()