import os
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch.utils.data import Dataset, DataLoader

import sys


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=400, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--num_workers", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--n_critic", type=int, default=10, help="number of training steps for discriminator per iter")
parser.add_argument("--feature_dim", type=int, default=91, help="dimensionality of the feature space")
parser.add_argument("--n_timesteps", type=int, default=960, help="sequence length")
parser.add_argument("--sample_interval", type=int, default=100, help="interval betwen image samples")
parser.add_argument("--gpu", type=str, default='0,1', help="interval betwen image samples")

args = parser.parse_args()
print(args)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from sklearn import preprocessing

class DatasetOffline(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample

    
class TimeSeriesLoader:
    def __init__(self, batch_size, num_workers):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.set_dataset()
        
    def min_max_scale(self, data):
        min_val = np.min(np.min(data, axis = 0), axis = 0)
        data = data - min_val
        max_val = np.max(np.max(data, axis = 0), axis = 0)
        norm_data = data / (max_val + 1e-7)
        self.min_val = min_val
        self.max_val = max_val
        return norm_data
    
    def undo_min_max_scale(self, data):
        return data * self.max_val + self.min_val
        
    def set_dataset(self):
        loaded_npz = np.load("../data/real_train.npz") #(num_samples, seq_len, num_dim)
        data_feature = self.min_max_scale(loaded_npz["trans"])
        self.dataset = DatasetOffline(data_feature)
    
    def load(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers, 
                          shuffle=True, pin_memory=True)
    
ts_loader = TimeSeriesLoader(batch_size=args.batch_size, num_workers=8)
loader = ts_loader.load()


class GeneratorMLP(nn.Module):
    def __init__(self, output_dim):
        super(GeneratorMLP, self).__init__()
        self.hidden_dim = 256
        self.model = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, args.n_timesteps * args.feature_dim),
        )
        
    def forward(self, z):
        batch_size = z.shape[0]
        cur_device = z.get_device()
        output = self.model(z)
        return output.view((batch_size, args.n_timesteps, args.feature_dim))
    
class DiscriminatorMLP(nn.Module):
    def __init__(self, input_dim):
        super(DiscriminatorMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(args.n_timesteps * input_dim, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        return self.model(x.view(batch_size,-1))
    
generator = GeneratorMLP(args.feature_dim).to(device)
discriminator = DiscriminatorMLP(args.feature_dim).to(device)
generator = nn.DataParallel(generator)
discriminator = nn.DataParallel(discriminator)

optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

lambda_gp = 10

def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    batch_size = real_samples.size(0)

    # Get random interpolation between real and fake samples
    alpha = torch.Tensor(np.random.random((batch_size, 1, 1))).to(device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    grad_outputs = torch.ones(batch_size,1).to(device)
    
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        inputs=interpolates,
        outputs=d_interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

with torch.backends.cudnn.flags(enabled=False):
    generator.train()
    discriminator.train()
    for epoch in tqdm(range(args.n_epochs)):
        for i, x in enumerate(loader):
            x = x.to(device)
            batch_size = x.shape[0]

            # =====  Train Discriminator =====
            optimizer_D.zero_grad()

            z = torch.Tensor(np.random.normal(0, 1, (batch_size, 256))).to(device)
            gen_x = generator(z)

            f_real = discriminator(x)
            f_fake = discriminator(gen_x)
            gradient_penalty = compute_gradient_penalty(discriminator, x.data, gen_x.data)

            loss_D = -torch.mean(f_real) + torch.mean(f_fake) + lambda_gp * gradient_penalty

            loss_D.backward()
            optimizer_D.step()

            # ===== Train Generator =====
            if i % args.n_critic == 0:
                optimizer_G.zero_grad()

                gen_x = generator(z)
                loss_G = -torch.mean(discriminator(gen_x))

                loss_G.backward()
                optimizer_G.step()

        if epoch % args.sample_interval == 0 or epoch == args.n_epochs - 1:
            generator.eval()
            print("[Epoch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, args.n_epochs, loss_D.item(), loss_G.item()))

            num_samples = 100
            z = torch.Tensor(np.random.normal(0, 1, (num_samples, 256))).to(device)
            gen_x = generator(z).detach().cpu().numpy()
            print(gen_x.shape)
            
generator.eval()
discriminator.eval()
generated_samples = np.zeros((400, args.n_timesteps, args.feature_dim))
for i in range(4):
    num_samples = 100
    z = torch.Tensor(np.random.normal(0, 1, (num_samples, 256))).to(device)
    gen_x = generator(z).detach().cpu().numpy()
    generated_samples[i:i+100] = gen_x
    
np.savez("generated_data.npz", data_feature=ts_loader.undo_min_max_scale(generated_samples))