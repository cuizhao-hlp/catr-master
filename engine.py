# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from torch import nn
import math
import sys
import tqdm

import random

from torch.nn import functional

from models import utils


#  HOW TO USE
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super(ContrastiveLoss, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.register_buffer("temperature", torch.tensor(temperature))
        self.verbose = False

    def process_do(self, z_i, z_j):
        # --------------------------------------------
        # Define forward pass
        # --------------------------------------------
        batch_size = z_i.shape[0]
        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = functional.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0),
                                                         dim=2)
        if self.verbose:
            print("Similarity matrix\n", similarity_matrix, "\n")

        def l_ij(i, j):
            sim_i_j = similarity_matrix[i, j]
            if self.verbose:
                print(f"sim({i}, {j})={sim_i_j}")
            numerator = torch.exp(sim_i_j / self.temperature)
            one_for_not_i = torch.ones((2 * batch_size,)).scatter_(0, torch.tensor([i]), 0.0).to(z_i.device)
            if self.verbose:
                print(f"1{{k!={i}}}", one_for_not_i)
            denominator = torch.sum(
                one_for_not_i * torch.exp(similarity_matrix[i, :] / self.temperature)
            )
            if self.verbose:
                print("Denominator", denominator)
            loss_ij = -torch.log(numerator / denominator)
            if self.verbose:
                print(f"loss({i},{j})={loss_ij}\n")
            return loss_ij.squeeze(0)

        N = batch_size
        loss = 0.0
        for k in range(0, N):
            loss += l_ij(k, k + N) + l_ij(k + N, k)
        return 1.0 / (2 * N) * loss


def train_one_epoch(model, criterion, data_loader,
                    optimizer, device, epoch, max_norm):
    model.train()
    criterion.train()

    epoch_loss = 0.0
    total = len(data_loader)

    contrast_loss = 0.0

    def da(samples_c, i):
        mask = random.randint(0, 0.25 * len(samples_c[i]))
        new_samples_c = samples_c
        new_samples_c[0:mask] = random.randint(0, 255)
        return new_samples_c

    with tqdm.tqdm(total=total) as pbar:
        for images, masks, caps, cap_masks in data_loader:
            samples = utils.NestedTensor(images, masks).to(device)

            for i in range(samples.size(0)):
                new_samples = da(samples,i)
                contrast_loss += ContrastiveLoss.process_do(samples, new_samples)

            caps = caps.to(device)
            cap_masks = cap_masks.to(device)

            outputs = model(samples, caps[:, :-1], cap_masks[:, :-1])
            loss = criterion(outputs.permute(0, 2, 1), caps[:, 1:])  # , dtype=loss.long)  # modified
            loss_value = loss.item()
            epoch_loss += loss_value + contrast_loss

            if not math.isfinite(loss_value):
                print(f'Loss is {loss_value}, stopping training')
                sys.exit(1)

            optimizer.zero_grad()
            loss.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

            pbar.update(1)

    return epoch_loss / total


@torch.no_grad()
def evaluate(model, criterion, data_loader, device):
    model.eval()
    criterion.eval()

    validation_loss = 0.0
    total = len(data_loader)

    with tqdm.tqdm(total=total) as pbar:
        for images, masks, caps, cap_masks in data_loader:
            samples = utils.NestedTensor(images, masks).to(device)
            caps = caps.to(device)
            cap_masks = cap_masks.to(device)

            outputs = model(samples, caps[:, :-1], cap_masks[:, :-1])
            loss = criterion(outputs.permute(0, 2, 1), caps[:, 1:])

            validation_loss += loss.item()

            pbar.update(1)

    return validation_loss / total
