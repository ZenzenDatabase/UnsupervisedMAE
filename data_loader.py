import pandas as pd
import numpy as np
import os
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import re
import h5py
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
class BaseDataset(Dataset):
    def __init__(self, viewA: np.ndarray, viewB: np.ndarray, Lab: np.ndarray):
        if viewA.dtype != np.float32:
            viewA = viewA.astype(np.float32)
        if viewB.dtype != np.float32:
            viewB = viewB.astype(np.float32)
        if Lab.dtype   != np.float32: 
            Lab   = Lab.astype(np.float32)

        self.viewA = viewA
        self.viewB = viewB
        self.Lab   = Lab
        assert len(self.viewA) == len(self.viewB) == len(self.Lab), \
            "All input arrays (viewA, viewB, Lab) must have the same number of samples."

    def __getitem__(self, index: int):
        viewA_sample = self.viewA[index]
        viewB_sample = self.viewB[index]
        Lab_sample   = self.Lab[index] 

        tensor_viewA = torch.tensor(viewA_sample, dtype=torch.float32)
        tensor_viewB = torch.tensor(viewB_sample, dtype=torch.float32)
        tensor_Lab   = torch.tensor(Lab_sample, dtype=torch.float32)
        return tensor_viewA, tensor_viewB, tensor_Lab

    def __len__(self) -> int:
        return len(self.viewA)

def normalize_v2(data: np.ndarray) -> np.ndarray:
    if data.ndim == 1: 
        data = data.reshape(-1, 1)
    scaler     = MinMaxScaler()
    normalized = scaler.fit_transform(data)
    return normalized.astype(np.float32) 

def ind2vec(ind: np.ndarray, N: int = None) -> np.ndarray:
    ind = np.asarray(ind).flatten()
    if N is None:
        N = ind.max() + 1
    one_hot = np.zeros((len(ind), N), dtype=np.float32)
    one_hot[np.arange(len(ind)), ind] = 1.0
    return one_hot

def _load_data(_path: str, batch_size: int = 128, audio_feat: str = "audio_feat",
              visual_feat: str = "visual_feat"):
    if not os.path.exists(_path):
        raise FileNotFoundError(f"the file not found at: {_path}")

    with h5py.File(_path, "r") as f:
        audio_feat  = normalize_v2(f["audio_feat"][:].astype(np.float32))
        visual_feat = normalize_v2(f["visual_feat"][:].astype(np.float32))
        label_ids   = f["Label_ID"][:]
        
        try:
            split = f["Train_test"][:].astype(str)
        except:
            split = f["Split"][:].astype(str)

    train_mask = (split == "1")
    test_mask  = (split == "0")

    audio_train  = audio_feat[train_mask]
    visual_train = visual_feat[train_mask]
    audio_test   = audio_feat[test_mask]
    visual_test  = visual_feat[test_mask]
    label_train  = label_ids[train_mask]
    label_test   = label_ids[test_mask]
    
    all_labels    = np.concatenate([label_train, label_test])
    unique_labels = sorted(list(set(all_labels)))
    num_classes   = len(unique_labels)
    label_train = ind2vec(label_train, N=num_classes)
    label_test  = ind2vec(label_test, N=num_classes)

    print(f'Audio Train Shape: {audio_train.shape}, Dtype: {audio_train.dtype}')
    print(f'Audio Test Shape: {audio_test.shape}, Dtype: {audio_test.dtype}')
    print(f'Visual Train Shape: {visual_train.shape}, Dtype: {visual_train.dtype}')
    print(f'Visual Test Shape: {visual_test.shape}, Dtype: {visual_test.dtype}')
    print(f'Label Train Shape (one-hot): {label_train.shape}, Dtype: {label_train.dtype}')
    print(f'Label Test Shape (one-hot): {label_test.shape}, Dtype: {label_test.dtype}\n')

    dataset = {
        "train": BaseDataset(audio_train, visual_train, label_train),
        "test": BaseDataset(audio_test, visual_test, label_test)
    }

    num_workers = os.cpu_count() // 2 if os.cpu_count() else 0 
    dataloader = {
        split: DataLoader(dataset[split], batch_size=batch_size, shuffle=(split == "train"), num_workers=num_workers)
        for split in ["train", "test"]
    }

    input_database = {
        'audio_test': audio_test,
        'visual_test': visual_test,
        'label_test': label_test,
        'audio_dim': audio_train.shape[1],
        'visual_dim': visual_train.shape[1],
        'num_classes': num_classes
    }
    return dataloader, input_database

def load_data(_path: str, batch_size: int = 128):
    Reader = h5py.File(_path, 'r')
    print(Reader.keys())
    audio_train =  np.asarray(Reader["audio_train"][()], dtype=np.float32)
    audio_test  =  np.asarray(Reader['audio_test'][()], dtype=np.float32)

    visual_train  =  np.asarray(Reader['visual_train'][()], dtype=np.float32)
    visual_test   =  np.asarray(Reader['visual_test'][()], dtype=np.float32)

    label_train =   np.asarray(Reader['lab_train'][()]).reshape(len(audio_train), 1)
    label_test  =   np.asarray(Reader['lab_test'][()]).reshape(len(audio_test), 1)

    print('audio_train shape:{},\
    \n audio_test shape:{},\
    \n visual_train shape:{}, \
    \n visual_test shape:{}, \
    \n label_train shape:{}, \
    \n label_test shape:{}.\n\
        '.format(audio_train.shape, \
                 audio_test.shape, \
                 visual_train.shape, \
                 visual_test.shape, \
                 label_train.shape, \
                 label_test.shape))

    audio_train  = normalize_v2(audio_train) 
    audio_test   = normalize_v2(audio_test) 
    visual_train = normalize_v2(visual_train) 
    visual_test  = normalize_v2(visual_test) 

    label_train  = ind2vec(label_train).astype(int)
    label_test   = ind2vec(label_test).astype(int)

    audios  = {'train': audio_train, 'test': audio_test}
    visuals = {'train': visual_train, 'test': visual_test}
    labels  = {'train': label_train, 'test': label_test}
    dataset = {x: BaseDataset(viewA=audios[x], viewB=visuals[x], Lab=labels[x]) for x in ['train', 'test']}

    shuffle = {'train': False, 'test': False}

    dataloader = {x: DataLoader(dataset[x], batch_size=batch_size,
                                shuffle=shuffle[x], num_workers=0) for x in ['train', 'test']}
    audio_dim  = audio_train.shape[1]
    visual_dim = visual_train.shape[1]

    input_database = {}
    input_database['audio_test']   = audio_test
    input_database['visual_test']  = visual_test
    input_database['label_test']   = label_test
    input_database['audio_train']  = audio_train
    input_database['visual_train'] = visual_train
    input_database['label_train']  = label_train
    input_database['audio_dim']    = audio_dim
    input_database['visual_dim']   = visual_dim

    return dataloader, input_database