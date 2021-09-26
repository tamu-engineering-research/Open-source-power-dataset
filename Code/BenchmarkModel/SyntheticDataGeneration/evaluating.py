import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class SeqDataset(Dataset):
    def __init__(self, sequences, labels, masks):
        self.sequences = sequences
        self.labels = labels
        self.masks = masks
        assert len(self.sequences) == len(self.labels)
        assert len(self.sequences) == len(self.masks)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx], self.masks[idx]

class GRUClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, num_layers=1, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)        
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, mask):
        packed_x = pack_padded_sequence(x, mask.sum(dim=1), batch_first=True, enforce_sorted=False)
        outputs, hidden = self.gru(packed_x)
        return self.linear(hidden[-1])
    
def train(model, device, train_loader, criterion, optimizer):
    losses, acc = AverageMeter(), AverageMeter()
    model.train()  # Train mode

    # Mini-batch training
    for inputs, labels, mask in train_loader:
        inputs = inputs.float().to(device)
        labels = labels.long().to(device)
        outputs = model(inputs, mask)
        loss = criterion(outputs, labels)

        # Measure accuracy and record loss
        batch_size = labels.shape[0]
        with torch.no_grad():
            y_pred = torch.argmax(outputs.data, dim=1).tolist()
            y_true = labels.tolist()
            accuracy = accuracy_score(y_true, y_pred)
        losses.update(loss.item(), batch_size)
        acc.update(accuracy, batch_size)

        # Compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return acc.avg

def validate(model, device, test_loader, criterion):
    losses = AverageMeter()
    model.eval()  # Evaluate mode

    y_pred = []
    y_true = []
    y_prob = []
    with torch.no_grad():
        for inputs, labels, mask in test_loader:
            inputs = inputs.float().to(device)
            labels = labels.long().to(device)

            # Compute output
            outputs = model(inputs, mask)
            loss = criterion(outputs, labels)
            y_prob += outputs.data[:,1].tolist()
            y_pred += torch.argmax(outputs.data, dim=1).tolist()
            y_true += labels.tolist()

            # Accumulate video level prediction
            batch_size = labels.shape[0]
            losses.update(loss.item(), batch_size)
            
    return accuracy_score(y_true, y_pred)


def train_test_divide (data_x, data_x_hat, train_rate = 0.8):
    """Divide train and test data for both original and synthetic data.
    Args:
    - data_x: original data
    - data_x_hat: generated data
    - data_t: original time
    - data_t_hat: generated time
    - train_rate: ratio of training data from the original data
    """
    # Divide train/test index (original data)
    no = len(data_x)
    idx = np.random.permutation(no)
    train_idx = idx[:int(no*train_rate)]
    test_idx = idx[int(no*train_rate):]

    train_x = [data_x[i] for i in train_idx]
    test_x = [data_x[i] for i in test_idx]     

    # Divide train/test index (synthetic data)
    no = len(data_x_hat)
    idx = np.random.permutation(no)
    train_idx = idx[:int(no*train_rate)]
    test_idx = idx[int(no*train_rate):]

    train_x_hat = [data_x_hat[i] for i in train_idx]
    test_x_hat = [data_x_hat[i] for i in test_idx]

    return train_x, train_x_hat, test_x, test_x_hat

def discriminative_score(ori_data, generated_data, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    no, seq_len, dim = np.asarray(ori_data).shape    
    hidden_dim = int(dim/2)
    iterations = 100
    batch_size = 128
            
    train_x, train_x_hat, test_x, test_x_hat = train_test_divide(ori_data, generated_data)
    
    train_features = train_x + train_x_hat
    train_labels = [1]*len(train_x) + [0]*len(train_x_hat)
    train_masks = [np.ones((seq_len)) for _ in range(len(train_features))]
    
    test_features = test_x + test_x_hat
    test_labels = [1]*len(test_x) + [0]*len(test_x_hat)
    test_masks = [np.ones((seq_len)) for _ in range(len(test_features))]
    
    train_loader = DataLoader(
        SeqDataset(train_features, train_labels, train_masks),
        batch_size=batch_size,
        num_workers=0,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        SeqDataset(test_features, test_labels, test_masks),
        batch_size=batch_size,
        num_workers=0,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
    )

    model = GRUClassifier(91,2, hidden_dim=hidden_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([{"params": model.parameters(), "lr": 1e-3},])

    for epoch in tqdm(range(iterations)):
        train(model, device, train_loader, criterion, optimizer)

    acc = validate(model, device, test_loader, criterion)
    discriminative_score = np.abs(0.5-acc)
    return discriminative_score

def run_evaluate_generation(root, input_dict):
    ori_data = np.load(root + 'generated_samples.npz')
    generated_data = np.load(root + 'real_samples.npz')
    print(ori_data.shape) #(num_samples, seq_len, dim)
    print(generated_data.shape) #(num_samples, seq_len, dim)
    
    metric_iteration = 5
    discriminative_scores = list()
    for _ in range(metric_iteration):
        print("TRAINING {}".format(_))
        temp_disc = discriminative_score(ori_data, generated_data, _)
        print(temp_disc)
        discriminative_scores.append(temp_disc)

    print('Discriminative score: ' + str(np.round(np.mean(discriminative_scores), 4)))
    print('Discriminative score (std): ' + str(np.round(np.std(discriminative_scores), 4)))
    return discriminative_scores
