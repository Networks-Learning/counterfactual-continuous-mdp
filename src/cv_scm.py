import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from src.models import SCM
import numpy as np
from sklearn.model_selection import KFold
import json
import logging
import click
import random

class CustomBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

        # Group indices based on the first column value
        self.groups = {}
        for idx, (X, _) in enumerate(self.dataset):
            key = X[0].item()
            if key not in self.groups:
                self.groups[key] = []
            self.groups[key].append(idx)

    def __iter__(self):
        keys = list(self.groups.keys())
        random.shuffle(keys)
        shuffled_dict = {key: self.groups[key] for key in keys}
        for group in shuffled_dict.values():
            random.shuffle(group)
            for i in range(0, len(group), self.batch_size):
                yield group[i:i + self.batch_size]

    def __len__(self):
        return sum(len(indices) // self.batch_size for indices in self.groups.values())

class RegressionDataset(Dataset):
    def __init__(self, input_df, output_df):
        self.input_data = torch.tensor(input_df.values, dtype=torch.float32)
        self.output_data = torch.tensor(output_df.values, dtype=torch.float32)

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        return self.input_data[idx], self.output_data[idx]

def train_loop(dataloader, model, optimizer, device, a_dim):
    size = len(dataloader.dataset)
    model.train()

    for batch, (X, y) in enumerate(dataloader):

        a = X[:, :a_dim].to(device)
        s = X[:, a_dim:].to(device)
        s_prime = y.to(device)        

        # Compute loss
        loss = - model.log_likelihood(s, a, s_prime).sum()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            logging.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def val_loop(dataloader, model, device, a_dim):
    
    model.eval()
    val_losses = []
    with torch.no_grad():
        for X, y in dataloader:

            a = X[:, :a_dim].to(device)
            s = X[:, a_dim:].to(device)
            s_prime = y.to(device)        

            val_losses.append(- model.log_likelihood(s, a, s_prime).sum().item())

    test_loss = np.mean(val_losses)
    
    # print(f"Avg loss: {test_loss:>8f}\n")
    logging.info(f"Avg loss: {test_loss:>8f}\n")
    return test_loss

def train_final_model(prediction_task, hidden_layers, hidden_units, lipschitz_loc, lipschitz_scale, learning_rate,\
                    batch_size, max_epochs, df_a_s, df_s_prime, a_dim, c_dim, prior_type, cuda, model_directory):

    num_of_features = len(df_a_s.columns) - a_dim
    dataset = RegressionDataset(df_a_s, df_s_prime)
    n_power_iterations = 10

    logging.info(f"Initiating training...\n")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    device = "cuda:{cuda}".format(cuda=str(cuda)) if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    logging.info(f"Using {device} device")
    model = SCM(num_of_features, hidden_layers, hidden_units, a_dim=a_dim, c_dim=c_dim, lipschitz_loc=lipschitz_loc, lipschitz_scale=lipschitz_scale, prior_type=prior_type, device=device, n_power_iterations=n_power_iterations).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for t in range(max_epochs):
        logging.info(f"Epoch {t+1}\n--------")
        train_loop(dataloader, model, optimizer, device, a_dim)

    logging.info('Training completed\n')

    if lipschitz_loc is None:
        lipschitz_loc = 'none'
    if lipschitz_scale is None:
        lipschitz_scale = 'none'
    
    path = ''.join([model_directory, f'{prediction_task}_hl_{hidden_layers}_hu_{hidden_units}_lr_{learning_rate}_bs_{batch_size}_lipschitzloc_{lipschitz_loc}_lipschitzscale_{lipschitz_scale}_prior_{prior_type}_maxepochs_{max_epochs}.pt'])
    torch.save(model.state_dict(), path)

def evaluate_configuration(prediction_task, temp_directory, hidden_layers, hidden_units, lipschitz_loc, lipschitz_scale, learning_rate,\
                    batch_size, max_epochs, df_a_s, df_s_prime, a_dim, c_dim, prior_type, cuda):
    
    logging.info(f"----------------NEW CONFIGURATION----------------")
    report = {
        "hidden_layers" : hidden_layers,
        "hidden_units" : hidden_units,
        "learning_rate" : learning_rate,
        "batch_size" : batch_size,
        "max_epochs" : max_epochs,
        "lipschitz_loc" : lipschitz_loc if lipschitz_loc is not None else 'none',
        "lipschitz_scale" : lipschitz_scale if lipschitz_scale is not None else 'none',
        "prediction_task" : prediction_task
    }

    num_of_features = len(df_a_s.columns) - a_dim
    dataset = RegressionDataset(df_a_s, df_s_prime)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    val_last_losses = []
    val_min_losses = []
    epochs_of_min_loss = []
    val_all_losses = []
    for train_index, val_index in kf.split(df_a_s):
        
        logging.info('Starting new fold')
        train_dataset = torch.utils.data.Subset(dataset, train_index)
        val_dataset = torch.utils.data.Subset(dataset, val_index)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        device = "cuda:{cuda}".format(cuda=str(cuda)) if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        logging.info(f"Using {device} device")
        model = SCM(num_of_features, hidden_layers, hidden_units, a_dim=a_dim, c_dim=c_dim, lipschitz_loc=lipschitz_loc, lipschitz_scale=lipschitz_scale, prior_type=prior_type, device=device).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        min_loss = torch.Tensor([float('Inf')]).item()
        val_losses = []
        for t in range(max_epochs):
            logging.info(f"Epoch {t+1}\n--------")
            train_loop(train_dataloader, model, optimizer, device, a_dim)
            loss = val_loop(val_dataloader, model, device, a_dim)
            val_losses.append(loss)
            if loss<min_loss:
                min_loss = loss
                t_min_loss = t+1

        val_all_losses.append(val_losses)
        val_min_losses.append(min_loss.item())
        val_last_losses.append(loss.item())
        epochs_of_min_loss.append(t_min_loss)
        logging.info('Fold completed\n')
    
    val_all_losses = np.array(val_all_losses)
    val_all_losses = np.mean(val_all_losses, axis=0).tolist()
    report["crossval_all_losses"] = val_all_losses
    report["crossval_last_loss"] = np.mean(val_last_losses)
    report["epoch_of_min_loss"] = np.mean(epochs_of_min_loss)
    report["crossval_min_loss"] = np.mean(val_min_losses)

    if lipschitz_loc is None:
        lipschitz_loc = 'none'
    if lipschitz_scale is None:
        lipschitz_scale = 'none'
    
    with open(''.join([temp_directory, f'{prediction_task}_cv_config_hl_{hidden_layers}_hu_{hidden_units}_lr_{learning_rate}_bs_{batch_size}_lipschitzloc_{lipschitz_loc}_lipschitzscale_{lipschitz_scale}_prior_{prior_type}.json']), 'w') as outfile:
        json.dump(report, outfile)
        outfile.write('\n')

@click.command()
@click.option('--prediction_task', type=str, required=True, help='name of the prediction task')
@click.option('--data_filename', type=str, required=True, help='location of processed data')
@click.option('--temp_directory', type=str, default='', help='directory of temporary outputs')
@click.option('--hidden_layers', type=int, required=True, help='number of hidden layers')
@click.option('--hidden_units', type=int, required=True, help='number of hidden units')
@click.option('--lipschitz_loc', type=float, default=None, help='target lipschitz constant for the location network')
@click.option('--lipschitz_scale', type=float, default=None, help='target lipschitz constant for the scaling network')
@click.option('--learning_rate', type=float, required=True, help='learning rate')
@click.option('--batch_size', type=int, required=True, help='optimization batch size')
@click.option('--max_epochs', type=int, required=True, help='maximum number of epochs')
@click.option('--prior_type', type=str, required=True, help='type of noise prior to use in the SCM')
@click.option('--cuda', type=int, default=1, help='cuda device to use if available')
@click.option('--final', is_flag=True, default=False, help='if set, the final model will be trained on the entire dataset')
@click.option('--model_directory', type=str, default='', help='directory of trained models')
def cv_predictor(prediction_task, data_filename, temp_directory, hidden_layers, hidden_units, lipschitz_loc, lipschitz_scale, learning_rate, batch_size, max_epochs, prior_type, cuda, final, model_directory):

    torch.manual_seed(42)
    logging.basicConfig(filename='log.log', level=logging.INFO)

    # read the data and split to train and test
    df_raw = pd.read_csv(data_filename)
    # find the column names that contain a: (actions) and c: (constant features)
    a_dim = len([col for col in df_raw.columns if col.startswith('a:')])
    c_dim = len([col for col in df_raw.columns if col.startswith('c:')])//2
    s_dim = (len(df_raw.columns) - a_dim)//2

    df_a_s = df_raw.iloc[:,:a_dim + s_dim]
    df_s_prime = df_raw.iloc[:,a_dim+s_dim:]
    
    if final:
        if model_directory == '':
            raise ValueError('model_directory must be specified if the --final flag is used')
        
        # train the final model on the entire dataset
        train_final_model(prediction_task=prediction_task, hidden_layers=hidden_layers, hidden_units=hidden_units,\
                lipschitz_loc=lipschitz_loc, lipschitz_scale=lipschitz_scale, learning_rate=learning_rate, batch_size=batch_size, max_epochs=max_epochs,\
                df_a_s=df_a_s, df_s_prime=df_s_prime, a_dim=a_dim, c_dim=c_dim, prior_type=prior_type, cuda=cuda, model_directory=model_directory)
    else:
        if temp_directory == '':
            raise ValueError('temp_directory must be specified if the --final flag is not used')
        # run cross validation
        evaluate_configuration(prediction_task=prediction_task, temp_directory=temp_directory, hidden_layers=hidden_layers, hidden_units=hidden_units,\
                    lipschitz_loc=lipschitz_loc, lipschitz_scale=lipschitz_scale, learning_rate=learning_rate, batch_size=batch_size, max_epochs=max_epochs,\
                    df_a_s=df_a_s, df_s_prime=df_s_prime, a_dim=a_dim, c_dim=c_dim, prior_type=prior_type, cuda=cuda)

if __name__ == '__main__':
    cv_predictor()
    # cv_predictor(prediction_task="mimic_transitions", data_filename="data/processed/dataset_transitions.csv", temp_directory='outputs/temp_outputs/', hidden_layers=1, hidden_units=100,\
    #             lipschitz_loc=1.1, lipschitz_scale=None, learning_rate=0.001, batch_size=256, max_epochs=20, prior_type='multigaussian', cuda=0, final=True, model_directory='outputs/models/')