import numpy as np
import pandas as pd
import pickle
import json
from sklearn.preprocessing import MinMaxScaler
from os.path import exists
import click
import logging

class Data():
    
    def __init__(self, data_filename, temp_directory, processed_data_directory, min_horizon):

        # id columns
        self.idcols = ['icustayid', 'bloc']
        # fixed demographic and contextual covariates
        self.demcols = ['gender', 're_admission', 'age'] # removed Weight_kg because it presents some very weird fluctuations (e.g., the patient's weight doubles in a few hours) -- also removed mechvent because it is binary
        # time-varying continuous covariates
        self.contcols = ['FiO2_1', 'paO2', 'Platelets_count', 'Total_bili', 'GCS', 'MeanBP', 'Creatinine', 'output_4hourly', 'SOFA']

        # input columns
        self.inputcols = ['max_dose_vaso', 'input_4hourly', 'mechvent']
        # outcome column
        self.outcomecol = ['died_within_48h_of_out_time']

        # features for binary normalization
        self.colbin = ['gender','re_admission']
        # features for z-score normalization
        self.colnorm = ['age', 'FiO2_1', 'paO2', 'Platelets_count', 'GCS', 'MeanBP', 'output_4hourly', 'SOFA']
        # features for log normalization
        self.collog = ['Creatinine', 'Total_bili']
        # keep normalization attributes for inverse scaling
        self.norm_attributes = {}

        self.data_filename = data_filename
        self.processed_data_directory = processed_data_directory
        self.temp_directory = temp_directory
        self.min_horizon = min_horizon

    def generate_data(self):
        
        self.raw_df = self._read_csv_file()
        self.normalized_df = self._scale_and_drop()

        self.trajectories = self._extract_trajectories()
        for patient_id in self.trajectories:
            temp_trajectories_filename = ''.join([self.temp_directory, 'trajectory_patient_', str(patient_id), '.pkl'])
            with open(temp_trajectories_filename, 'wb') as f:
                pickle.dump(self.trajectories[patient_id], f)
        
        self._export_datasets()

        return
    
    def _read_csv_file(self):

        logging.info('reading csv file...')
        # Read the original csv file
        df = pd.read_csv(self.data_filename, delimiter=',', header=0)
        
        # Drop patients that stayed in the ICU for less than 'min_horizon' time steps
        id_counts = df['icustayid'].value_counts()
        df = df[df['icustayid'].isin(id_counts.index[id_counts >= self.min_horizon])]

        # Select and reorder columns
        feature_columns = self.demcols + self.contcols
        df = df[df.columns.intersection(self.idcols + feature_columns + self.inputcols + self.outcomecol)]
        df = df[self.idcols + feature_columns + self.inputcols + self.outcomecol]

        # Discretize action columns
        ivfluids_bins = np.quantile(df[df['input_4hourly'] > 0]['input_4hourly'], [0, 0.25, 0.5, 0.75])
        vaso_bins = np.quantile(df[df['max_dose_vaso'] > 0]['max_dose_vaso'], [0, 0.25, 0.5, 0.75])
        df['ivfluids_bin'] = np.digitize(df['input_4hourly'], ivfluids_bins)
        df['vaso_bin'] = np.digitize(df['max_dose_vaso'], vaso_bins)
        ivfluids_medians = df.groupby(df['ivfluids_bin'])['input_4hourly'].median()
        vaso_medians = df.groupby(df['vaso_bin'])['max_dose_vaso'].median()
        
        df['vaso_action'] = df['vaso_bin']/4 - 0.5
        df['ivfluids_action'] = df['ivfluids_bin']/4 - 0.5
        df['mechvent_action'] = df['mechvent'] - 0.5

        action_dictionary = {}
        for vaso_bin in range(5):
            for ivfluids_bin in range(5):
                for mechvent_bin in range(2):
                    vaso_action = vaso_bin/4 - 0.5
                    ivfluids_action = ivfluids_bin/4 - 0.5
                    mechvent_action = mechvent_bin - 0.5
                    action_dictionary[str((vaso_action, ivfluids_action, mechvent_action))] = {
                        'vaso' : vaso_medians[vaso_bin],
                        'ivfluids' : ivfluids_medians[ivfluids_bin],
                        'mechvent' : mechvent_action
                    }
        
        with open(''.join([self.processed_data_directory, 'action_dictionary.json']), 'w') as f:
            json.dump(action_dictionary, f)
        
        return df

    def _scale_and_drop(self):

        logging.info('scaling features...')
        df = self.raw_df.copy()
        
        # normalize and scale covariates similar to the AI clinician paper
        df[self.colbin] = df[self.colbin]-0.5

        coljoint = self.colnorm + self.collog
        df_selected = df[coljoint]
        scaler = MinMaxScaler()
        df_selected_scaled = scaler.fit_transform(df_selected)
        df[coljoint] = df_selected_scaled - 0.5

        # Store the mins and maxs for later
        self.norm_attributes['min'] = { col : scaler.data_min_[ind] for ind,col in enumerate(coljoint)}
        self.norm_attributes['max'] = { col : scaler.data_max_[ind] for ind,col in enumerate(coljoint)}
        
        # Drop columns that are related to the actions
        df = df.drop(columns=['max_dose_vaso', 'input_4hourly','ivfluids_bin', 'vaso_bin', 'mechvent'])

        with open(''.join([self.processed_data_directory, 'feature_normalization.json']), 'w') as f:
            json.dump(self.norm_attributes, f)
        
        return df

    def _extract_trajectories(self):

        logging.info('extracting trajectories...')
        patient_IDs = self.normalized_df['icustayid'].unique()

        trajectories = {}
        for patient_id in patient_IDs:

            patient_df = self.normalized_df[self.normalized_df['icustayid'] == patient_id]
            trajectories[patient_id] = {'states' : [], 'actions' : []}
            
            # Set the state vectors
            trajectories[patient_id]['states'] = patient_df[self.demcols + self.contcols].to_numpy()
            
            # Set the action vectors and replace the last action with a discharge action (42,42)
            trajectories[patient_id]['actions'] = patient_df[['vaso_action', 'ivfluids_action', 'mechvent_action']].to_numpy()
            trajectories[patient_id]['actions'][-1, :] = np.array([42, 42, 42])

            # If the patient died within 48 hours, set survived to 0, otherwise to 1
            o = patient_df[self.outcomecol].to_numpy().flatten()[-1]
            if o == 1:
                trajectories[patient_id]['survived'] = 0
            elif o == 0:
                trajectories[patient_id]['survived'] = 1

        return trajectories

    def _export_datasets(self):
        '''
        This function takes the trajectories object and creates regression datasets (x, x') for each action and a classification dataset (x, y) for the outcome.
        '''
        
        logging.info('exporting datasets...')

        data_trans = []
        data_outcome = []
        data_all = []
        ids_per_horizon = {}
        # Iterate over IDs
        for patient_ID in self.trajectories:
            
            # Store the ID to the respective horizon group
            horizon = len(self.trajectories[patient_ID]['states'])
            if horizon not in ids_per_horizon:
                ids_per_horizon[horizon] = []
            ids_per_horizon[horizon].append(patient_ID)

            # Create a data point to predict the outcome based on the terminal state (-1: Patient died, 1: Patient survived)
            outcome = self.trajectories[patient_ID]['survived']
            s_last = self.trajectories[patient_ID]['states'][-1]
            a_last = self.trajectories[patient_ID]['actions'][-1]
            data_outcome.append(np.concatenate([s_last, [outcome]]))

            # Iterate over timesteps (except the last one)
            for t in range(self.trajectories[patient_ID]['states'].shape[0] - 1):
                actions = self.trajectories[patient_ID]['actions'][t]
                curr_state = self.trajectories[patient_ID]['states'][t]
                next_state = self.trajectories[patient_ID]['states'][t+1]
                data_trans.append(np.concatenate([actions, curr_state, next_state]))
                data_all.append(np.concatenate([[patient_ID, outcome], actions, curr_state]))
            
            # Add the last state to the dataset
            data_all.append(np.concatenate([[patient_ID, outcome], a_last, s_last]))
        
        # Write outcome dataset to CSV file
        outcome_filename = ''.join([self.processed_data_directory, 'dataset_outcome.csv'])
        df_outcome = pd.DataFrame(np.array(data_outcome), columns=[self.demcols+self.contcols+['Patient_survived']])
        df_outcome.to_csv(outcome_filename, index=False, header=True)

        # Write transition dataset to CSV file
        trans_filename = ''.join([self.processed_data_directory, 'dataset_transitions.csv'])
        demcol_labels = [':'.join(['c', x]) for x in self.demcols]
        df_trans = pd.DataFrame(np.array(data_trans), columns=[['a:vaso', 'a:ivfluids', 'a:mechvent']+demcol_labels+self.contcols+demcol_labels+self.contcols])
        df_trans.to_csv(trans_filename, index=False, header=True)

        # Write trajectory dataset to CSV file
        trajectories_filename = ''.join([self.processed_data_directory, 'trajectories.csv'])
        df_trajectories = pd.DataFrame(np.array(data_all), columns=[['patient_id', 'survived', 'vaso', 'ivfluids', 'mechvent']+self.demcols+self.contcols])
        # make sure that the patient_id and survived columns are integers
        df_trajectories['patient_id'] = df_trajectories['patient_id'].astype(int)
        df_trajectories['survived'] = df_trajectories['survived'].astype(int)
        df_trajectories.to_csv(trajectories_filename, index=False, header=True)

        # Write the horizon groups to txt files
        for horizon in ids_per_horizon:
            horizon_filename = ''.join([self.processed_data_directory, 'horizon_', str(horizon), '.txt'])
            with open(horizon_filename, 'w') as f:
                for ID in ids_per_horizon[horizon]:
                    f.write(str(ID) + '\n')
        
        return

@click.command()
@click.option('--data_filename', type=str, required=True, help='location of original mimic data')
@click.option('--temp_directory', type=str, required=True, help='directory of temporary outputs')
@click.option('--processed_data_directory', type=str, required=True, help='directory to store the processed datasets')
@click.option('--min_horizon', type=int, default=10, help='cutoff for minimum patient trajectory horizon')
def build_datasets(data_filename, temp_directory, processed_data_directory, min_horizon):
    
    logging.basicConfig(level=logging.INFO)
    my_data = Data(data_filename=data_filename, temp_directory=temp_directory,
                        processed_data_directory=processed_data_directory, min_horizon=min_horizon)
    my_data.generate_data()

if __name__ == '__main__':
    build_datasets()