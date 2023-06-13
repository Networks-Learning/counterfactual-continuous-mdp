import numpy as np
import pickle
import torch
import logging
import click
import logging
import os


def solve_facility_location(size, c_dim, logger, device, temp_output_directory):
        
        ###############################################################
        # solve the facility location problem as described in the paper
        ###############################################################
        
        # get all pkl files in the temp_output_directory
        pkl_files = [f for f in os.listdir(temp_output_directory) if f.startswith('trajectory_patient_')]
        
        # load all trajectory states
        s_all = []
        for pkl_file in pkl_files:
            with open(''.join([temp_output_directory, pkl_file]), 'rb') as f:
                trajectory = pickle.load(f)
            s_all.append(torch.tensor(trajectory['states'], dtype=torch.float32, device=device))

        s_all = torch.cat(s_all, dim=0)
        s_all = s_all[:, c_dim:]
        
        s_anchor = s_all[np.random.randint(s_all.shape[0]),:].reshape(1,-1)
        
        while s_anchor.shape[0] < size:
            
            s_all_downsampled = s_all[np.random.choice(s_all.shape[0], size=50000, replace=False),:]
            dists = torch.cdist(s_all_downsampled, s_anchor)
            vals = dists.min(dim=1).values
            max_val, furthest_id = vals.max(dim=0)
            s_anchor = torch.cat((s_anchor, s_all_downsampled[furthest_id,:].reshape(1,-1)), dim=0)

            if s_anchor.shape[0] % 500 == 0:
                logger.info('Current size is {siz} and max. distance is {val}'.format(siz=s_anchor.shape[0], val=max_val))
            if s_anchor.shape[0] % 5000 == 0:
                with open(''.join([temp_output_directory, 'facility_anchor_set_size_', str(s_anchor.shape[0]), '.pkl']), 'wb') as f:
                    pickle.dump(s_anchor.cpu(), f)

        return s_anchor


@click.command()
@click.option('--temp_directory', type=str, required=True, help='directory of temporary outputs')
@click.option('--device', type=str, default='cpu', help='device to run the experiment on')
@click.option('--seed', type=int, default=42, help='random seed')
@click.option('--max_facility_num', type=int, default=1000, help='maximum integer anchor set size')
def facility(temp_directory, device, seed, max_facility_num):
    
    logging.basicConfig(level=logging.INFO)
    
    # set random seed
    np.random.seed(seed)

    # prepare logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler('log.log')
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)

    logger.info('Computing facility location anchor sets...')
    s_anchor = solve_facility_location(size=max_facility_num, c_dim=3, logger=logger, device=device, temp_output_directory=temp_directory)
    return

if __name__ == '__main__':
    facility()
    # facility(temp_directory='outputs/temp_outputs/', device='cpu', seed=42, max_facility_num=1000, nologging=False)