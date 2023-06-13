from math import radians
import numpy as np
from scipy.stats.mstats import gmean
import click
from joblib import Parallel, delayed
import json
import multiprocessing as mp

# Compute the arithmetic and geometric mean of N random numbers in [0,1)
def computation(N, seed):
    
    rng = np.random.default_rng(seed=seed)
    random_numbers = rng.random(size=N)
    ar_mean = np.mean(random_numbers)
    geo_mean = gmean(random_numbers)
    
    return ar_mean, geo_mean

# Saves configuration and results to a JSON file
def generate_summary(N, ar_means, geo_means, seeds):

    summary = {}
    summary['N'] = str(N)
        
    summary['seeds'] = {}
    for seed in range(1, seeds+1):
        summary['seeds'][seed] = {}
        summary['seeds'][seed]['ar_mean'] = str(ar_means[seed-1])
        summary['seeds'][seed]['geo_mean'] = str(geo_means[seed-1])
         
    return summary

@click.command() # Comment the click commands for testing
@click.option('--n', type=int, required=True, help="parameter N")
@click.option('--seeds', type=int, required=True, help="Number of seeds")
@click.option('--njobs', type=int, required=True, help="Number of parallel threads")
@click.option('--output', type=str, required=True, help="Output file name")
def experiment(n, seeds, njobs, output):

    N = n # click doesn't accept upper case arguments

    print('Computing...')
    results = Parallel(n_jobs=njobs, backend='multiprocessing')(delayed(computation)(N, seed) for seed in range(1, seeds+1))

    ar_means = [x[0] for x in results]
    geo_means = [x[1] for x in results]

    print('Saving results...')
    summary = generate_summary(N=N, ar_means=ar_means, geo_means=geo_means, seeds=seeds)
        
    with open('{output}.json'.format(output=output), 'w') as outfile:
        json.dump(summary, outfile)

    return

# Testing function
def testing(N, seed):
    
    result = computation(N, seed)
    return result

if __name__ == '__main__':
    experiment()
    # testing(N=20, seed=2)