from .preprocess import Converter
from .split import RandomSplitter, FixedSplitter
import zarr
import time
from typing import List
import pandas


def convert_fixed_split(sources: List[str], samples_files: List[str], zarr_path: str, split_path: str, batch_size: int = 2048):

    converter = Converter(sources, zarr_path, 
                          rows_count=None, 
                          columns=None, 
                          batch_size=batch_size, 
                          verbose=True)

    start = time.time()
    converter.convert()
    end = time.time()

    print(f'converting dataset with all columns required {end - start:.3f}s')
    array = zarr.open_group(zarr_path, mode='r')
    rows, cols = array['dataset'].shape
    print(f'we have total of {rows} rows and {cols} columns in main dataset')
    
    # train_samples_path = '/home/ag3r/ukb_ml/processed_data/white_british_split_filtered/train_ids.tsv'
    train_samples_path, val_samples_path, test_samples_path = samples_files

    start = time.now()
    train_samples = pandas.read_table(train_samples_path, header=None, names=['fid', 'iid']).iid.tolist()
    val_samples = pandas.read_table(val_samples_path, header=None, names=['fid', 'iid']).iid.tolist()
    test_samples = pandas.read_table(test_samples_path, header=None, names=['fid', 'iid']).iid.tolist()

    splitter = FixedSplitter(zarr_path, split_path, train_samples=train_samples, val_samples=val_samples, test_samples=test_samples)
    train2, val2, test2 = splitter.split()
    end = time.now()
    print(f'splitting dataset with all columns required {end - start:.3f}s')


def convert_everything(sources: List[str], zarr_path: str, split_path: str, batch_size: int = 2048):

    converter = Converter(sources, zarr_path, 
                          rows_count=None, 
                          columns=None, 
                          batch_size=batch_size, 
                          verbose=True)

    start = time.time()
    converter.convert()
    end = time.time()

    print(f'converting dataset with all columns required {end - start:.3f}s')
    array = zarr.open_group(zarr_path, mode='r')
    rows, cols = array['dataset'].shape
    print(f'we have total of {rows} rows and {cols} columns in main dataset')

    split_path = '/media/data1/ag3r/ukb/dataset/all/splits/random'
    start = time.time()
    splitter = RandomSplitter(zarr_path, split_path, seed=0)
    train, val, test = splitter.split()
    end = time.time()
    print(f'randomly splitting main dataset into train, val, test took {end - start:.3f}s')
    print(f'train has {len(train)} rows, val has {len(val)} rows, test has {len(test)} rows')

    