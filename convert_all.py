from preprocess import Converter
from split import RandomSplitter
import zarr
import time


if __name__ == '__main__':

    dataset_path = '/media/data1/ag3r/ukb/dataset/ukb27349.csv'
    
    zarr_path = '/media/data1/ag3r/ukb/dataset/all/zarr'
    converter = Converter(dataset_path, zarr_path, rows_count=None, columns=None, batch_size=5000)

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
    