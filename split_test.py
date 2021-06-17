from split import RandomSplitter
from preprocess import Converter


DATASET_PATH = '/media/data1/ag3r/ukb/dataset/ukb27349.csv'


def test_random_split():

    columns = ['31-0.0', '21002-0.0', '21002-1.0']
    
    zarr_path = '/media/data1/ag3r/ukb/test/small_20'
    converter = Converter(DATASET_PATH, zarr_path, rows_count=20, columns=columns, batch_size=10)

    converter.convert()

    split_path = '/media/data1/ag3r/ukb/dataset/splits/random/'
    splitter = RandomSplitter(zarr_path, split_path, seed=0)
    train, val, test = splitter.split()
    assert len(train) == 16
    assert len(val) == 2
    assert len(test) == 2
