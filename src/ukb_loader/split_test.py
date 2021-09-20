import tempfile
import os
from .split import RandomSplitter, FixedSplitter
from .preprocess import Converter
import zarr


DATASET_PATH = os.environ.get('UKB_DATASET_PATH', '/media/data1/ag3r/ukb/dataset/ukb27349.csv')
ICD10_PATH = os.environ.get('UKB_ICD10_PATH', '/media/data1/ag3r/ukb/dataset/ukb44577.csv')


def test_random_split():

    columns = ['31-0.0', '21002-0.0', '21002-1.0', '41270-0.0']
    
    zarr_path = tempfile.TemporaryDirectory()
    converter = Converter([DATASET_PATH, ICD10_PATH], zarr_path.name, rows_count=20, columns=columns, batch_size=10)

    converter.convert()

    split_path = os.path.join(tempfile.TemporaryDirectory().name, 'random')
    splitter = RandomSplitter(zarr_path.name, split_path, seed=0)
    train, val, test = splitter.split()
    assert len(train) == 16
    assert len(val) == 2
    assert len(test) == 2

    array = zarr.open_group(f'{split_path}/train', mode='r')
    assert array['dataset'].shape == (16, 3)
    assert array['str_dataset'].shape == (16, 1)


def test_fixed_split():

    columns = ['31-0.0', '21002-0.0', '21002-1.0', '41270-0.0']
    
    zarr_path = tempfile.TemporaryDirectory()
    converter = Converter([DATASET_PATH, ICD10_PATH], zarr_path.name, rows_count=20, columns=columns, batch_size=10)

    converter.convert()

    samples = zarr.open_group(zarr_path.name, mode='r')['eid'][:]

    split_path = os.path.join(tempfile.TemporaryDirectory().name, 'fixed')
    train, val, test = samples[:16], samples[16:18], samples[18:]
    splitter = FixedSplitter(zarr_path.name, split_path, train_samples=list(train.flatten()), val_samples=list(val.flatten()), test_samples=list(test.flatten()))
    train2, val2, test2 = splitter.split()
    assert len(train2) == 16
    assert len(val2) == 2
    assert len(test2) == 2

    assert all(t1 == t2 for t1, t2 in zip(train, train2))
    assert all(t1 == t2 for t1, t2 in zip(val, val2))
    assert all(t1 == t2 for t1, t2 in zip(test, test2))


    array = zarr.open_group(f'{split_path}/train', mode='r')
    assert array['dataset'].shape == (16, 3)
    assert array['str_dataset'].shape == (16, 1)
    
