import tempfile
import os
from .split import RandomSplitter, FixedSplitter
from .preprocess import Converter
from .load import UKBDataLoader, BinaryICDLoader, BinarySDLoader
import numpy
from sklearn.linear_model import LinearRegression


DATASET_PATH = os.environ.get('UKB_DATASET_PATH', '/media/data1/ag3r/ukb/dataset/ukb27349.csv')
ICD10_PATH = os.environ.get('UKB_ICD10_PATH', '/media/data1/ag3r/ukb/dataset/ukb44577.csv')


def test_load_real_target():

    columns = ['31-0.0', '50-0.0', '50-1.0', '50-2.0', '21002-0.0', '21002-1.0', '21002-2.0']
    
    zarr_path = tempfile.TemporaryDirectory().name
    converter = Converter([DATASET_PATH], zarr_path, rows_count=20, columns=columns, batch_size=10)

    converter.convert()

    split_dir = tempfile.TemporaryDirectory().name
    split_path = os.path.join(split_dir, 'random')
    splitter = RandomSplitter(zarr_path, split_path, seed=0)
    train, val, test = splitter.split()
    assert len(train) == 16
    assert len(val) == 2
    assert len(test) == 2

    loader = UKBDataLoader(split_dir, 'random', '50', ['31', '21002'])
    train = loader.load_train()
    assert train.shape == (16, 3)
    
    val = loader.load_val()
    assert val.shape == (2, 3)

    test = loader.load_test()
    assert test.shape == (2, 3)

    assert list(train.columns) == ['31', '21002', '50']


def test_load_real_arrayed_target():

    columns = ['31-0.0', '3062-0.0', '3062-0.1', '3062-0.2', '3062-1.0', '3062-1.1', '3062-1.2', '3062-2.0', '3062-2.1', '3062-2.2', '21002-0.0', '21002-1.0', '21002-2.0']
    
    zarr_path = tempfile.TemporaryDirectory().name
    converter = Converter([DATASET_PATH], zarr_path, rows_count=20, columns=columns, batch_size=10)

    converter.convert()

    split_dir = tempfile.TemporaryDirectory().name
    split_path = os.path.join(split_dir, 'random')
    splitter = RandomSplitter(zarr_path, split_path, seed=0)
    train, val, test = splitter.split()
    assert len(train) == 16
    assert len(val) == 2
    assert len(test) == 2

    loader = UKBDataLoader(split_dir, 'random', '3062', ['31', '21002'])
    train = loader.load_train()
    assert train.shape == (15, 3)
    
    val = loader.load_val()
    assert val.shape == (2, 3)

    test = loader.load_test()
    assert test.shape == (2, 3)

    assert list(train.columns) == ['31', '21002', '3062']


def test_real_target_10k_rows(benchmark):
    columns = ['31-0.0', '50-0.0', '50-1.0', '50-2.0', '21002-0.0', '21002-1.0', '21002-2.0']
    
    zarr_path = tempfile.TemporaryDirectory().name
    rows_count = 10*1000
    batch_size = 1000
    converter = Converter([DATASET_PATH], zarr_path, rows_count=rows_count, columns=columns, batch_size=batch_size)

    converter.convert()

    split_dir = tempfile.TemporaryDirectory().name
    split_path = os.path.join(split_dir, 'random')
    splitter = RandomSplitter(zarr_path, split_path, seed=0)
    train, val, test = splitter.split()
    assert len(train) == int(rows_count*0.8)
    assert len(val) == int(rows_count*0.1)
    assert len(test) == int(rows_count*0.1)

    loader = UKBDataLoader(split_dir, 'random', '50', ['31', '21002'])
    benchmark(loader.load_train)
    

def test_real_target_all_rows(benchmark):
    columns = ['31-0.0', '50-0.0', '50-1.0', '50-2.0', '21002-0.0', '21002-1.0', '21002-2.0', '78-0.0']
    
    zarr_path = tempfile.TemporaryDirectory().name
    rows_count = None
    batch_size = 1000
    converter = Converter([DATASET_PATH], zarr_path, rows_count=rows_count, columns=columns, batch_size=batch_size)

    converter.convert()

    split_dir = tempfile.TemporaryDirectory().name
    split_path = os.path.join(split_dir, 'random')
    splitter = RandomSplitter(zarr_path, split_path, seed=0)
    train, val, test = splitter.split()
    assert 502000*0.8 < len(train) < 503000*0.8
    assert 502000*0.1 < len(val) < 503000*0.1
    assert 502000*0.1 < len(test) < 503000*0.1

    loader = UKBDataLoader(split_dir, 'random', '50', ['31', '78', '21002'])
    benchmark(loader.load_train)
    

def test_real_target_regression(benchmark):
    columns = ['31-0.0', '50-0.0', '50-1.0', '50-2.0', '21002-0.0', '21002-1.0', '21002-2.0']
    
    zarr_path = tempfile.TemporaryDirectory().name

    rows_count = 10*1000
    batch_size = 1000
    converter = Converter([DATASET_PATH], zarr_path, rows_count=rows_count, columns=columns, batch_size=batch_size)

    converter.convert()

    split_dir = tempfile.TemporaryDirectory().name
    split_path = os.path.join(split_dir, 'random')
    splitter = RandomSplitter(zarr_path, split_path, seed=0)
    train, val, test = splitter.split()
    assert len(train) == int(rows_count*0.8)
    assert len(val) == int(rows_count*0.1)
    assert len(test) == int(rows_count*0.1)

    loader = UKBDataLoader(split_dir, 'random', '50', ['31', '21002'])
    
    train = loader.load_train()
    train.fillna(0.0, inplace=True)
    X, y = train.iloc[:, :-1].values, train.iloc[:, -1].values

    lr = LinearRegression()
    lr.fit(X, y)

    val = loader.load_val()
    val.fillna(0.0, inplace=True)
    X_val, y_val = val.iloc[:, :-1].values, val.iloc[:, -1].values
    score = lr.score(X_val, y_val)
    print(f'val r^2 is {score:.5f}')
    assert score > 0.33


def test_real_target_all_rows_and_cols(benchmark):

    loader = UKBDataLoader('/media/data1/ag3r/ukb/dataset/all/splits/', 'random', '50', ['31', '78', '21002'])
    benchmark(loader.load_train)
    

def test_binary_icd10_target():
    columns = ['31-0.0', '50-0.0', '50-1.0', '50-2.0', '21002-0.0', '21002-1.0', '21002-2.0']

    sd_columns = []
    
    for arr_idx in range(223): # value from https://biobank.ctsu.ox.ac.uk/crystal/field.cgi?id=41270, array instances
        sd_columns.append(f'41270-0.{arr_idx}')
    
    columns = columns + sd_columns
    zarr_path = tempfile.TemporaryDirectory().name
    converter = Converter([DATASET_PATH, ICD10_PATH], zarr_path, rows_count=1000, columns=columns, batch_size=500)

    converter.convert()

    split_dir = tempfile.TemporaryDirectory().name
    split_path = os.path.join(split_dir, 'random')
    splitter = RandomSplitter(zarr_path, split_path, seed=0)
    train, val, test = splitter.split()
    assert len(train) == 800
    assert len(val) == 100
    assert len(test) == 100
    icd10_code = 'E119' # E11.9 - non-insulin dependent diabetes mellitus without complications
    loader = BinaryICDLoader(split_dir, 'random', '41270', ['31', '50', '21002'], icd10_code) 
    train = loader.load_train()
    assert train.shape == (654, 4)
    
    val = loader.load_val()
    assert val.shape == (80, 4)

    test = loader.load_test()
    assert test.shape == (80, 4)

    assert list(train.columns) == ['31','50', '21002', '41270']

    un, c = numpy.unique(train.iloc[:, -1], return_counts=True)
    assert len(un) == 2
    assert (un == numpy.array([0.0, 1.0])).all()
    assert 100 > c[1] > 10


def test_binary_icd10_target_sexonly():
    columns = ['31-0.0', '50-0.0', '50-1.0', '50-2.0', '21002-0.0', '21002-1.0', '21002-2.0']

    sd_columns = []
    
    for arr_idx in range(223): # value from https://biobank.ctsu.ox.ac.uk/crystal/field.cgi?id=41270, array instances
        sd_columns.append(f'41270-0.{arr_idx}')
    
    columns = columns + sd_columns
    zarr_path = tempfile.TemporaryDirectory().name
    converter = Converter([DATASET_PATH, ICD10_PATH], zarr_path, rows_count=None, columns=columns, batch_size=5000)

    converter.convert()

    split_dir = tempfile.TemporaryDirectory().name
    split_path = os.path.join(split_dir, 'random')
    splitter = RandomSplitter(zarr_path, split_path, seed=0)
    train, val, test = splitter.split()
    assert 401000 < len(train) < 403000
    assert 49000 < len(val) < 52000
    assert 49000 < len(test) < 52000
    icd10_code = 'E119' # E11.9 - non-insulin dependent diabetes mellitus without complications
    loader = BinaryICDLoader(split_dir, 'random', '41270', ['31'], icd10_code) 
    train = loader.load_train()
    assert train.shape[1] == 2
    
    val = loader.load_val()
    assert val.shape[1] == 2

    test = loader.load_test()
    assert test.shape[1] == 2

    assert list(train.columns) == ['31', '41270']

    counts = []
    un, c = numpy.unique(train.iloc[:, -1], return_counts=True)
    assert len(un) == 2
    assert (un == numpy.array([0.0, 1.0])).all()
    counts.append(c[1])

    un, c = numpy.unique(val.iloc[:, -1], return_counts=True)
    counts.append(c[1])

    un, c = numpy.unique(test.iloc[:, -1], return_counts=True)
    counts.append(c[1])
    assert 401000 < len(train) < 403000
    assert sum(counts) == 38791


def test_binary_sd_target():
    columns = ['31-0.0', '50-0.0', '50-1.0', '50-2.0', '21002-0.0', '21002-1.0', '21002-2.0']

    for ai in range(4):
        for arr_idx in range(35): # value from https://biobank.ctsu.ox.ac.uk/crystal/field.cgi?id=20002, array instances
            columns.append(f'20002-{ai}.{arr_idx}')
            
    zarr_path = tempfile.TemporaryDirectory().name
    converter = Converter([DATASET_PATH, ICD10_PATH], zarr_path, 
                           rows_count=1000, columns=columns, batch_size=500)

    converter.convert()

    split_dir = tempfile.TemporaryDirectory().name
    split_path = os.path.join(split_dir, 'random')
    splitter = RandomSplitter(zarr_path, split_path, seed=0)
    train, val, test = splitter.split()
    assert len(train) == 800
    assert len(val) == 100
    assert len(test) == 100
    sd_code = 1220 # diabetes, umbrella code, includes type I and type II diabetes
    loader = BinarySDLoader(split_dir, 'random', '20002', ['31', '50', '21002'], sd_code, na_as_false=False) 
    train = loader.load_train()
    assert train.shape == (607, 4)
    
    val = loader.load_val()
    assert val.shape == (73, 4)

    test = loader.load_test()
    assert test.shape == (78, 4)

    assert list(train.columns) == ['31','50', '21002', '20002']

    un, c = numpy.unique(train.iloc[:, -1], return_counts=True)
    assert len(un) == 2
    assert (un == numpy.array([0.0, 1.0])).all()
    assert 100 > c[1] > 10


def test_binary_sd_target_prevalence():
    columns = ['31-0.0', '50-0.0', '50-1.0', '50-2.0', '21002-0.0', '21002-1.0', '21002-2.0']

    for ai in range(3):
        for arr_idx in range(34): # value from https://biobank.ctsu.ox.ac.uk/crystal/field.cgi?id=20002, array instances
            columns.append(f'20002-{ai}.{arr_idx}')
            
    zarr_path = tempfile.TemporaryDirectory().name
    converter = Converter([DATASET_PATH, ICD10_PATH], zarr_path, 
                           rows_count=None, columns=columns, batch_size=10000, verbose=True)

    converter.convert()

    split_dir = tempfile.TemporaryDirectory().name
    split_path = os.path.join(split_dir, 'random')
    splitter = RandomSplitter(zarr_path, split_path, seed=0)
    train, val, test = splitter.split()
    assert 402000 < len(train) < 403000
    sd_code = 1111 # asthma
    loader = BinarySDLoader(split_dir, 'random', '20002', ['31', '50', '21002'], sd_code) 
    train = loader.load_train()
    assert train.shape[1] == 4
    
    val = loader.load_val()
    assert val.shape[1] == 4

    test = loader.load_test()
    assert test.shape[1] == 4

    assert list(train.columns) == ['31','50', '21002', '20002']

    un, c = numpy.unique(train.iloc[:, -1], return_counts=True)
    assert len(un) == 2
    assert (un == numpy.array([0.0, 1.0])).all()

    prevalence = (train.iloc[:, -1].sum() + val.iloc[:, -1].sum() + test.iloc[:, -1].sum())
    assert 59312 <= prevalence <= 59314


def test_no_aggregation_arrayed_target():
    columns = ['31-0.0', '21002-0.0', '21002-1.0', '21002-2.0']
    pca_columns = [f'22009-0.{i}' for i in range(1, 41)]
    columns += pca_columns

    zarr_path = tempfile.TemporaryDirectory().name

    rows_count = 10*1000
    batch_size = 1000
    converter = Converter([DATASET_PATH], zarr_path, rows_count=rows_count, columns=columns, batch_size=batch_size)

    converter.convert()

    split_dir = tempfile.TemporaryDirectory().name
    split_path = os.path.join(split_dir, 'random')
    splitter = RandomSplitter(zarr_path, split_path, seed=0)
    train, val, test = splitter.split()
    assert len(train) == int(rows_count*0.8)
    assert len(val) == int(rows_count*0.1)
    assert len(test) == int(rows_count*0.1)

    loader = UKBDataLoader(split_dir, 'random', '22009', ['31', '21002'], array_agg_func=None)
    
    train = loader.load_train()
    assert train.shape[1] == 42

    pca = train.values[:, 2:]
    assert numpy.isnan(pca).sum() == 0
    assert -0.05 < pca.mean() < 0.05
    assert 9 < pca.std() < 11 
    