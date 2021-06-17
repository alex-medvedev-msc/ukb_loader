from split import RandomSplitter
from preprocess import Converter
from load import UKBDataLoader, BinaryICDLoader
import numpy
from sklearn.linear_model import LinearRegression


DATASET_PATH = '/media/data1/ag3r/ukb/dataset/ukb27349.csv'


def test_load_real_target():

    columns = ['31-0.0', '50-0.0', '50-1.0', '50-2.0', '21002-0.0', '21002-1.0', '21002-2.0']
    
    zarr_path = '/media/data1/ag3r/ukb/test/small_20'
    converter = Converter(DATASET_PATH, zarr_path, rows_count=20, columns=columns, batch_size=10)

    converter.convert()

    split_path = '/media/data1/ag3r/ukb/test/splits/random'
    splitter = RandomSplitter(zarr_path, split_path, seed=0)
    train, val, test = splitter.split()
    assert len(train) == 16
    assert len(val) == 2
    assert len(test) == 2

    loader = UKBDataLoader('/media/data1/ag3r/ukb/test/splits/', 'random', '50', ['31', '21002'])
    train = loader.load_train()
    assert train.shape == (16, 3)
    
    val = loader.load_val()
    assert val.shape == (2, 3)

    test = loader.load_test()
    assert test.shape == (2, 3)

    assert list(train.columns) == ['31', '21002', '50']


def test_real_target_10k_rows(benchmark):
    columns = ['31-0.0', '50-0.0', '50-1.0', '50-2.0', '21002-0.0', '21002-1.0', '21002-2.0']
    
    zarr_path = '/media/data1/ag3r/ukb/test/avg_10k'
    rows_count = 10*1000
    batch_size = 1000
    converter = Converter(DATASET_PATH, zarr_path, rows_count=rows_count, columns=columns, batch_size=batch_size)

    converter.convert()

    split_path = '/media/data1/ag3r/ukb/test/splits/random'
    splitter = RandomSplitter(zarr_path, split_path, seed=0)
    train, val, test = splitter.split()
    assert len(train) == int(rows_count*0.8)
    assert len(val) == int(rows_count*0.1)
    assert len(test) == int(rows_count*0.1)

    loader = UKBDataLoader('/media/data1/ag3r/ukb/test/splits/', 'random', '50', ['31', '21002'])
    benchmark(loader.load_train)
    '''
    train = loader.load_train()
    assert train.shape == (int(rows_count*0.8), 3)
    
    val = loader.load_val()
    assert val.shape == (int(rows_count*0.1), 3)

    test = loader.load_test()
    assert test.shape == (int(rows_count*0.1), 3)

    assert list(train.columns) == ['31', '21002', '50']

    assert 165 < numpy.nanmean(train['50']) < 175
    assert numpy.isnan(train['50'].values).sum() < 50
    '''


def test_real_target_all_rows(benchmark):
    columns = ['31-0.0', '50-0.0', '50-1.0', '50-2.0', '21002-0.0', '21002-1.0', '21002-2.0', '78-0.0']
    
    zarr_path = '/media/data1/ag3r/ukb/test/all_rows'
    rows_count = None
    batch_size = 1000
    converter = Converter(DATASET_PATH, zarr_path, rows_count=rows_count, columns=columns, batch_size=batch_size)

    converter.convert()

    split_path = '/media/data1/ag3r/ukb/test/splits/random'
    splitter = RandomSplitter(zarr_path, split_path, seed=0)
    train, val, test = splitter.split()
    assert 502000*0.8 < len(train) < 503000*0.8
    assert 502000*0.1 < len(val) < 503000*0.1
    assert 502000*0.1 < len(test) < 503000*0.1

    loader = UKBDataLoader('/media/data1/ag3r/ukb/test/splits/', 'random', '50', ['31', '78', '21002'])
    benchmark(loader.load_train)
    '''
    train = loader.load_train()
    assert train.shape == (int(rows_count*0.8), 3)
    
    val = loader.load_val()
    assert val.shape == (int(rows_count*0.1), 3)

    test = loader.load_test()
    assert test.shape == (int(rows_count*0.1), 3)

    assert list(train.columns) == ['31', '21002', '50']

    assert 165 < numpy.nanmean(train['50']) < 175
    assert numpy.isnan(train['50'].values).sum() < 50
    '''


def test_real_target_regression(benchmark):
    columns = ['31-0.0', '50-0.0', '50-1.0', '50-2.0', '21002-0.0', '21002-1.0', '21002-2.0']
    
    zarr_path = '/media/data1/ag3r/ukb/test/avg_10k'
    rows_count = 10*1000
    batch_size = 1000
    converter = Converter(DATASET_PATH, zarr_path, rows_count=rows_count, columns=columns, batch_size=batch_size)

    converter.convert()

    split_path = '/media/data1/ag3r/ukb/test/splits/random'
    splitter = RandomSplitter(zarr_path, split_path, seed=0)
    train, val, test = splitter.split()
    assert len(train) == int(rows_count*0.8)
    assert len(val) == int(rows_count*0.1)
    assert len(test) == int(rows_count*0.1)

    loader = UKBDataLoader('/media/data1/ag3r/ukb/test/splits/', 'random', '50', ['31', '21002'])
    
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

def test_binary_target():
    columns = ['31-0.0', '50-0.0', '50-1.0', '50-2.0', '21002-0.0', '21002-1.0', '21002-2.0']

    sd_columns = []
    for assessment in range(3):
        for arr_idx in range(34):
            sd_columns.append(f'20002-{assessment}.{arr_idx}')
    
    columns = columns + sd_columns
    zarr_path = '/media/data1/ag3r/ukb/test/small_100'
    converter = Converter(DATASET_PATH, zarr_path, rows_count=1000, columns=columns, batch_size=500)

    converter.convert()

    split_path = '/media/data1/ag3r/ukb/test/splits/random'
    splitter = RandomSplitter(zarr_path, split_path, seed=0)
    train, val, test = splitter.split()
    assert len(train) == 800
    assert len(val) == 100
    assert len(test) == 100

    loader = BinaryICDLoader('/media/data1/ag3r/ukb/test/splits/', 'random', '20002', ['31', '50', '21002'], '1226') # 1226 - hypothyroidism
    train = loader.load_train()
    assert train.shape == (800, 4)
    
    val = loader.load_val()
    assert val.shape == (100, 4)

    test = loader.load_test()
    assert test.shape == (100, 4)

    assert list(train.columns) == ['31','50', '21002', '20002']

    un, c = numpy.unique(train.iloc[:, -1], return_counts=True)
    assert len(un) == 2
    assert (un == numpy.array([0.0, 1.0])).all()
    assert 100 > c[1] > 10
    
