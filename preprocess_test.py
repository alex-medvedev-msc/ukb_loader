import numpy
from preprocess import Converter, get_bad_date_columns, get_all_columns, build_dtype_dictionary
import zarr
import pandas


DATASET_PATH = '/media/data1/ag3r/ukb/dataset/ukb27349.csv'
BIOMARKERS_PATH = '/media/data1/ag3r/ukb/dataset/ukb42491.csv'
ICD10_PATH = '/media/data1/ag3r/ukb/dataset/ukb44577.csv'

"""
'eid': 'eid',
    '31-0.0': 'sex',
    '50-0.0': 'height',
    '53-0.0': 'date_assessment',
    '53-1.0': 'date_assessment1',
    '53-2.0': 'date_assessment2',
    '21000-0.0': 'ethnicity',
    '21001-0.0': 'body_mass_index',
    '21002-0.0': 'weight',
"""


def test_build_dtype_dictionary():
    dtype_dict = build_dtype_dictionary([DATASET_PATH, BIOMARKERS_PATH, ICD10_PATH])
    assert dtype_dict['31-0.0'] == pandas.Int64Dtype()
    assert dtype_dict['50-0.0'] == pandas.Float64Dtype()
    assert dtype_dict['53-0.0'] == pandas.StringDtype()
    assert dtype_dict['41270-0.0'] == pandas.StringDtype()


def test_converter():
    columns = ['31-0.0', '50-0.0', '50-1.0', '50-2.0','21002-0.0', '21002-1.0', '21002-2.0']
    
    zarr_path = '/media/data1/ag3r/ukb/test/zarr'
    converter = Converter([DATASET_PATH], zarr_path, rows_count=20, columns=columns, batch_size=10)

    converter.convert()
    
    array = zarr.open_group(zarr_path, mode='r')
    assert array['dataset'].shape == (20, 7)
    assert array['columns'].shape == (7, )
    assert array['eid'].shape == (20, 1)
    assert array['eid'][0] == 1000011
    assert 160 < numpy.nanmean(array['dataset'][:, 1]) < 180
    assert list(array['columns'][:]) == ['31-0.0', '50-0.0', '50-1.0', '50-2.0', '21002-0.0', '21002-1.0', '21002-2.0']


def test_convert_str_columns():
    columns = ['31-0.0', '50-0.0', '50-1.0', '50-2.0', '41270-0.0', '41270-0.1', '41270-0.2']
    
    zarr_path = '/media/data1/ag3r/ukb/test/zarr'
    paths = [DATASET_PATH, ICD10_PATH]
    converter = Converter(paths, zarr_path, rows_count=20, columns=columns, batch_size=10)

    converter.convert()
    
    array = zarr.open_group(zarr_path, mode='r')
    assert array['dataset'].shape == (20, 4)
    assert array['columns'].shape == (4, )
    assert array['eid'].shape == (20, 1)
    assert array['eid'][0] == 1000011
    assert 160 < numpy.nanmean(array['dataset'][:, 1]) < 180
    assert list(array['columns'][:]) == ['31-0.0', '50-0.0', '50-1.0', '50-2.0']

    assert array['str_dataset'].shape == (20, 3)
    assert array['str_columns'].shape == (3, )
    assert list(array['str_columns'][:]) == ['41270-0.0', '41270-0.1', '41270-0.2']


def test_convert_all_columns():
    columns = None
    
    zarr_path = '/media/data1/ag3r/ukb/test/zarr'
    converter = Converter([DATASET_PATH], zarr_path, rows_count=1000, columns=columns, batch_size=500)

    converter.convert()
    
    array = zarr.open_group(zarr_path, mode='r')
    assert array['dataset'].shape == (1000, 11298)
    assert array['eid'].shape == (1000, 1)
    assert array['eid'][0] == 1000011
    # assert list(array['columns'][:]) == ['eid', '31-0.0', '21002-0.0', '21002-1.0']


def test_bad_date_columns():
    all_columns = get_all_columns(DATASET_PATH)
    bdc = get_bad_date_columns(all_columns, date_fields_path='/media/data1/ag3r/ukb/dataset/date_fields.csv')
    assert len(bdc) > 0