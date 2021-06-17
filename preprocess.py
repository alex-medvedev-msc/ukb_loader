import enum
from typing import List, Tuple
from numpy import dtype
from pandas.core.frame import DataFrame
import zarr
import pandas


DATE_COLUMNS = ['53-0.0', '53-1.0', '53-2.0']
DATE_FIELDS_PATH = '/media/data1/ag3r/ukb/dataset/date_fields.csv'
BAD_COLS_PATH = '/media/data1/ag3r/ukb/dataset/bad_cols_back.csv'


def get_all_columns(dataset_path: str) -> List[str]:
    with open(dataset_path, 'r') as d:
        all_columns = [c.strip().strip('"') for c in d.readline().split(',')]

    return all_columns

def get_bad_columns(columns, date_fields_path: str = 'date_fields.csv', bad_cols_path: str = 'bad_cols.csv') -> List[str]:
    dc = set(pandas.read_csv(date_fields_path).field_id)
    bad = []
    for c in columns[1:]: # without eid
        if int(c.split('-')[0]) in dc:
            bad.append(c)

    bc = pandas.read_csv(bad_cols_path).field_id
    bc = set([int(c.split('-')[0]) for c in bc])
    for c in columns[1:]:
        if int(c.split('-')[0]) in bc:
            bad.append(c)

    return bad

def get_bad_date_columns(columns, date_fields_path: str = 'date_fields.csv') -> List[str]:
    dc = set(pandas.read_csv(date_fields_path).field_id)
    bad = []
    for c in columns[1:]: # without eid
        if int(c.split('-')[0]) in dc:
            bad.append(c)

    return bad


class Converter:
    def __init__(self, dataset_path: str, zarr_path: str, rows_count: int = None, batch_size: int = 1024, columns: List[str] = None) -> None:
        self.dataset_path = dataset_path
        self.zarr_path = zarr_path
        self.rows_count = rows_count
        self.columns = columns
        self.batch_size = batch_size

        if columns is not None and 'eid' in columns:
            raise ValueError('Please remove eid from columns, it will be read anyway')
        if columns is not None:
            for d in DATE_COLUMNS:
                if d in columns:
                    raise ValueError(f'Please remove {d} column, date assessments will be loaded anyway')

        self.all_columns = get_all_columns(dataset_path)
        self.eid_index = 0
        
        # quadratic, but probably not that bad
        self.date_indices = [self.all_columns.index(d) for d in DATE_COLUMNS]
        if columns is not None:
            self.column_indices = [self.all_columns.index(c) for c in self.columns]
            self.column_indices = self.column_indices
        else:
            bad_columns = get_bad_columns(self.all_columns, DATE_FIELDS_PATH, BAD_COLS_PATH)
            self.columns = [d for d in self.all_columns[1:] if d not in DATE_COLUMNS and d not in bad_columns] # remove eid and date columns
            self.column_indices = [self.all_columns.index(c) for c in self.columns]
        
        if self.rows_count is None:
            self.rows_count = 502536


    def convert(self):

        with zarr.open_group(self.zarr_path, mode='w') as group:

            eid_array = group.zeros('eid', mode='w', shape=(self.rows_count, 1), dtype='i4')
            array = group.zeros('dataset', mode='w', shape=(self.rows_count, len(self.columns)), chunks=(self.batch_size, len(self.columns)), dtype='f4')
            col_array = group.array('columns', self.columns, dtype='U16')
            col_indices_array = group.array('column_indices', self.column_indices, dtype='i4')
            dates = group.create('dates', mode='w', shape=(self.rows_count, len(self.date_indices)), dtype='M8[D]')
            
            for i, chunk in enumerate(
                        pandas.read_csv(
                            self.dataset_path,
                            usecols=[self.eid_index] + self.date_indices,
                            chunksize=self.batch_size,
                            nrows=self.rows_count,
                            parse_dates=list(range(1, len(self.date_indices) + 1)))
                    ):
                start, end = i*self.batch_size, i*self.batch_size + chunk.shape[0]
                dates[start: end, :] = chunk.iloc[:, 1:].values
                eid_array[start: end] = chunk.iloc[:, 0].values.reshape(-1, 1)

            for j, fc in enumerate(
                        pandas.read_csv(
                            self.dataset_path,
                            usecols=self.column_indices,
                            chunksize=self.batch_size,
                            nrows=self.rows_count)
                    ):
                start, end = j*self.batch_size, j*self.batch_size + fc.shape[0]
                # bad_cols = [self.column_indices.index(c) for c in [22]]
                # names = [self.columns[bc] for bc in bad_cols]
                # print(names)
                array[start: end, :] = fc.apply(pandas.to_numeric, errors='coerce').values
    