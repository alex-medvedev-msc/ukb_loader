import enum
from typing import List, Tuple
from numpy import dtype, iinfo
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


def find_indices(source, to_find):
    indices = []
    found = []
    for fi, f in enumerate(to_find):
        for i, s in enumerate(source):
            if f == s:
                indices.append(i)
                found.append(f)
                break
    return indices, found

class IndexData:
    def __init__(self, date: List[int], date_found: List[str], 
                 str_col: List[int], str_found: List[str], 
                 float_col: List[int], float_found: List[str]) -> None:
        self.date = date
        self.str_col = str_col
        self.float_col = float_col
        self.date_found = date_found
        self.str_found = str_found
        self.float_found = float_found


class Converter:
    def __init__(self, datasets: List[str], zarr_path: str, rows_count: int = None, batch_size: int = 1024, columns: List[str] = None) -> None:
        self.datasets = datasets
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
        
        self.all_columns = {path: get_all_columns(path) for path in self.datasets}
        self.eid_index = 0
        
        if self.rows_count is None:
            self.rows_count = 502536

        self.index_dict = self._create_index_dict(self.datasets, self.all_columns, self.columns)


    def _create_index_dict(self, datasets, all_columns, requested_columns):
        index_dict = {}
        for path in datasets:
            ac = all_columns[path]
            all_str_columns = get_bad_columns(ac, DATE_FIELDS_PATH, BAD_COLS_PATH)
            date_indices, date_found = find_indices(ac, DATE_COLUMNS)
            # good float columns from path
            if requested_columns is None:
                ac_columns = [d for d in ac if d not in DATE_COLUMNS and d not in all_str_columns]
                str_columns = [d for d in ac if d in all_str_columns]
            else:
                ac_columns = [d for d in ac if d not in DATE_COLUMNS and d in requested_columns and d not in all_str_columns]
                str_columns = [d for d in ac if d in requested_columns and d in all_str_columns]
            
            float_col, found_float_col = find_indices(ac, ac_columns)
            str_col, found_str_col = find_indices(ac, str_columns)

            index_data = IndexData(date_indices, date_found, str_col, found_str_col, float_col, found_float_col)
            index_dict[path] = index_data

        return index_dict

    
    def _read_dates_eid(self, source_path, date_indices, dates, eid_array):
        for i, chunk in enumerate(
                        pandas.read_csv(
                            source_path,
                            usecols=[self.eid_index] + date_indices,
                            chunksize=self.batch_size,
                            nrows=self.rows_count,
                            parse_dates=list(range(1, len(date_indices) + 1)))
                    ):
                start, end = i*self.batch_size, i*self.batch_size + chunk.shape[0]
                dates[start: end, :] = chunk.iloc[:, 1:].values
                eid_array[start: end] = chunk.iloc[:, 0].values.reshape(-1, 1)

    def _read_float(self, source_path, indices, dataset, left, col_array, cols):
        for j, fc in enumerate(
                        pandas.read_csv(
                            source_path,
                            usecols=indices,
                            chunksize=self.batch_size,
                            nrows=self.rows_count)
                    ):
                start, end = j*self.batch_size, j*self.batch_size + fc.shape[0]
                right = left + len(indices)
                dataset[start: end, left:right] = fc.apply(pandas.to_numeric, errors='coerce').values
                col_array[left:right] = cols

    def _read_str(self, source_path, indices, str_dataset, left, col_array, cols):
        for j, fc in enumerate(
                        pandas.read_csv(
                            source_path,
                            usecols=indices,
                            chunksize=self.batch_size,
                            nrows=self.rows_count)
                    ):
                start, end = j*self.batch_size, j*self.batch_size + fc.shape[0]
                fc.fillna('', inplace=True)
                right = left + len(indices)
                str_dataset[start: end, left:right] = fc.astype(str).values
                col_array[left:right] = cols

    def _calculate_str_float_array_len(self):
        return sum([len(indices.str_col) for indices in self.index_dict.values()]),\
               sum([len(indices.float_col) for indices in self.index_dict.values()])

    def convert(self):

        with zarr.open_group(self.zarr_path, mode='w') as group:

            str_len, float_len = self._calculate_str_float_array_len()

            eid_array = group.zeros('eid', mode='w', shape=(self.rows_count, 1), dtype='i4')
            array = group.zeros('dataset', mode='w', shape=(self.rows_count, float_len), chunks=(self.batch_size, float_len), dtype='f4')
            col_array = group.create('columns', shape=(float_len, ), dtype='U16')
            dates_array = group.create('dates', mode='w', shape=(self.rows_count, len(DATE_COLUMNS)), dtype='M8[D]')
            str_array = group.create('str_dataset', mode='w', shape=(self.rows_count, str_len), dtype='U16')
            str_col_array = group.create('str_columns', mode='w', shape=(str_len, ), dtype='U16')
            
            str_left, float_left = 0, 0
            for path, indices in self.index_dict.items():
                if len(indices.date) > 0:
                    self._read_dates_eid(path, indices.date, dates_array, eid_array)
                if len(indices.str_col) > 0:
                    self._read_str(path, indices.str_col, str_array, str_left, str_col_array, indices.str_found)
                    str_left += len(indices.str_col)
                if len(indices.float_col) > 0:
                    self._read_float(path, indices.float_col, array, float_left, col_array, indices.float_found)
                    float_left += len(indices.float_col)


    def convert_old(self):

        with zarr.open_group(self.zarr_path, mode='w') as group:

            eid_array = group.zeros('eid', mode='w', shape=(self.rows_count, 1), dtype='i4')
            array = group.zeros('dataset', mode='w', shape=(self.rows_count, len(self.columns)), chunks=(self.batch_size, len(self.columns)), dtype='f4')
            col_array = group.array('columns', self.columns, dtype='U16')
            col_indices_array = group.array('column_indices', self.column_indices, dtype='i4')
            dates = group.create('dates', mode='w', shape=(self.rows_count, len(self.date_indices)), dtype='M8[D]')
            str_array = group.create('str_dataset', mode='w', shape=(self.rows_count, len(self.bad_column_indices)), dtype='U16')
            str_col_array = group.array('str_columns', self.bad_columns, dtype='U16')
            
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
                array[start: end, :] = fc.apply(pandas.to_numeric, errors='coerce').values

            for j, fc in enumerate(
                        pandas.read_csv(
                            self.dataset_path,
                            usecols=self.bad_column_indices,
                            chunksize=self.batch_size,
                            nrows=self.rows_count)
                    ):
                start, end = j*self.batch_size, j*self.batch_size + fc.shape[0]
                fc.fillna('', inplace=True)
                str_array[start: end, :] = fc.astype(str).values
    