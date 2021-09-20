import enum
from typing import List, Tuple
from numpy import dstack, dtype, iinfo
import numpy
from pandas.core.frame import DataFrame
import zarr
import pandas
import pkg_resources


DATE_COLUMNS = ['53-0.0', '53-1.0', '53-2.0']


def load_dtype_dictionary():
    path = pkg_resources.resource_filename('ukb_loader', 'all_dtypes.csv')
    dtype_frame = pandas.read_csv(path)

    dtype_dict = {}
    convert_dict = {
        'string': pandas.StringDtype(),
        'Int64': pandas.Int64Dtype(),
        'Float64': pandas.Float64Dtype()
    }
    for col, d in zip(dtype_frame.column, dtype_frame.inferred_dtype):
        dtype_dict[col] = convert_dict[d]

    return dtype_dict


def get_all_columns(dataset_path: str) -> List[str]:
    with open(dataset_path, 'r') as d:
        all_columns = [c.strip().strip('"') for c in d.readline().split(',')]

    return all_columns


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
    def __init__(self, datasets: List[str], zarr_path: str, rows_count: int = None, batch_size: int = 1024, columns: List[str] = None, verbose: bool = False) -> None:
        self.datasets = datasets
        self.zarr_path = zarr_path
        self.rows_count = rows_count
        self.columns = columns
        self.batch_size = batch_size
        self.str_batch_size = batch_size // 8
        self.verbose = verbose
        if not isinstance(datasets, List) or len(datasets) == 0:
            raise ValueError(f'Datasets should be a list of full paths to them, not {datasets}')
        if columns is not None and 'eid' in columns:
            raise ValueError('Please remove eid from columns, it will be read anyway')
        if columns is not None:
            for d in DATE_COLUMNS:
                if d in columns:
                    raise ValueError(f'Please remove {d} column, date assessments will be loaded anyway')
        
        self.all_columns = {path: get_all_columns(path) for path in self.datasets}
        self.eid_index = 0
        
        self.dtype_dict = load_dtype_dictionary()
        self.index_dict = self._create_index_dict(self.datasets, self.all_columns, self.columns, self.dtype_dict)
        self.common_eids, self.masks_indices = self._create_eid_data(datasets)

        if self.rows_count is None:
            self.rows_count = len(self.common_eids)


    def _get_ac_str_columns(self, requested_columns, columns, all_str_columns, index_dict):
        old_ac_set = set()
        old_str_set = set()
        for index_data in index_dict.values():
            old_ac_set |= set(index_data.float_found)
            old_str_set |= set(index_data.str_found)
        
        if requested_columns is None:
            ac_columns = [d for d in columns if d not in DATE_COLUMNS and d not in all_str_columns and d not in old_ac_set]
            str_columns = [d for d in columns if d in all_str_columns and d not in old_str_set]
        else:
            ac_columns = [d for d in columns if d not in DATE_COLUMNS and d in requested_columns and d not in all_str_columns and d not in old_ac_set]
            str_columns = [d for d in columns if d in requested_columns and d in all_str_columns and d not in old_str_set]

        return ac_columns, str_columns

    def _create_index_dict(self, datasets, all_columns, requested_columns, dtype_dict):
        index_dict = {}
        for path in datasets:
            ac = all_columns[path]
            # all_str_columns = get_bad_columns(ac, DATE_FIELDS_PATH, BAD_COLS_PATH)
            all_str_columns = set()
            for col, dtype in dtype_dict.items():
                if dtype == object or dtype == pandas.StringDtype():
                    if col not in DATE_COLUMNS and col in ac:
                        all_str_columns.add(col)

            date_indices, date_found = find_indices(ac, DATE_COLUMNS)
            # good float columns from path
            ac_columns, str_columns = self._get_ac_str_columns(requested_columns, ac, all_str_columns, index_dict)

            float_col, found_float_col = find_indices(ac, ac_columns)
            str_col, found_str_col = find_indices(ac, str_columns)

            index_data = IndexData(date_indices, date_found, str_col, found_str_col, float_col, found_float_col)
            index_dict[path] = index_data

        return index_dict

    
    def _create_eid_data(self, datasets):
        eid_dict = {}
        common_eids = None
        for path in datasets:
            if self.verbose:
                print(f'starting reading eid data from {path}')
            eid = pandas.read_csv(path, usecols=[0], nrows=self.rows_count).iloc[:, 0].values
            if common_eids is None:
                common_eids = set(eid)
            else:
                common_eids |= set(eid)
            eid_dict[path] = eid
            if self.verbose:
                print(f'ended reading eid data from {path}')

        masks_indices = {}
        ce_list = sorted(list(common_eids))
        for path, eid in eid_dict.items():
            eid_set = set(eid)
            mask = numpy.array([True if e in eid_set else False for e in ce_list])
            mask_indices = numpy.arange(len(ce_list))[mask]
            masks_indices[path] = mask_indices
            if self.verbose:
                print(f'mask analysis for {path}')
                print(f'min: {min(mask_indices)}, max: {max(mask_indices)}, sum: {mask.sum()}')
            

        if self.verbose:
            print(f'created masks_indices and eid_dict')
        return ce_list, masks_indices

    
    def _read_dates_eid(self, source_path: str, date_indices, dates: zarr.Array, eid_array: zarr.Array, eid_mask_indices):
        for i, chunk in enumerate(
                        pandas.read_csv(
                            source_path,
                            usecols=[self.eid_index] + date_indices,
                            chunksize=self.batch_size,
                            nrows=self.rows_count,
                            encoding='cp1252',
                            parse_dates=list(range(1, len(date_indices) + 1)))
                    ):
                start, end = i*self.batch_size, i*self.batch_size + chunk.shape[0]
                chunk_indices = eid_mask_indices[start: end]
                dates.set_orthogonal_selection((chunk_indices, slice(None)), chunk.iloc[:, 1:].values)
                eid_chunk = chunk.iloc[:, 0].values.reshape(-1, 1)
                eid_array.set_orthogonal_selection((chunk_indices, slice(None)),  eid_chunk)

    def _read_float(self, source_path: str, indices, dataset: zarr.Array, left, col_array, cols, eid_mask_indices):
        for j, fc in enumerate(
                        pandas.read_csv(
                            source_path,
                            usecols=indices,
                            chunksize=self.batch_size,
                            nrows=self.rows_count,
                            encoding='cp1252',
                            low_memory=False)
                    ):
                start, end = j*self.batch_size, j*self.batch_size + fc.shape[0]
                right = left + len(indices)
                chunk_indices = eid_mask_indices[start: end]
                dataset.set_orthogonal_selection((chunk_indices, slice(left, right)), fc.apply(pandas.to_numeric, errors='coerce').values)
                # dataset[chunk_indices, left:right] = fc.apply(pandas.to_numeric, errors='coerce').values
                col_array[left:right] = cols
                if self.verbose:
                    print(f'converted float batch number {j} with {len(cols)} columns')

    def _read_str(self, source_path: str, indices, str_dataset: zarr.Array, left, col_array, cols, eid_mask_indices):
        for j, fc in enumerate(
                        pandas.read_csv(
                            source_path,
                            usecols=indices,
                            chunksize=self.str_batch_size,
                            nrows=self.rows_count,
                            encoding='cp1252',
                            low_memory=False)
                    ):
                start, end = j*self.str_batch_size, j*self.str_batch_size + fc.shape[0]
                fc.fillna('', inplace=True)
                right = left + len(indices)
                chunk_indices = eid_mask_indices[start: end]
                str_dataset.set_orthogonal_selection((chunk_indices, slice(left, right)), fc.astype(str).values)
                # str_dataset[chunk_indices, left:right] = fc.astype(str).values
                col_array[left:right] = cols
                if self.verbose:
                    print(f'converted str batch number {j} with {len(cols)} columns')

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
            str_array = group.create('str_dataset', mode='w', shape=(self.rows_count, str_len), dtype='U16', chunks=(self.batch_size, str_len))
            str_col_array = group.create('str_columns', mode='w', shape=(str_len, ), dtype='U16')
            
            str_left, float_left = 0, 0
            for path, indices in self.index_dict.items():
                if self.verbose:
                    print()
                    print(f'Starting to convert dataset {path}')
                eid_mask_indices = self.masks_indices[path]
                if len(indices.date) > 0:
                    self._read_dates_eid(path, indices.date, dates_array, eid_array, eid_mask_indices)
                if len(indices.str_col) > 0:
                    self._read_str(path, indices.str_col, str_array, str_left, str_col_array, indices.str_found, eid_mask_indices)
                    str_left += len(indices.str_col)
                if len(indices.float_col) > 0:
                    self._read_float(path, indices.float_col, array, float_left, col_array, indices.float_found, eid_mask_indices)
                    float_left += len(indices.float_col)

    