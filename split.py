import abc
import pandas
from typing import List, Tuple
import zarr
from sklearn.model_selection import train_test_split
import numpy


class Splitter(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def split(self) -> Tuple[List[str], List[str], List[str]]:
        pass

'''
            eid_array = group.zeros('eid', mode='w', shape=(self.rows_count, 1), dtype='i4')
            array = group.zeros('dataset', mode='w', shape=(self.rows_count, len(self.columns)), chunks=(self.batch_size, len(self.columns)), dtype='f4')
            col_array = group.array('columns', ['eid'] + self.columns, dtype='U16')
            col_indices_array = group.array('column_indices', self.column_indices, dtype='i4')
            dates = group.create('dates', mode='w', shape=(self.rows_count, len(self.date_indices)), dtype='M8[D]')
'''

class RandomSplitter(Splitter):

    def __init__(self, zarr_ds_path: str, split_path: str, seed: int = 0, proportions: List[int] = None) -> None:
        super().__init__()
        self.seed = seed
        self.zarr_ds_path = zarr_ds_path
        self.split_path = split_path
        # proportions are [train_ratio, val_ratio, test_ratio]
        if proportions is None:
            self.proportions = [0.8, 0.1, 0.1]
        elif len(proportions) == 3 and abs(sum(proportions) - 1) < 1e-7:
            self.proportions = proportions
        else:
            raise ValueError(f'len of proportions {proportions} should be 3 or sum {sum(proportions)} should be 1')

    def split(self) -> Tuple[List[str], List[str], List[str]]:
        source = zarr.open_group(self.zarr_ds_path, mode='r')
        target = zarr.open_group(self.split_path, mode='w')
        samples = source['eid'][:]

        x_train, x_testval = train_test_split(numpy.arange(len(samples)), test_size=1-self.proportions[0], random_state=self.seed)

        x_val, x_test = train_test_split(x_testval, test_size=self.proportions[2]/(self.proportions[1] + self.proportions[2]), random_state=self.seed)

        train, val, test = target.create_groups('train', 'val', 'test')
        dataset = source['dataset']
        x_train, x_val, x_test = numpy.sort(x_train), numpy.sort(x_val), numpy.sort(x_test)

        for group, x in zip([train, val, test], [x_train, x_val, x_test]):
            group.array('eid', samples[x])
            group.array('columns', source['columns'][:])
            group.array('column_indices', source['column_indices'][:])
            group.array('dates', source['dates'][:][x])

            new_dataset = group.zeros('dataset', shape=(0, dataset.shape[1]), chunks=dataset.chunks)

            chunk_size = dataset.chunks[0]
            for chunk_start in range(0, len(x), chunk_size):
                rows = min(dataset.shape[0] - chunk_start, chunk_size) 
                chunk = dataset.get_orthogonal_selection((x[chunk_start: chunk_start + rows], slice(None)))
                new_dataset.append(chunk, axis=0)

        return list(samples[x_train]), list(samples[x_val]), list(samples[x_test])

