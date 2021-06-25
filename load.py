import numpy
import pandas
from typing import List
import zarr
import os


class UKBDataLoader():
    def __init__(self, data_dir: str, split: str, phenotype_id: str, features: List[str]) -> None:
        
        self.dataset = zarr.open_group(os.path.join(data_dir, split), mode='r')
        self.train, self.val, self.test = self.dataset['train'], self.dataset['val'], self.dataset['test']
        self.columns = self.train.columns[:]

        if '-' in phenotype_id:
            raise ValueError(f'- should not be in {phenotype_id}, we need only a field id without any assessments and array numbers')
        self.phenotype_id = phenotype_id

        for f in features:
            if '-' in f:
                raise ValueError(f'Feature {f} should not contain -, we need only a field id without any assessments and array numbers')

        self.features = features

    def _find_assessment_columns(self, column):
        to_find = column + '-'
        found = []
        for i, col in enumerate(self.columns):
            if col[:len(to_find)] == to_find:
                found.append(i)
        
        return found

    def _process_chunk(self, chunk, target_cols, feature_cols):
        targets = chunk[:, target_cols]
        assessment_indices = targets.shape[1] - numpy.argmax(numpy.flip(~numpy.isnan(targets), axis=1), axis=1) - 1
        true_target = targets[numpy.arange(targets.shape[0]), assessment_indices].reshape(-1, 1)

        features = []
        for fc in feature_cols:
            f = chunk[:, fc]
            if len(fc) == 1: # for features like sex
                features.append(f.reshape(-1, 1))
            else:
                features.append(f[numpy.arange(f.shape[0]), assessment_indices].reshape(-1, 1))
        
        features = numpy.hstack(features)
        return true_target, features

    def _load_real_value_target(self, dataset):
        target_cols = self._find_assessment_columns(self.phenotype_id)
        feature_cols = [self._find_assessment_columns(feature) for feature in self.features]

        data = dataset['dataset']

        targets, all_features = [], []
        for start in range(0, data.shape[0], data.chunks[0]):

            chunk = data[start:start + data.chunks[0]]
            target, features = self._process_chunk(chunk, target_cols, feature_cols)
            targets.append(target)
            all_features.append(features)

        targets = numpy.vstack(targets)
        all_features = numpy.vstack(all_features)
        return targets, all_features

    def _load(self, dataset):
        targets, features = self._load_real_value_target(dataset)

        data = numpy.concatenate([features, targets], axis=1)
        frame = pandas.DataFrame(data=data, columns=self.features + [self.phenotype_id])
        return frame

    def load_train(self) -> pandas.DataFrame:
        return self._load(self.train)

    def load_val(self) -> pandas.DataFrame:
        return self._load(self.val)

    def load_test(self) -> pandas.DataFrame:
        return self._load(self.test)


class BinaryICDLoader():
    def __init__(self, data_dir: str, split: str, phenotype_col: str, features: List[str], icd10_code: str) -> None:
        
        self.dataset = zarr.open_group(os.path.join(data_dir, split), mode='r')
        self.train, self.val, self.test = self.dataset['train'], self.dataset['val'], self.dataset['test']
        self.columns = self.train.columns[:]
        self.str_columns = self.train.str_columns[:]

        if '-' in phenotype_col:
            raise ValueError(f'- should not be in {phenotype_col}, we need only a field id without any assessments and array numbers')
        self.phenotype_col = phenotype_col
        self.icd10_code = icd10_code

        for f in features:
            if '-' in f:
                raise ValueError(f'Feature {f} should not contain -, we need only a field id without any assessments and array numbers')

        self.features = features

    def _find_assessment_columns(self, column):
        to_find = column + '-'
        found = []
        for i, col in enumerate(self.columns):
            if col[:len(to_find)] == to_find:
                found.append(i)
        
        return found

    def _find_arrayed_target_columns(self, column):
        to_find = column + '-'
        found = [[]]
        for i, col in enumerate(self.str_columns):
            if col[:len(to_find)] == to_find:
                assessment = int(col[len(to_find)]) # should always be 0
                found[assessment].append(i)
        
        return found

    def _process_chunk(self, str_chunk, chunk, target_cols, feature_cols):
        codes = []
        for i, assessment_cols in enumerate(target_cols):
            targets = str_chunk[:, assessment_cols]
            code_in = (targets == self.icd10_code).sum(axis=1) > 0
            codes.append(code_in.reshape(-1, 1))

        codes = numpy.hstack(codes)
        assessment_indices = numpy.argmax(codes, axis=1)
        true_target = (codes.sum(axis=1) > 0).astype(int).reshape(-1, 1)

        features = []
        for fc in feature_cols:
            f = chunk[:, fc]
            if len(fc) == 1: # for features like sex
                features.append(f.reshape(-1, 1))
            else:
                features.append(f[numpy.arange(f.shape[0]), assessment_indices].reshape(-1, 1))
        
        features = numpy.hstack(features)
        return true_target, features

    def _load_binary_icd_target(self, dataset):
        target_cols = self._find_arrayed_target_columns(self.phenotype_col)
        feature_cols = [self._find_assessment_columns(feature) for feature in self.features]
        data = dataset['dataset']
        str_data = dataset['str_dataset']
        targets, all_features = [], []
        for start in range(0, data.shape[0], data.chunks[0]):

            chunk = data[start:start + data.chunks[0]]
            str_chunk = str_data[start: start + data.chunks[0]]
            target, features = self._process_chunk(str_chunk, chunk, target_cols, feature_cols)
            targets.append(target)
            all_features.append(features)

        targets = numpy.vstack(targets)
        all_features = numpy.vstack(all_features)
        return targets, all_features

    def _load(self, dataset):
        targets, features = self._load_binary_icd_target(dataset)
        data = numpy.concatenate([features, targets], axis=1)
        frame = pandas.DataFrame(data=data, columns=self.features + [self.phenotype_col])
        return frame

    def load_train(self) -> pandas.DataFrame:
        return self._load(self.train)

    def load_val(self) -> pandas.DataFrame:
        return self._load(self.val)

    def load_test(self) -> pandas.DataFrame:
        return self._load(self.test)
