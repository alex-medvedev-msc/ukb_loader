import pandas
from typing import List


def build_dtype_dictionary(sources: List[str]):
    dtype_dict = {}
    for source in sources:
        for data in pandas.read_csv(source, chunksize=5000, low_memory=False):
            converted = data.convert_dtypes(infer_objects=True, convert_string=True, convert_integer=True)
            for column, dtype in converted.dtypes.iteritems():
                if column not in dtype_dict:
                    dtype_dict[column] = dtype
                elif dtype == pandas.StringDtype():
                    dtype_dict[column] = dtype
        print(f'source {source} converted')
    return dtype_dict


if __name__ == '__main__':

    DATASET_PATH = '/media/data1/ag3r/ukb/dataset/ukb27349.csv'
    BIOMARKERS_PATH = '/media/data1/ag3r/ukb/dataset/ukb42491.csv'
    ICD10_PATH = '/media/data1/ag3r/ukb/dataset/ukb44577.csv'

    dtype_dict = build_dtype_dictionary([DATASET_PATH, BIOMARKERS_PATH, ICD10_PATH])

    dtypes = pandas.DataFrame()
    dtypes.loc[:, 'column'] = list(dtype_dict.keys())
    dtypes.loc[:, 'inferred_dtype'] = list(dtype_dict.values())
    dtypes.to_csv('all_dtypes.csv', index=False)