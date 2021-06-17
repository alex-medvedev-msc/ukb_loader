from preprocess import Converter
import zarr


DATASET_PATH = '/media/data1/ag3r/ukb/dataset/ukb27349.csv'

if __name__ == '__main__':
    columns = ['31-0.0', '21002-0.0', '21002-1.0']
    
    zarr_path = '/media/data1/ag3r/ukb/test/zarr'
    converter = Converter(DATASET_PATH, zarr_path, rows_count=20, columns=columns, batch_size=10)

    converter.convert()
    
    array = zarr.open_group(zarr_path, mode='r')