from bed_reader import open_bed
from typing import List, Tuple
import pandas
import numpy
import zarr
    

def get_variant_mask(gwas_results_path: str, limit: int):
    results = pandas.read_table(gwas_results_path)
    results.loc[:, 'ind'] = numpy.arange(results.shape[0])
    sorted_results = results.loc[:, ['ind', 'P']].sort_values(by='P', ascending=True)
    to_load = sorted(sorted_results.ind.values[:limit])
    return to_load


def load_from_zarr(zarr_path: str, samples: List[int], gwas_results_path: str, limit: int = 1000) -> Tuple[numpy.ndarray, numpy.ndarray]:
    dataset = zarr.open_group(zarr_path, mode='r')
    
    to_load = set(samples)
    genotype_samples = dataset['samples'][:]
    samples_mask = [int(s) in to_load for s in genotype_samples]
    variants_mask = get_variant_mask(gwas_results_path, limit)
    array = dataset['data'].get_orthogonal_selection((slice(None), variants_mask))

    return array[samples_mask], genotype_samples


def load_from_bed(bed_path: str, samples: List[int], gwas_results_path: str, limit: int = 1000):
    # fam_names = ['fid', 'iid']
    # fam = pandas.read_table(bfile_prefix + '.fam', header=None, names=fam_names, usecols=[0, 1])
    with open_bed(bed_path) as bed:
        to_load = set(samples)
        samples_mask = [int(s) in to_load for s in bed.iid]
        variants_mask = get_variant_mask(gwas_results_path, limit)
        data = bed.read((samples_mask, variants_mask), dtype='int8')
        return data


def bed_to_zarr(bfile_prefix: str, zarr_path: str, batch_size: int):
    with open_bed(bfile_prefix + '.bed') as bed:
        group = zarr.open_group(zarr_path)
        group.create_dataset('samples', data=bed.iid.astype(int), mode='w', overwrite=True)
        group.create_dataset('positions', data=bed.bp_position, mode='w', overwrite=True)
        samples, variants = bed.shape
        array = group.create_dataset('data', shape=(samples, variants), dtype=numpy.int8, chunks=(batch_size, 10000), mode='w', overwrite=True)
        for i in range(0, samples, batch_size):
            for j in range(0, variants, 10000):
                chunk = bed.read(numpy.s_[i:i+batch_size, j:j+10000], dtype='int8', num_threads=4)
                array[i:i+batch_size, j:j+10000] = chunk
                print(f'samples {i}:{i+batch_size}, variants {j}:{j+10000} are read')


def bed_to_bit_zarr(bfile_prefix: str, zarr_path: str, batch_size: int):
    with open_bed(bfile_prefix + '.bed', num_threads=2) as bed:
        group = zarr.open_group(zarr_path)
        group.create_dataset('samples', data=bed.iid.astype(int), mode='w', overwrite=True)
        group.create_dataset('positions', data=bed.bp_position, mode='w', overwrite=True)
        samples, variants = bed.shape
        
        snp_chunk_size = variants // (4*4) + 1
        array = group.create_dataset('data', shape=(samples, variants // 4 + 1),
                dtype=numpy.uint8, chunks=(batch_size, snp_chunk_size), 
                mode='w', overwrite=True)

        for i in range(0, samples, batch_size):
            for j in range(0, variants, snp_chunk_size*4):
                chunk = bed.read(numpy.s_[i:i+batch_size, j:j+snp_chunk_size*4], dtype='int8', num_threads=4)
                to_write = recode_chunk(chunk)
                array[i:i+batch_size, j//4:j//4+snp_chunk_size] = to_write
                print(f'samples {i}:{i+batch_size}, variants {j}:{j+snp_chunk_size*4} are read and recoded')


def bed_to_simple_bit_zarr(bfile_prefix: str, zarr_path: str, batch_size: int):
    with open_bed(bfile_prefix + '.bed', num_threads=2) as bed:
        group = zarr.open_group(zarr_path)
        group.create_dataset('samples', data=bed.iid.astype(int), mode='w', overwrite=True)
        group.create_dataset('positions', data=bed.bp_position, mode='w', overwrite=True)
        samples, variants = bed.shape
        
        snp_chunk_size = variants // (4) + 1
        array = group.create_dataset('data', shape=(samples, snp_chunk_size),
                dtype=numpy.uint8, chunks=(batch_size, snp_chunk_size), 
                mode='w', overwrite=True)

        for i in range(0, samples, batch_size):
            chunk = bed.read(numpy.s_[i:i+batch_size, :], dtype='int8', num_threads=4)
            to_write = recode_chunk(chunk)
            array[i:i+batch_size, :] = to_write
            print(f'samples {i}:{i+batch_size} were read and recoded')


def recode_chunk(chunk):
    chunk[chunk == -127] = 0
    binary_chunk = numpy.zeros((chunk.shape[0], 2*chunk.shape[1]), dtype=numpy.bool8)

    binary_chunk[:, ::2] = chunk > 1
    binary_chunk[:, 1::2] = chunk % 2 == 1
    return numpy.packbits(binary_chunk, axis=1)


def recode_zarr(zarr_path: str, new_zarr_path: str):
    old = zarr.open_group(zarr_path)
    new = zarr.open_group(new_zarr_path)

    new.create_dataset('samples', data=old['samples'][:], mode='w', overwrite=True)
    new.create_dataset('positions', data=old['positions'][:], mode='w', overwrite=True)

    old_data = old['data']
    snp_chunk_size = old_data.shape[1] // (4*4) + 1 # number of workers * number of snps in one byte
    batch_size = old_data.chunks[0]
    array = new.create_dataset('data', shape=(old_data.shape[0], old_data.shape[1] // 4 + 1),
                dtype=numpy.uint8, chunks=(batch_size, snp_chunk_size), 
                mode='w', overwrite=True)

    for i in range(0, old_data.shape[0], batch_size):
        for j in range(0, old_data.shape[1], snp_chunk_size*4):
            chunk = old_data[i:i+batch_size, j:j+snp_chunk_size*4][:]
            to_write = recode_chunk(chunk)
            array[i:i+batch_size, j//4:j//4+snp_chunk_size] = to_write
        
        print(f'batch {i // batch_size} completed          ', end='\r')


if __name__ == '__main__':
    train_bfile = '/media/data1/ag3r/ukb/runs/all/split/train/plink'
    val_bfile = '/media/data1/ag3r/ukb/runs/all/split/val/plink'

    val_zarr = '/media/data1/ag3r/ukb/runs/all/split/val/zarr'
    train_zarr = '/media/data1/ag3r/ukb/runs/all/split/train/zarr'

    recoded_val_zarr = '/media/data1/ag3r/ukb/runs/all/split/val/zarr_bits'
    recoded_train_zarr = '/media/data1/ag3r/ukb/runs/all/split/train/zarr_bits'

    # bed_to_zarr(train_bfile, train_zarr, batch_size=1024)
    recode_zarr(val_zarr, recoded_val_zarr)
    recode_zarr(train_zarr, recoded_train_zarr)
