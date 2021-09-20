from os import name
from numpy import load
from .bed import load_from_bed, recode_zarr
import pandas
import numpy
from .load import BinaryICDLoader
import zarr
import pytest


@pytest.mark.skip(reason="Requires gwas results")
def test_load_from_bed():
    bfile_prefix = '/media/data1/ag3r/ukb/runs/all/split/train'

    fam = f'{bfile_prefix}/plink.fam'
    bed = f'{bfile_prefix}/plink.bed'
    names = ['fid', 'iid', '2', '3', '4', '5']
    samples_frame = pandas.read_csv(fam, names=names, header=None, sep='\s+')
    samples = samples_frame.iloc[:50, 0].tolist()

    gwas_path = '/media/data1/ag3r/ukb/runs/gwas/diabetes_e119/plink2.PHENO1.glm.logistic.hybrid'
    
    data = load_from_bed(bed, samples, gwas_path, limit=100)
    assert data.shape == (50, 100)
    unique = numpy.unique(data)
    assert all(unique == numpy.array([-127, 0, 1, 2]))


@pytest.mark.skip(reason="Requires old zarr recoding")
def test_recode_zarr():
    # val_zarr = '/media/data1/ag3r/ukb/runs/all/split/val/zarr'
    test_zarr_path = '/media/data1/ag3r/test/recode_zarr_old'
    group = zarr.open_group(test_zarr_path, mode='w')
    group.create_dataset('samples', data=numpy.array([1, 2]))
    group.create_dataset('positions', data=numpy.arange(15))
    data = numpy.random.randint(0, 4, size=(2, 15), dtype=numpy.uint8)
    group.create_dataset('data', data=data, dtype=numpy.uint8)
    new_zarr_path = '/media/data1/ag3r/test/recode_zarr_new'

    recode_zarr(test_zarr_path, new_zarr_path)

    new_group = zarr.open_group(new_zarr_path, mode='r')
    print(new_group.tree())
    print(group.tree())
    new_data = new_group['data'][:]
    new_data = numpy.unpackbits(new_data, axis=1)
    new_data = new_data[:, ::2]*2 + new_data[:, 1::2]
    assert (data == new_data[:, :15]).all()


@pytest.mark.skip(reason="Requires old zarr recoding")
def test_recoded_zarr_benchmark():
    test_zarr_path = '/media/data1/ag3r/test/recode_zarr_old'
    group = zarr.open_group(test_zarr_path, mode='w')
    samples_count = 1024*10
    variants_count = 1024*32
    group.create_dataset('samples', data=numpy.arange(samples_count))
    group.create_dataset('positions', data=numpy.arange(variants_count))
    data = numpy.random.randint(0, 4, size=(samples_count, variants_count), dtype=numpy.uint8)
    group.create_dataset('data', data=data, dtype=numpy.uint8)
    new_zarr_path = '/media/data1/ag3r/test/recode_zarr_new'

    recode_zarr(test_zarr_path, new_zarr_path)

    new_group = zarr.open_group(new_zarr_path, mode='r')
    print(new_group.tree())
    print(group.tree())
    new_data = new_group['data'][:]
    new_data = numpy.unpackbits(new_data, axis=1)
    new_data = new_data[:, ::2]*2 + new_data[:, 1::2]
    assert (data == new_data[:, :15]).all()
