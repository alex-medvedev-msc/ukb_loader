# UK Biobank data loader

This repository provides a library and set of utilities for the efficient loading of phenotype and genotype data from the [UK Biobank](https://www.ukbiobank.ac.uk/).

Features include:
* Loading quantitative and categorical phenotypes, includeding self-reported phenotypes and phenotypes based on [ICD-10 disease codes](https://biobank.ndph.ox.ac.uk/ukb/field.cgi?id=41202).
* Fast parallelized loading that leverages chunked and compressed [Zarr arrays](https://zarr.readthedocs.io/en/stable/).
* Utilities for splitting the dataset samples randomly, or based on a predefined structure.

## Usage

First, the UKB dataset needs to be converted into the Zarr format with the desired test/train/validation split. For this, use the provided [conversion script](src/ukb_loader/convert_all.py).

For examples on loading various types of phenotypes, see [this example notebook](examples/load-phenotype-example.ipynb).