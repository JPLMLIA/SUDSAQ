import setuptools

setuptools.setup(
    name        = 'sudsaq',
    version     = '0.0.1',
    description = 'Package containing libraries and scripts for the SUDS AQ project',
    packages    = setuptools.find_packages(),
    classifiers = [
        'Programming Language :: Python :: 3'
    ],
    install_requires = [
        'dask',
        'h5py>=2.10.0',
        'matplotlib>=3.3.0',
        'numpy>=1.19.1',
        'pandas>=0.23.4',
        'pyyaml>=5.4.1',
        'ray',
        'requests>=2.24.0',
        'scikit-learn>=0.22',
        'seaborn>=0.11.0',
        'shap',
        'tables',
        'tqdm>=4.59.0',
        'xarray'
    ],
    python_requires = '~=3.7',
)
