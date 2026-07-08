from setuptools import setup, find_packages
from pathlib import Path

# Automatically use README as the long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text() if (this_directory / "README.md").exists() else ""

setup(
    name='scTopoDEC',
    version='0.1.0',
    packages=find_packages(), # Automatically finds 'scTopoDEC' folder
    install_requires=[
        'numpy',
        'keras>=3.0',
        'tensorflow',
        'scanpy',
        'scikit-learn',
        'h5py',      
        'anndata',   
        'optuna',     
        'tqdm'        
    ],
    entry_points={
        'console_scripts': [
            'scTopoDEC = scTopoDEC.__main__:main'
        ]
    },
    url='https://github.com/AugustusKH/scTopoDEC',
    author='Pakorn Sagulkoo',
    author_email='pakorn.sagulkoo@cmu.ac.th',
    description='Topological Deep Embedded Clustering for single-cell data',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='MIT',
    keywords=['single-cell', 'clustering', 'topology', 'persistent-homology', 'autoencoder'],
    classifiers=[
        'License :: OSI Approved :: MIT License', 
        'Topic :: Scientific/Engineering :: Bio-Informatics', 
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
    ]
)