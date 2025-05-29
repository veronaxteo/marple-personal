from setuptools import find_packages, setup

package_dir = {"": "code"}

setup(
    name='src',
    version='0.1.0',
    package_dir=package_dir,
    packages=find_packages(where="code"),
    python_requires='>=3.8',
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'matplotlib',
        'pytest',
        'igraph',
        'joblib',
        'networkx',
        'tqdm'
    ]
)
