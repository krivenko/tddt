from setuptools import setup, find_packages

setup(
    name='tddt',
    version='0.1.0',
    url='https://github.com/krivenko/tddt.git',
    author='Viktor Valmispild, Igor Krivenko',
    author_email='valmispild@gmail.com, igor.s.krivenko@gmail.com',
    description='Implementation of the time-dependent dual TRILEX theory',
    packages=['tddt'],
    install_requires=['numpy >= 1.12.0', 'mpmath >= 1.2.0'],
)
