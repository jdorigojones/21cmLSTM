from setuptools import setup, find_packages

setup(
  name='21cmLSTM',
  version='1.0.0',
  url='https://github.com/johnnydorigojones/21cmLSTM',
  author='Johnny Dorigo Jones',
  author_email='johnny.dorigojones@colorado.edu',
  description='21cmLSTM - a memory-based emulator of the 21-cm global signal with unprecedented accuracy and speed',
  packages=find_packages(),    
  install_requires=open('requirements.txt').read().splitlines(),
  license='MIT',
  classifiers=['Intended Audience :: Science/Research',
               'Natural Language :: English',
               'License :: OSI Approved :: MIT License',
               'Programming Language :: Python :: 3',
               'Topic :: Scientific/Engineering :: Astronomy',
               'Topic :: Scientific/Engineering :: Physics'],
)