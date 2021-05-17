from setuptools import setup, find_packages

setup(
  name = 'En-transformer',
  packages = find_packages(),
  version = '0.2.8',
  license='MIT',
  description = 'E(n)-Equivariant Transformer',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/En-transformer',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'equivariance',
    'transformer'
  ],
  install_requires=[
    'einops>=0.3',
    'torch>=1.7'
  ],
  setup_requires=[
    'pytest-runner',
  ],
  tests_require=[
    'pytest'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)