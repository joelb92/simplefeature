from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(name='simplefeature',
      version='1.0',
      description='',
      long_description=long_description,
      author='D.D. Cox, and N. Pinto, Joel Brogan',
      url='https://github.com/joelb92/simplefeature',
      packages=['simplefeature'],
      package_dir={'simplefeature': './'}
     )

