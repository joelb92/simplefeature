from setuptools import setup,find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(name='simplefeature',
      version='1.1.2',
      description='',
      long_description=long_description,
      author='D.D. Cox, and N. Pinto, Joel Brogan',
      author_email="joelbrogan92@gmail.com",
      url='https://github.com/joelb92/simplefeature',
      # packages=['simplefeature'],
      package_dir={'simplefeature': './'},
      long_description_content_type="text/markdown",
      packages=find_packages(),
      classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
     ],
      install_requires=[
          'opencv-python',
          'numpy',
          'scikit-image',
      ],
     )

