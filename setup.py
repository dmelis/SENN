from setuptools import setup, find_packages

setup(name='SENN',
      version='0.1',
      description='Self Explaining Neural Networks',
      url='https://github.mit.edu/dalvmel/SENN',
      author='David Alvarez Melis (MIT)',
      author_email='dalvmel@mit.edu',
      license='MIT',
      packages=find_packages(),#'src'),
      #package_dir = {'': 'src'},
      include_package_data=True,
      install_requires=[
            'scipy',
            'numpy',
            'matplotlib',
            'scikit-image',
            'tqdm',
            'attrdict'
      ],
      extras_require={
            "tf": ["tensorflow>=1.0.0"],
            "tf_gpu": ["tensorflow-gpu>=1.0.0"],
      },
      zip_safe=False,
      test_suite='nose.collector',
      tests_require=['nose'],
      )
