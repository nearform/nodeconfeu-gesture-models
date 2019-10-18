from setuptools import setup

setup(name='nodeconfeu_watch',
      version='0.1.0',
      description='Implementation of NALU with stable training',
      url='https://github.com/nearform/nodeconfeu-gesture-models',
      author='Andreas Madsen',
      author_email='andreas.madsen@nearform.com',
      license='Apache License 2.0',
      packages=['nodeconfeu_watch'],
      install_requires=[
          'numpy',
          'pandas',
          'seaborn',
          'tensorflow',
          'matplotlib',
          'flatbuffers'
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      include_package_data=True,
      zip_safe=False)
