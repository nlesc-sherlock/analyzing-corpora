#!/usr/bin/env python
# SIM-CITY client
# 
# Copyright 2015 Netherlands eScience Center <info@esciencecenter.nl>
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


setup(name='corpora',
      version='0.1',
      description='Topic models for a corpus of text.',
      author='Joris Borgdorff',
      author_email='j.borgdorff@esciencecenter.nl',
      url='https://github.com/nlesc-sherlock/analyzing-corpora',
      packages=['corpora'],
      classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Environment :: Console',
        'Development Status :: 2 - Pre-Alpha',
        'Natural Language :: English',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Text Processing :: Linguistic',
        'Intended Audience :: Science/Research',
      ],
      install_requires=['gensim', 'scikit-learn', 'nltk', 'wordcloud', 'matplotlib', 'pandas', 'pattern', 'numpy', 'progressbar', 'joblib'],
      tests_require=['nose', 'pyflakes', 'pep8', 'coverage']
     )
