# Copyright 2021 Alexey Tochin
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
# ==============================================================================
from setuptools import setup
import pathlib

import tf_dataclass

here = pathlib.Path(__file__).parent.resolve()

setup(
    name='tf-dataclass',
    version=tf_dataclass.__version__,
    description='Dataclasses for TensorFlow',
    long_description=(here / 'README.md').read_text(encoding='utf-8'),
    long_description_content_type='text/markdown',
    url='https://github.com/alexeytochin/tf-dataclass',
    author='Alexey Tochin',
    author_email='alexey.tochin@gmail.com',
    license='Apache 2.0',
    license_files=('LICENSE',),
    packages=['tf_dataclass'],
    install_requires=['tensorflow>=2.0.0'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    keywords='tensorflow, dataclass',
)