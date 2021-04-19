try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

from os import path
this_directory = path.abspath(path.dirname(__file__))

with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

requires = [
            'jax',
            'jaxlib',
            'chex'
            ]

setup(
     name='gymnax',
     version='0.0.1',
     author="Robert Tjarko Lange",
     author_email="robertlange0@gmail.com",
     description="gymnax - A JAX-compatible version of Open AI's gym",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/RobertTLange/gymnax",
     #download_url="https://github.com/RobertTLange/gymnax/archive/v_01.tar.gz",
     classifiers=[
         "Programming Language :: Python :: 3.6",
         "Programming Language :: Python :: 3.7",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent"],
     packages=find_packages(),
     include_package_data=True,
     zip_safe=False,
     platforms='any',
     install_requires=requires
 )
