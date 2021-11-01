from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
#long_description = (here / 'README.md').read_text(encoding='utf-8')
with (here /'requirements.txt').open() as fp:
    install_requires = fp.read()
    if '-i http' in install_requires:
        install_requires = install_requires.replace('-i https://pypi.org/simple', '')

setup(
    name='puffin',
    version='0.1.0',
    packages=find_packages(include=['puffin', 'puffin.*']),
    python_requires='>=3.9',

    # This field lists other packages that your project depends on to run.
    # Any package you put here will be installed by pip when your project is
    # installed, so they must be valid existing projects.
    #
    # For an analysis of "install_requires" vs pip's requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=install_requires,  # Optional

)