from setuptools import setup, find_packages

setup(
    name="fv-methods",
    version="0.1.0",
    description="""Comparison of slope limiting methods for high order finte volume
    schemes.""",
    author="Jonathan Palafoutas",
    author_email="jpalafou@princeton.edu",
    packages=find_packages(),
    install_requires=["numpy", "matplotlib"],
)
