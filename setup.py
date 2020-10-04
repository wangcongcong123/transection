from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

setup(
    name="ptt",
    version="0.0.0",
    author="Congcong Wang",
    author_email="wangcongcongcc@gmail.com",
    description="Fine-tuning Sequence-to-sequence Transformers for English to Chinese Translation",
    long_description=readme,
    long_description_content_type="text/markdown",
    license="MIT License",
    url="https://github.com/wangcongcong123/transection", # not released yet
    download_url="...",
    packages=find_packages(),
    install_requires=[
        "tqdm",
        "sacrebleu",
        "datasets",
        "transformers==3.1.0"
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    keywords="Transformers, BART, Translation, PyTorch, GPUs acceleration"
)

# pip install -e .
# commands for uploading to pypi
# python setup.py sdist
# pip install twine
# twine upload dist/*
