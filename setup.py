"""
setup.py
--------
Minimal packaging configuration for the Biosignal Stress Classification Pipeline.

Install in editable mode for development:
    pip install -e .

This exposes a single CLI entry point:
    python -m PipelinePython <data|peaks|stress-onnx|all>
"""

from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", encoding="utf-8") as fh:
    install_requires = [
        line.strip() for line in fh if line.strip() and not line.startswith("#")
    ]

setup(
    name="biosignal-pipeline",
    version="3.0.0",
    author="HRV Stress Classification Research Team",
    description=(
        "Reproducible pipeline for biosignal wrangling, PPG/ECG validation, "
        "stress classification, and Android edge deployment."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.12",
    packages=find_packages(include=["PipelinePython", "PipelinePython.*"]),
    install_requires=install_requires,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "hypothesis>=6.82.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "biosignal-pipeline=PipelinePython.__main__:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
