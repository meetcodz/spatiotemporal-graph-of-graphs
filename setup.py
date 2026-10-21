from setuptools import setup, find_packages

setup(
    name="stgog",
    version="1.0.0",
    description="A Spatiotemporal Graph-of-Graphs Framework for Multivariate Time Series Classification",
    author="Tushar Sharma, ABV-IIITM Gwalior",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.22.0",
        "scipy>=1.8.0",
        "scikit-learn>=1.0.0",
        "pyyaml>=6.0",
        "matplotlib>=3.5.0",
        "tqdm>=4.64.0",
    ],
    extras_require={
        "benchmarks": ["sktime>=0.28.0"],
        "dev": ["pytest>=7.0.0", "black>=23.0.0", "flake8>=5.0.0"],
        "report": ["reportlab>=3.6.0", "pymupdf>=1.23.0"],
    },
    entry_points={
        "console_scripts": [
            "stgog-train=scripts.train:main",
            "stgog-eval=scripts.evaluate:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
