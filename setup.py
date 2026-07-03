"""pip install -e . entry point."""
from setuptools import setup, find_packages

setup(
    name="cortex-filter",
    version="2.0.0",
    author="Shweta Mishra",
    author_email="shweta.mishra.research@gmail.com",
    description="Pre-execution repository filtering for LLM-based developer tools",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Shweta-Mishra-ai/CORTEX-",
    package_dir={"cortex": "src/python"},
    packages=["cortex", "cortex.prompt", "cortex.swe", "cortex.quality"],
    python_requires=">=3.9",
    install_requires=[],
    extras_require={
        "dev": ["pytest>=7.4", "pytest-cov>=4.1", "ruff>=0.4"],
        "experiments": [
            "matplotlib>=3.7", "numpy>=1.24", "pandas>=2.0",
            "scipy>=1.11", "tiktoken>=0.5", "requests>=2.31",
            "datasets>=2.14", "openai>=1.0",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={"console_scripts": ["cortex=cortex.filter:main"]},
)
