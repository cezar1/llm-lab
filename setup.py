from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="llm-lab",
    version="1.0.0",
    author="LLM Learning Lab",
    description="Hands-on curriculum for mastering LLM concepts through implementation, visualization, and ablation studies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cezar1/llm-lab",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Education",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Education",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
        "matplotlib>=3.5.0",
        "scikit-learn>=1.0.0",
    ],
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
        "all": [
            "jupyter>=1.0.0",
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [],
    },
    keywords=[
        "llm",
        "language-model",
        "transformer",
        "attention",
        "tokenization",
        "education",
        "deep-learning",
        "pytorch",
    ],
    project_urls={
        "Bug Reports": "https://github.com/cezar1/llm-lab/issues",
        "Source": "https://github.com/cezar1/llm-lab",
        "Documentation": "https://github.com/cezar1/llm-lab#readme",
    },
    include_package_data=True,
    zip_safe=False,
)
