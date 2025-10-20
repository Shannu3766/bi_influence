from setuptools import setup, find_packages

setup(
    name="adalora_bi",
    version="0.1.0",
    description="BI-based Adaptive LoRA Rank Allocation (Algorithm 2) with dynamic per-epoch reallocation",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "datasets",
        "numpy"
    ],
    author="Your Name",
)
