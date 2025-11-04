from setuptools import setup, find_packages

setup(
    name="adaptive_lora",
    version="0.2.0",
    description="Dynamic Adaptive LoRA fine-tuning with per-epoch rank adaptation",
    packages=find_packages(),
    install_requires=[
        "torch>=1.13",
        "transformers>=4.30",
        "peft>=0.3.0",
        "datasets>=2.0",
        "tqdm",
        "scikit-learn",
        "matplotlib",
        "accelerate"
    ],
)
