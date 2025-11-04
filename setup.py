from setuptools import setup, find_packages

setup(
    name="adaptive_lora",
    version="1.0.0",
    description="Dynamic Adaptive LoRA with Block Influence scoring and epoch-wise rank adaptation",
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
