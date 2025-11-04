from setuptools import setup, find_packages
setup(
    name="adaptive_lora",
    version="2.0.0",
    description="Adaptive LoRA (Option B) - reinitialize LoRA adapters each epoch based on BI scores.",
    packages=find_packages(),
    install_requires=[
        "torch>=1.13",
        "transformers>=4.30",
        "peft>=0.3.0",
        "datasets>=2.0",
        "tqdm",
        "numpy",
        "scikit-learn",
        "accelerate"
    ],
)
