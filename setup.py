from setuptools import setup, find_packages

setup(
    name="adaptive_lora",
    version="1.0.3",
    description="Dynamic Adaptive LoRA with forced BI printing (on_train_end hook)",
    packages=find_packages(),
    install_requires=[
        "torch>=1.13",
        "transformers>=4.30",
        "peft>=0.3.0",
        "datasets>=2.0",
        "tqdm",
        "scikit-learn",
        "accelerate"
    ],
)
