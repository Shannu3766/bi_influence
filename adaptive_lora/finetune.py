from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments
from .datasets import get_dataloader
def finetune_dynamic(*args, **kwargs):
    raise RuntimeError('Use Trainer + AdaptiveLoRACallback for dynamic epochwise adaptation (Option B).')
