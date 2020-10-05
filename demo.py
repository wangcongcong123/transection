# pip install transformers
import torch
from transformers import BartTokenizer, BartForConditionalGeneration,AutoTokenizer, AutoModelWithLMHead
model_name_or_path = "tmp/pt_bart-base_translation/ck_at_epoch_1"
device = "cuda" if torch.cuda.is_available() else "cpu"
examples = ["Truth, good and beauty have always been considered as the three top pursuits of human beings",
            "Warm wind caresses my face",
            "Sang Lan is one of the best athletes in our country."]

# PyTorch
model = BartForConditionalGeneration.from_pretrained(model_name_or_path)
# or Tensorflow 2.0
# model = BartForConditionalGeneration.from_pretrained(model_name_or_path)
tokenizer = BartTokenizer.from_pretrained(model_name_or_path)
# Batch size 2. change "pt" to "tf" if using Tensorflow 2.0
inputs = tokenizer(examples,padding=True, return_tensors="pt").to(device)
model.eval().to(device)
outputs = model.generate(**inputs,max_length=128)
print("customized model outputs:")
print([tokenizer.decode(ids,skip_special_tokens=True) for ids in outputs])

# as a comparison, let's try the same examples on "Helsinki-NLP/opus-mt-en-zh"
model_name_or_path="Helsinki-NLP/opus-mt-en-zh"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelWithLMHead.from_pretrained(model_name_or_path)
inputs = tokenizer(examples,padding=True, return_tensors="pt").to(device)
model.eval().to(device)
outputs = model.generate(**inputs,max_length=128)
print("Helsinki-NLP/opus-mt-en-zh model outputs:")
print([tokenizer.decode(ids,skip_special_tokens=True) for ids in outputs])
