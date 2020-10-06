"""
Helsinki-NLP/opus-mt-en-zh is trained on 41,649,946 english-chinese pairs, see: https://github.com/Helsinki-NLP/Tatoeba-Challenge/blob/master/Data.md
And mode card: https://huggingface.co/Helsinki-NLP/opus-mt-en-zh
This script is used to evaluate the pre-trained "Helsinki-NLP/opus-mt-en-zh" model and compare its sacrebleu score with
 a customized en2zh model that was fine-tuned on 5,161,434 english-chinese pairs from "bart-base" pre-trained weights.
 Although "Helsinki-NLP/opus-mt-en-zh" is half of the customized model in size (~77M vs ~139M), it has 8 times training examples as the customized model
Copyright: wangcongcong123
"""
import argparse
import torch, os, sacrebleu
from transformers import AutoTokenizer, AutoModelWithLMHead
from datasets import load_dataset, load_metric, HF_DATASETS_CACHE
from ptt.utils import count_params
import transformers

transformers.logging.set_verbosity_info()
logger = transformers.logging.get_logger()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyper params')
    parser.add_argument('--model_path', type=str, default="tmp/pt_bart-base_translation/ck_at_epoch_1",
                        help='model select for evaluation, congcongwang/bart-base-en-zh or Helsinki-NLP/opus-mt-en-zh')
    args = parser.parse_args()

    # prepare model and tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelWithLMHead.from_pretrained(args.model_path)
    model.to(device)

    count_params(model, logger, print_details=True)

    dataset = load_dataset("data_scripts/translation.py")
    train, validation = dataset["train"], dataset["validation"]

    evaldata = validation
    # prepare evaluation metric
    metric = load_metric("sacrebleu")
    # to be successfully has to be sacrebleu-1.4.14
    refs = [[]]
    sources = []
    references = []
    predictions = []


    def eval_in_batch(examples):
        input_ids = tokenizer(examples["source"], padding=True, truncation=True, return_tensors="pt")["input_ids"].to(
            device)
        outputs = model.generate(input_ids, max_length=128)
        # since these evaluation examples are usually short sequences, we simoly use greedy decoding
        sys_batch = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outputs]
        reference_batch = [each.split("<sep>") for each in examples["target"]]
        references.extend(reference_batch)
        predictions.extend(sys_batch)
        # to prepare refs for raw sacrebleu
        for each in examples["target"]:
            ref = each.split("<sep>")
            refs[0].append(ref[0])
            if len(ref) > 1:
                if len(refs) == 1:
                    refs.append([])
                refs[1].append(ref[1])
        sources.extend(examples["source"])


    evaldata.map(eval_in_batch, batched=True, batch_size=32)
    hf_bleu = metric.compute(predictions=predictions, references=references, tokenize="zh")  # this is chinese
    bleu = sacrebleu.corpus_bleu(predictions, refs, tokenize="zh")  # this is chinese
    assert hf_bleu["score"] == bleu.score  # the two scores should be the same
    print(bleu.score)
