from datasets import load_dataset, load_from_disk, HF_DATASETS_CACHE
from transformers import AutoTokenizer, AutoModelWithLMHead, AutoConfig

from ptt.utils import *
from ptt.trainer import train
from ptt.args import get_args

# we use transformers logger here so we can log message to a file locally
transformers.logging.set_verbosity_info()
logger = transformers.logging.get_logger()


def convert_to_features(example_batch):
    encoded_source = tokenizer(example_batch["source"], padding=True, truncation=True,
                               max_length=args.max_src_length)
    encoded_target = tokenizer(example_batch["target"], padding=True, truncation=True,
                               max_length=args.max_tgt_length)
    encoded_source.update(
        {"labels": encoded_target["input_ids"], "decoder_attention_mask": encoded_target["attention_mask"]})
    return encoded_source


def collate_fn(examples):
    source_inputs = [{"input_ids": each["input_ids"], "attention_mask": each["attention_mask"]} for each in
                     examples]
    target_inputs = [{"input_ids": each["labels"], "attention_mask": each["decoder_attention_mask"]} for each in
                     examples]
    source_inputs_padded = tokenizer.pad(source_inputs, return_tensors='pt')
    target_inputs_padded = tokenizer.pad(target_inputs, return_tensors='pt')
    source_inputs_padded.update({"labels": target_inputs_padded["input_ids"],
                                 "decoder_attention_mask": target_inputs_padded["attention_mask"]})
    return source_inputs_padded


if __name__ == '__main__':
    args = get_args()
    ############### customize args
    args.dataset_name = "translation"
    args.model_select = "facebook/bart-base"
    args.output_path = os.path.join("tmp", args.model_select.split("/")[-1] + "_" + args.dataset_name)
    args.n_gpu = 1
    args.visible_devices = "0"  # sperated by ","
    args.from_pretrained = True
    args.num_epochs_train = 24
    args.batch_size = 8
    args.log_steps = 5000
    args.lr = 2e-4
    args.max_src_length = 128
    args.max_tgt_length = 128
    args.optimizer = "adamw"
    args.scheduler = "constantlr" # or warmuplinear?
    args.grad_norm_clip = 1.0
    args.do_train = True
    args.do_eval = False
    args.use_tb = True
    ##################
    logger.info(f"   see seed for random, numpy and torch {args.seed}")
    set_seed(args.seed, torch.cuda.device_count())

    if args.do_train:
        check_output_path(args.output_path, force=False)

    add_filehandler_for_logger(args.output_path, logger)
    tokenizer = AutoTokenizer.from_pretrained(args.model_select)

    if not os.path.isdir(f".cache/{args.output_path}"):
        # program gets stuck when loading dataset if not having the following operation (a weird problem both on my windows desktop and my friend's linux server)
        pyarrow_path = os.path.join(HF_DATASETS_CACHE, args.dataset_name, "0.0.0")
        if not os.path.isdir(pyarrow_path):
            os.makedirs(pyarrow_path, exist_ok=True)

        dataset = load_dataset(f"data_scripts/{args.dataset_name}.py")
        encoded = dataset.map(convert_to_features, batched=True)
        encoded.save_to_disk(f".cache/{args.output_path}")
    else:
        encoded = load_from_disk(f".cache/{args.output_path}")

    columns = ['input_ids', 'attention_mask', 'labels', 'decoder_attention_mask']
    encoded.set_format(type='torch', columns=columns)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.do_train:
        if args.from_pretrained:
            model = AutoModelWithLMHead.from_pretrained(args.model_select)
        else:
            config = AutoConfig.from_pretrained(args.model_select)
            model = AutoModelWithLMHead.from_config(config)
        train(args, logger, model, tokenizer, device, encoded["train"], collate_fn=collate_fn, verbose=True)
