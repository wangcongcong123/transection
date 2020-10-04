import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Hyper params')

    parser.add_argument('--model_select', type=str, default="t5-small",
                        help='model select from MODEL_MAP')

    parser.add_argument('--task', type=str, default="default",
                        help='tasks available in TASKS_SUPPORT')

    parser.add_argument('--per_device_train_batch_size', type=int, default=8,
                        help='input batch size for training')

    parser.add_argument('--eval_batch_size', type=int, default=32,
                        help='input batch size for training')

    parser.add_argument('--num_epochs_train', type=int, default=6,
                        help='number of epochs to train')

    parser.add_argument('--log_steps', type=int, default=400,
                        help='logging steps for evaluation based on global step if it is not -1 and based on epoch if it is -1, and tracking metrics using tensorboard if use_tb is active')

    parser.add_argument('--max_seq_length', type=int, default=128,
                        help='maximum sequence length of samples in a batch for training')

    parser.add_argument('--max_src_length', type=int, default=128,
                        help='only working for t5-like t2t-based tasks, maximum source sequence length of samples in a batch for training')

    parser.add_argument('--max_tgt_length', type=int, default=10,
                        help='only working for t5-like t2t-based tasks, maximum target sequence length of samples in a batch for training')

    parser.add_argument('--source_field_name', type=str, default="text",
                        help='only working for t5-like t2t-based tasks, the source field name of the provided jsonl-formatted data')

    parser.add_argument('--target_field_name', type=str, default="label",
                        help='only working for t5-like t2t-based tasks, the target field name of the provided jsonl-formatted data')

    parser.add_argument('--lr', type=float, default=0.001,
                        help='default learning rate for fine-tuning as described in the T5 paper')

    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='warmup_ratio, only working if scheduler is not constant')

    parser.add_argument('--patience', type=int, default=20,
                        help='patience based on the log_steps')

    parser.add_argument('--scheduler', type=str, default="constant",
                        help='scheduler, default is constant as described in the T5 paper')

    parser.add_argument('--seed', type=int, default=122,
                        help='random seed')

    parser.add_argument('--eval_on', type=str, default="acc",
                        help='eval on for best ck saving and patience-based training early stop')

    parser.add_argument('--keep_ck_num', type=int, default=3,
                        help='keep_ck_num except for the best ck (evaluated on validation set using the metric specified by --eval_on')

    parser.add_argument('--ck_index_select', type=int, default=0,
                        help='ck_index_select, use the best one by default, negative one to specify a latest one, working when --do_test is active')

    parser.add_argument(
        "--do_train", action="store_true", help="Do training"
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="do evaluation on validation set for saving checkpoint"
    )
    parser.add_argument(
        "--do_test", action="store_true", help="eval on test set if test set is availale"
    )
    parser.add_argument(
        "--use_tb", action="store_true", help="use tensorboard for tracking training process, default save to ./runs"
    )
    args = parser.parse_args()
    return args