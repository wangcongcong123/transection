import random, os, glob, sys, shutil
import numpy as np
import torch
import logging
import json
import transformers

def add_filehandler_for_logger(output_path, logger, out_name="train"):
    logFormatter = logging.Formatter('%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s')
    fileHandler = logging.FileHandler(os.path.join(output_path, f"{out_name}.log"), mode="a")
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

def set_seed(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

def get_existing_cks(output_path, best_ck=False, return_best_ck=True):
    cks_already = [name for name in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, name))]

    if best_ck:
        for ex in [each for each in cks_already if each.startswith("best")]:
            cks_already.remove(ex)
            shutil.rmtree(os.path.join(output_path, ex))

    index2path = {}

    for each_ck in cks_already:
        if return_best_ck or not each_ck.startswith("best"):
            index2path[int(os.path.basename(each_ck).split("_")[-1])] = os.path.join(output_path, each_ck)

    sorted_indices = sorted(index2path)  # index here refers to the epoch number
    return sorted_indices, index2path

def save_ck(args, logger, model, tokenizer, steps=0, tag="epoch", best_ck=False):
    sorted_indices, index2path = get_existing_cks(args.output_path, best_ck=best_ck)
    if len(sorted_indices) >= args.keep_ck_num:
        logger.info(
            f"there are already {len(sorted_indices)} checkpoints saved that will be more than keep_ck_num={args.keep_ck_num}")
        logger.info(f"hence, remove the oldest one: {index2path[sorted_indices[0]]}")
        shutil.rmtree(
            index2path[sorted_indices[0]])  # remove the oldest checkpoint, i.e., the one with the lowest epoch number

    if best_ck:
        logger.info(
            f'save best model weights and tokenizer to {os.path.join(args.output_path, f"best_ck_at_{tag}_{steps}.h5")}')
        tokenizer.save_pretrained(os.path.join(args.output_path, f"best_ck_at_{tag}_{steps}"))
        if isinstance(model, torch.nn.DataParallel):
            model.module.save_pretrained(os.path.join(args.output_path, f"best_ck_at_{tag}_{steps}"))
        else:
            model.save_pretrained(os.path.join(args.output_path, f"best_ck_at_{tag}_{steps}"))
    else:
        logger.info(
            f'save model weights and tokenizer to {os.path.join(args.output_path, f"ck_at_{tag}_{steps}")}')
        tokenizer.save_pretrained(os.path.join(args.output_path, f"ck_at_{tag}_{steps}"))
        if isinstance(model, torch.nn.DataParallel):
            model.module.save_pretrained(os.path.join(args.output_path, f"ck_at_{tag}_{steps}"))
        else:
            model.save_pretrained(os.path.join(args.output_path, f"ck_at_{tag}_{steps}"))


def save_and_check_if_early_stop(eval_score, args, logger, model, tokenizer, steps=0, tag="epoch"):
    logger.info("\n")
    logger.info(f"*******eval at {tag} = {steps}*********")
    logger.info(f"val_{args.eval_on}: {eval_score}")
    if args.eval_on == "acc":
        if eval_score >= args.best:
            args.wait = 0
            args.best = eval_score
            logger.info(f"so far the best check point at {tag}={steps} based on eval_on {args.eval_on}")
            save_ck(args, logger, model, tokenizer, steps=steps, tag=tag, best_ck=True)
        else:
            args.wait += 1
    else:
        raise ValueError("not support yet")
    logger.info(f"best so far({args.eval_on}): {args.best}")
    logger.info(f"early stop count: {args.wait}/{args.patience}")
    save_ck(args, logger, model, tokenizer, steps=steps, tag=tag, best_ck=False)
    if args.wait >= args.patience:
        logger.info("run out of patience, early stop")
        return True
    return False

def check_output_path(output_path, force=False):
    if os.path.isdir(output_path):
        if force:
            print(f"{output_path} exists, remove it as force=True")
            shutil.rmtree(output_path)
            os.makedirs(output_path, exist_ok=True)
        else:
            out = input(
                "Output directory ({}) already exists and is not empty, you wanna remove it before start training? (y/n)".format(
                    output_path))
            if out.lower() == "y":
                shutil.rmtree(output_path)
                os.makedirs(output_path, exist_ok=True)
            else:
                sys.exit(0)
    else:
        print(f"{output_path} not found, create it now")
        os.makedirs(output_path, exist_ok=True)

def get_scheduler(optimizer, scheduler: str, warmup_steps: int, num_total: int):
    assert scheduler in ["constantlr", "warmuplinear", "warmupconstant", "warmupcosine",
                         "warmupcosinewithhardrestarts"], (
        'scheduler should be one of ["constantlr","warmupconstant","warmupcosine","warmupcosinewithhardrestarts"]')
    if scheduler == 'constantlr':
        return transformers.get_constant_schedule(optimizer)
    elif scheduler == 'warmupconstant':
        return transformers.get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
    elif scheduler == 'warmuplinear':
        return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                            num_training_steps=num_total)
    elif scheduler == 'warmupcosine':
        return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                            num_training_steps=num_total)
    elif scheduler == 'warmupcosinewithhardrestarts':
        return transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,
                                                                               num_warmup_steps=warmup_steps,
                                                                               num_training_steps=num_total)

def get_optimizer(args, optim_groups):
    args.optimizer = args.__dict__.pop("optimizer", "adamw").lower()
    assert args.optimizer in ["adamw", "adam", "adagrad", "adadelta", "sgd"], (
        'optimizer now only supports ["adamw", "adam", "adagrad", "adadelta", "sgd"]')

    if args.optimizer == 'adam':
        args.adam_eps = args.__dict__.pop("adam_eps", 1e-8)
        args.adam_betas = args.__dict__.pop("adam_betas", (0.9, 0.95))
        return torch.optim.Adam(optim_groups, lr=args.lr, eps=args.adam_eps, betas=args.adam_betas)
    elif args.optimizer == 'adagrad':
        args.adagrad_lr_decay = args.__dict__.pop("adagrad_lr_decay", 0)
        args.adagrad_eps = args.__dict__.pop("adagrad_eps", 1e-10)
        return torch.optim.Adagrad(optim_groups, lr=args.lr, lr_decay=args.adagrad_lr_decay,
                                   eps=args.__dict__.pop("adagrad_eps", 1e-10))
    elif args.optimizer == 'adadelta':
        args.adadelta_eps = args.__dict__.pop("adadelta_eps", 1e-10)
        return torch.optim.Adadelta(optim_groups, lr=args.lr, eps=args.adadelta_eps)
    elif args.optimizer == 'sgd':
        args.sgd_momentum = args.__dict__.pop("sgd_momentum", 0)
        return torch.optim.SGD(optim_groups, lr=args.lr, momentum=args.sgd_momentum)
    else:
        args.adamw_eps = args.__dict__.pop("adamw_eps", 1e-8)
        args.adamw_betas = args.__dict__.pop("adamw_betas", (0.9, 0.95))
        return torch.optim.AdamW(optim_groups, lr=args.lr, eps=args.adamw_eps, betas=args.adamw_betas)

def write_args(args):
    with open(os.path.join(args.output_path, "args.json"), "w") as f:
        print(json.dumps(args.__dict__, indent=2))
        f.write(json.dumps(args.__dict__, indent=2))


def print_model_state_dict(model, logger):
    for param_tensor in model.state_dict():
        logger.info(f"{param_tensor}\t{model.state_dict()[param_tensor].size()}")


def count_params(model, logger, print_details=False):
    trainable_count = 0
    total_count = 0
    if isinstance(model, torch.nn.Sequential):
        for index in model._modules:
            if print_details:
                print_model_state_dict(model._modules[index], logger)
                logger.info(model._modules[index])
            trainable_count += sum(p.numel() for p in model._modules[index].parameters() if p.requires_grad)
            total_count += sum(p.numel() for p in model._modules[index].parameters())
    else:
        if print_details:
            print_model_state_dict(model, logger)
            logger.info(model)
        total_count = sum(p.numel() for p in model.parameters())
        trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'  Model Total params: {total_count}')
    logger.info(f'  Model Trainable params: {trainable_count}')
    logger.info(f'  Model Non-trainable params: {total_count - trainable_count}')
