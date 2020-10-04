import sys
import torch, os
from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np
from ptt.utils import save_ck, get_scheduler, get_optimizer, write_args, count_params

def train(args, logger, model, tokenizer, device, train_dataset, collate_fn=None, val_dataset=None, evaluate=None,
          verbose=False):
    if args.do_eval:
        assert val_dataset is not None and evaluate is not None
    args.best = np.Inf if args.eval_on == "loss" else - np.Inf

    if hasattr(args, "n_gpu"):
        assert args.n_gpu <= torch.cuda.device_count()
    else:
        args.n_gpu = torch.cuda.device_count()

    assert args.n_gpu >= 0 and args.n_gpu <= torch.cuda.device_count()

    if hasattr(args, "visible_devices"):
        assert len(args.visible_devices.split(",")) == args.n_gpu
    else:
        logger.info("not specifying --visible_devices, so set it to 0 and using n_gpu=1")
        args.visible_devices = "0"
        args.n_gpu = 1

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=[int(i) for i in args.visible_devices.split(',')])
        model.train().to(f'cuda:{model.device_ids[0]}')
    elif args.n_gpu == 1:
        model.train().to(device)
    else:
        raise ValueError("not support for CPU training")
    if args.use_tb:
        tb_writer = SummaryWriter(log_dir=os.path.join("runs", args.output_path.split(os.sep)[-1]))

    no_decay = ["bias", 'LayerNorm.bias', "LayerNorm.weight"]
    params_decay = [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]
    params_nodecay = [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)]
    args.weight_decay=args.__dict__.pop("weight_decay", 0.1)
    optim_groups = [
        {"params": params_decay, "weight_decay":args.weight_decay},
        {"params": params_nodecay, "weight_decay": 0.0},
    ]

    optimizer = get_optimizer(args, optim_groups)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, collate_fn=collate_fn,
                                                   batch_size=args.batch_size * args.n_gpu)
    if args.do_eval:
        args.dev_batch_size = args.__dict__.pop("dev_batch_size",args.batch_size * 3)
        logger.info(f"batch size for evaluation during training: {args.batch_size * 3}")
        val_dataloader = torch.utils.data.DataLoader(val_dataset, collate_fn=collate_fn, shuffle=False,
                                                     batch_size=args.dev_batch_size)

    args.warmup_ratio = args.__dict__.pop("warmup_steps", 0.1)
    scheduler = get_scheduler(optimizer, scheduler=args.__dict__.pop("scheduler", "constantlr").lower(),
                              warmup_steps=int(args.warmup_ratio * len(train_dataloader)),
                              num_total=args.num_epochs_train * len(train_dataloader))
    args.grad_norm_clip = args.__dict__.pop("grad_norm_clip",1.0)
    write_args(args)  # write args before training
    losses = []
    count_params(model, logger, print_details=verbose)
    for epoch in tqdm(range(args.num_epochs_train), desc='epochs'):
        logger.info(f"start training epoch {epoch + 1}/{args.num_epochs_train}")
        logger.info(f"n_gpu = {args.n_gpu}, visible_devices: {args.visible_devices}")
        logger.info(f"number of examples per batch: {args.batch_size}*{args.n_gpu}={args.batch_size * args.n_gpu}")
        logger.info(f"number of iterations per epoch: {len(train_dataloader)}")
        logger.info(f"number of examples per epoch: {len(train_dataset)}")

        base_steps = len(train_dataloader) * epoch
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for it, batch in pbar:
            batch.to(device)
            if "label" in batch:
                batch["labels"] = batch.pop("label")
            outputs = model(**batch, return_dict=True)
            loss = outputs.loss
            if args.n_gpu > 1:
                loss = loss.mean()
            losses.append(loss.item())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm_clip)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            pbar.set_description(
                f"training - epoch {epoch + 1}/{args.num_epochs_train} iter {it}: train loss {loss.item():.8f}. lr {scheduler.get_last_lr()[0]:e}")

            if args.log_steps > 0 and (base_steps + it + 1) % args.log_steps == 0:
                logger.info(f"evaluate at global step = {base_steps + it + 1}")
                logger.info(f'Step {base_steps + it + 1} - mean train loss: {np.mean(losses):.3}')
                if args.use_tb:
                    tb_writer.add_scalar("train_loss_global_step", np.mean(losses), base_steps + it + 1)
                    tb_writer.add_scalar("train_lr_global_step", scheduler.get_last_lr()[0], base_steps + it + 1)
                losses = []
                if args.do_eval:
                    eval_return = evaluate(args, logger, model, tokenizer, device, val_dataloader,
                                           steps=base_steps + it + 1,
                                           tag="global_step")
                    if args.use_tb:
                        if "eval_scores" in eval_return:
                            for key, value in eval_return["eval_scores"].items():
                                tb_writer.add_scalar(f"eval_{key}_global_step", value, base_steps + it + 1)
                    if "is_early_stop" in eval_return and eval_return["is_early_stop"]:
                        logger.info(f"run out of patience at global step = {base_steps + it + 1}, early stop")
                        if args.use_tb:
                            tb_writer.close()
                        sys.exit(0)
                    model.train()
                # else:
                #     save_ck(args, logger, model, tokenizer, steps=base_steps + it + 1, tag="global_step")

        if args.log_steps < 0:
            logger.info(f'Epoch {epoch + 1} - mean train loss: {np.mean(losses):.3}')
            logger.info(f"evaluate at epoch = {epoch + 1}")
            if args.use_tb:
                tb_writer.add_scalar("train_loss_epoch", np.mean(losses), epoch + 1)
                tb_writer.add_scalar("train_lr_epoch", scheduler.get_last_lr()[0], epoch + 1)
            losses = []
            if args.do_eval:
                eval_return = evaluate(args, logger, model, tokenizer, device, val_dataloader, steps=epoch + 1,
                                       tag="epoch")
                if args.use_tb:
                    if "eval_scores" in eval_return:
                        for key, value in eval_return["eval_scores"].items():
                            tb_writer.add_scalar(f"eval_{key}_epoch", value, epoch + 1)
                if "is_early_stop" in eval_return and eval_return["is_early_stop"]:
                    logger.info(f"run out of patience at epoch = {epoch + 1}, early stop")
                    if args.use_tb:
                        tb_writer.close()
                    sys.exit(0)
                model.train()
            else:
                # if do not do evaluate, the checkpoint at the end of epoch needs to be saved
                save_ck(args, logger, model, tokenizer, steps=epoch + 1, tag="epoch")
    if args.use_tb:
        tb_writer.close()
