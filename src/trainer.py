import os
import math
import time
import datetime
import subprocess
import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only


def my_save(args, trainer, dd, ff):
    """Save model state dictionary to local and remote storage."""
    fn = os.path.basename(ff)
    if '14b-run1' in ff:
        fff = f'/dev/shm/{fn}'
        torch.save(dd, fff)
        subprocess.Popen(f"aws s3 mv {fff} s3://rwkv-14b-4k/{fn} --quiet", shell=True)
    elif 'world/14b' in ff or 'world/7b' in ff:
        aa = ff.split('/')[1]
        fff = f'/dev/shm/{aa}-{fn}'
        torch.save(dd, fff)
        subprocess.Popen(f"aws s3 mv {fff} s3://rwkv-world/{aa}-{fn} --quiet", shell=True)
    else:
        if 'deepspeed_stage_3' in args.strategy:
            trainer.save_checkpoint(ff, weights_only=True)
        else:
            if args.train_type == 'states':
                ddd = {k: v.clone() for k, v in dd.items() if 'time_sta' in k}
                torch.save(ddd, ff)
            else:
                torch.save(dd, ff)


class train_callback(pl.Callback):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def on_train_batch_start(self, trainer, pl_module):
        args = self.args
        real_step = trainer.global_step + args.epoch_begin * args.epoch_steps

        # Learning rate schedule
        w_step = args.warmup_steps
        if args.lr_final == args.lr_init or args.epoch_count == 0:
            lr = args.lr_init
        else:
            decay_step = real_step - args.my_pile_edecay * args.epoch_steps
            decay_total = (args.epoch_count - args.my_pile_edecay) * args.epoch_steps
            progress = (decay_step - w_step + 1) / (decay_total - w_step)
            progress = min(1, max(0, progress))

            if args.lr_final == 0 or args.lr_init == 0:  # Linear decay
                lr = args.lr_init + (args.lr_final - args.lr_init) * progress
            else:  # Exponential decay
                lr = args.lr_init * math.exp(math.log(args.lr_final / args.lr_init) * pow(progress, 1))

        if args.my_exit_tokens != 0:  # Cosine decay
            real_tokens = real_step * args.ctx_len * args.real_bsz
            warmup_tokens = w_step * args.ctx_len * args.real_bsz
            progress = (real_tokens - warmup_tokens) / (abs(args.my_exit_tokens) - warmup_tokens)
            progress = max(0, min(1, progress))
            lr_final_factor = args.lr_final / args.lr_init                
            lr_mult = (0.5 + lr_final_factor / 2) + (0.5 - lr_final_factor / 2) * math.cos(math.pi * progress)
            if args.my_exit_tokens > 0:
                lr = args.lr_init * lr_mult
            else:
                lr = (lr + args.lr_init * lr_mult) / 2
            if progress >= 1:
                if trainer.is_global_zero or 'deepspeed_stage_3' in args.strategy:
                    my_save(args, trainer, pl_module.state_dict(), f"{args.proj_dir}/rwkv-final.pth")
                    exit(0)

        if trainer.global_step < w_step:
            lr = lr * (0.2 + 0.8 * trainer.global_step / w_step)

        wd_now = args.weight_decay
        if args.weight_decay_final > 0:
            wd_now *= math.exp(math.log(args.weight_decay_final / args.weight_decay) * progress)

        for param_group in trainer.optimizers[0].param_groups:
            if param_group["weight_decay"] > 0:
                param_group["weight_decay"] = wd_now
            if args.layerwise_lr > 0:
                param_group["lr"] = lr * param_group["my_lr_scale"]
            else:
                param_group["lr"] = lr

        trainer.my_lr = lr
        trainer.my_wd = wd_now

        if trainer.global_step == 0 and trainer.is_global_zero:
            self._initialize_logging(trainer)

    def _initialize_logging(self, trainer):
        """Initialize logging."""
        trainer.my_loss_sum = 0
        trainer.my_loss_count = 0
        trainer.my_log = open(os.path.join(self.args.proj_dir, "train_log.txt"), "a")
        trainer.my_log.write(f"NEW RUN {self.args.my_timestamp}\n{vars(self.args)}\n")
        try:
            print(f"\n{trainer.strategy.config}\n")
            trainer.my_log.write(f"{trainer.strategy.config}\n")
        except AttributeError:
            pass
        trainer.my_log.flush()
        if self.args.wandb:
            import wandb
            print("Login to wandb...")
            wandb.init(
                project=self.args.wandb,
                name=f"{self.args.run_name} {self.args.my_timestamp}",
                config=self.args,
                save_code=False,
            )
            trainer.my_wandb = wandb

    def on_train_batch_end(self, trainer, pl_module, outputs):
        args = self.args
        token_per_step = args.ctx_len * args.real_bsz
        real_step = trainer.global_step + args.epoch_begin * args.epoch_steps
        if trainer.is_global_zero:  # Logging
            self._log_training_metrics(trainer, outputs, token_per_step, real_step)
        
        if (trainer.is_global_zero or 'deepspeed_stage_3' in args.strategy) and args.magic_prime > 0:
            expand_factor = 2 if args.my_qa_mask > 0 else 1
            if int(real_step) == int(args.magic_prime * expand_factor // args.real_bsz) - 1 + int(args.my_random_steps):
                to_save_dict = pl_module.state_dict()
                my_save(args, trainer, to_save_dict, f"{args.proj_dir}/rwkv-final.pth")

    def _log_training_metrics(self, trainer, outputs, token_per_step, real_step):
        """Log training metrics."""
        t_now = time.time_ns()
        kt_s = 0
        try:
            t_cost = (t_now - trainer.my_time_ns) / 1e9
            kt_s = token_per_step / t_cost / 1000
            self.log("REAL it/s", 1.0 / t_cost, prog_bar=True, on_step=True)
            self.log("Kt/s", kt_s, prog_bar=True, on_step=True)
        except AttributeError:
            pass
        trainer.my_time_ns = t_now
        trainer.my_loss = outputs["loss"].float().mean().item() if pl.__version__[0] == '2' else trainer.my_loss_all.float().mean().item()
        trainer.my_loss_sum += trainer.my_loss
        trainer.my_loss_count += 1
        trainer.my_epoch_loss = trainer.my_loss_sum / trainer.my_loss_count
        self.log("lr", trainer.my_lr, prog_bar=True, on_step=True)
        self.log("loss", trainer.my_epoch_loss, prog_bar=True, on_step=True)
        if self.args.wandb:
            log_data = {
                "loss": trainer.my_loss,
                "lr": trainer.my_lr,
                "wd": trainer.my_wd,
                "Gtokens": real_step * token_per_step / 1e9,
            }
            if kt_s > 0:
                log_data["kt/s"] = kt_s
            trainer.my_wandb.log(log_data, step=int(real_step))

    def on_train_epoch_start(self, trainer):
        args = self.args
        dataset = trainer.train_dataloader.dataset if pl.__version__[0] == '2' else trainer.train_dataloader.dataset.datasets
        assert "MyDataset" in str(dataset)
        dataset.global_rank = trainer.global_rank
        dataset.real_epoch = int(args.epoch_begin + trainer.current_epoch)
        dataset.world_size = trainer.world_size

    def on_train_epoch_end(self, trainer, pl_module):
        args = self.args
        if trainer.is_global_zero or 'deepspeed_stage_3' in args.strategy:
            self._save_epoch_checkpoint(args, trainer, pl_module)

        if trainer.is_global_zero:
            self._log_epoch_end(trainer)

    def _save_epoch_checkpoint(self, args, trainer, pl_module):
        """Save model checkpoint at the end of the epoch."""
        if (args.epoch_save > 0 and trainer.current_epoch % args.epoch_save == 0) or (trainer.current_epoch == args.epoch_count - 1):
            to_save_dict = pl_module.state_dict()
            if args.data_type == 'wds_img':
                to_save_dict = {k: v for k, v in to_save_dict.items() if k.startswith('encoder.') or k.startswith('decoder.')}
            try:
                my_save(args, trainer, to_save_dict, f"{args.proj_dir}/rwkv-{args.epoch_begin + trainer.current_epoch}.pth")
            except Exception as e:
                print('Error\n\n', e, '\n\n')

    def _log_epoch_end(self, trainer):
        """Log metrics at the end of the epoch."""
        args = self.args
        trainer.my_log.write(f"{args.epoch_begin + trainer.current_epoch} {trainer.my_epoch_loss:.6f} {math.exp(trainer.my_epoch_loss):.4f} {trainer.my_lr:.8f} {datetime.datetime.now()} {trainer.current_epoch}\n")
        trainer.my_log.flush()
        trainer.my_loss_sum = 0
        trainer.my_loss_count = 0
        if (args.epoch_begin + trainer.current_epoch) >= args.my_exit:
            exit(0)


@rank_zero_only
def generate_init_weight(model, init_weight_name):
    """Generate and save initial weights for the model."""
    mm = model.generate_init_weight()

    if model.args.my_pile_stage == 1 and model.args.load_model:
        print(f"Combine weights from {model.args.load_model}...")
        load_dict = torch.load(model.args.load_model, map_location="cpu")
        for k, src in load_dict.items():
            if k not in mm:
                print('missing', k)
                exit(0)
            try:
                mm[k] = src.reshape(mm[k].shape)
            except ValueError:
                tmp = mm[k].squeeze().clone()
                ss, dd = src.shape[0], tmp.shape[0]
                for i in range(dd):
                    pos = i / dd * ss
                    p0, ii = int(math.floor(pos)), pos - int(math.floor(pos))
                    tmp[i] = src[min(p0, ss-1)] * (1-ii) + src[min(p0+1, ss-1)] * ii
                mm[k] = tmp.reshape(mm[k].shape)
                print(src.squeeze().cpu().numpy()[:10], '...', src.squeeze().cpu().numpy()[-10:])
                print(mm[k].squeeze().cpu().numpy()[:10], '...', mm[k].squeeze().cpu().numpy()[-10:])

    print(f"Save to {init_weight_name}...")
    torch.save(mm, init_weight_name)

    if model.args.my_pile_stage == 1:
        print("Done. Now go for stage 2.")
        exit(0)
