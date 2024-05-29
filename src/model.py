import os
import math
import gc
from src.mixes import RWKV_CMix_x060, RWKV_Tmix_x060, RWKV_Tmix_x060_state

import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.strategies import DeepSpeedStrategy
from torch.utils.cpp_extension import load

if importlib.util.find_spec('deepspeed'):
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

if 'x060' in os.environ.get("RWKV_MY_TESTING", ""):
    if os.environ["RWKV_TRAIN_TYPE"] == 'states':
        wkv6state_cuda = load(
            name="wkv6state",
            sources=["cuda/wkv6state_op.cpp", f"cuda/wkv6state_cuda.cu"],
            verbose=True,
            extra_cuda_cflags=[
                "-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3",
                "--extra-device-vectorization", f"-D_N_={int(os.environ['RWKV_HEAD_SIZE_A'])}",
                f"-D_T_={int(os.environ['RWKV_CTXLEN'])}"
            ]
        )

        class WKV_6STATE(torch.autograd.Function):
            @staticmethod
            def forward(ctx, B, T, C, H, r, k, v, w, u, s):
                ctx.save_for_backward(r, k, v, w, u, s)
                ctx.B, ctx.T, ctx.C, ctx.H = B, T, C, H
                assert all(x.dtype == torch.bfloat16 and x.is_contiguous() for x in [r, k, v, w, u, s])
                assert C // H == int(os.environ['RWKV_HEAD_SIZE_A'])
                y = torch.empty((B, T, C), device=r.device, dtype=torch.bfloat16, memory_format=torch.contiguous_format)
                wkv6state_cuda.forward(B, T, C, H, r, k, v, w, u, s, y)
                return y

            @staticmethod
            def backward(ctx, gy):
                assert gy.dtype == torch.bfloat16 and gy.is_contiguous()
                B, T, C, H = ctx.B, ctx.T, ctx.C, ctx.H
                r, k, v, w, u, s = ctx.saved_tensors
                gr, gk, gv, gw = (torch.empty((B, T, C), device=gy.device, requires_grad=False,
                                          dtype=torch.bfloat16, memory_format=torch.contiguous_format) for _ in range(4))
                gu = torch.empty((B, C), device=gy.device, requires_grad=False,
                                 dtype=torch.bfloat16, memory_format=torch.contiguous_format)
                gs = torch.empty((B, H, C//H, C//H), device=gy.device, requires_grad=False,
                                 dtype=torch.bfloat16, memory_format=torch.contiguous_format)
                wkv6state_cuda.backward(B, T, C, H, r, k, v, w, u, s, gy, gr, gk, gv, gw, gu, gs)
                return (None, None, None, None, gr, gk, gv, gw, torch.sum(gu, 0).view(H, C//H),
                        torch.sum(gs, 0).view(H, C//H, C//H))

        RUN_CUDA_RWKV6_STATE = WKV_6STATE.apply
    else:
        wkv6_cuda = load(
            name="wkv6",
            sources=["cuda/wkv6_op.cpp", f"cuda/wkv6_cuda.cu"],
            verbose=True,
            extra_cuda_cflags=[
                "-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3",
                "--extra-device-vectorization", f"-D_N_={int(os.environ['RWKV_HEAD_SIZE_A'])}",
                f"-D_T_={int(os.environ['RWKV_CTXLEN'])}"
            ]
        )

        class WKV_6(torch.autograd.Function):
            @staticmethod
            def forward(ctx, B, T, C, H, r, k, v, w, u):
                ctx.save_for_backward(r, k, v, w, u)
                ctx.B, ctx.T, ctx.C, ctx.H = B, T, C, H
                assert all(x.dtype == torch.bfloat16 and x.is_contiguous() for x in [r, k, v, w, u])
                assert C // H == int(os.environ['RWKV_HEAD_SIZE_A'])
                y = torch.empty((B, T, C), device=r.device, dtype=torch.bfloat16, memory_format=torch.contiguous_format)
                wkv6_cuda.forward(B, T, C, H, r, k, v, w, u, y)
                return y

            @staticmethod
            def backward(ctx, gy):
                assert gy.dtype == torch.bfloat16 and gy.is_contiguous()
                B, T, C, H = ctx.B, ctx.T, ctx.C, ctx.H
                r, k, v, w, u = ctx.saved_tensors
                gr, gk, gv, gw = (torch.empty((B, T, C), device=gy.device, requires_grad=False,
                                          dtype=torch.bfloat16, memory_format=torch.contiguous_format) for _ in range(4))
                gu = torch.empty((B, C), device=gy.device, requires_grad=False,
                                 dtype=torch.bfloat16, memory_format=torch.contiguous_format)
                wkv6_cuda.backward(B, T, C, H, r, k, v, w, u, gy, gr, gk, gv, gw, gu)
                return (None, None, None, None, gr, gk, gv, gw, torch.sum(gu, 0).view(H, C//H))

        RUN_CUDA_RWKV6 = WKV_6.apply

# Use a boolean variable instead of string comparison
USE_JIT = os.environ.get("RWKV_JIT_ON") == "1"

if USE_JIT:
    MyModule = torch.jit.ScriptModule
    MyFunction = torch.jit.script_method
else:
    MyModule = nn.Module
    MyFunction = lambda x: x  


########################################################################################################
# The RWKV Model with our blocks
########################################################################################################

class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        if layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)
            self.pos_emb_x = nn.Parameter(torch.zeros((1, args.my_pos_emb, args.n_embd))) if args.my_pos_emb > 0 else None
            self.pos_emb_y = nn.Parameter(torch.zeros((args.my_pos_emb, 1, args.n_embd))) if args.my_pos_emb > 0 else None

        if layer_id == 0 and self.args.pre_ffn > 0:
            self.ffnPre = RWKV_CMix_x060(args, 0)
        else:
            self.att = (RWKV_Tmix_x060_state if os.environ["RWKV_TRAIN_TYPE"] == 'states'
                        else RWKV_Tmix_x060)(args, layer_id) if 'x060' in os.environ.get("RWKV_MY_TESTING", "") else None

        self.ffn = RWKV_CMix_x060(args, layer_id) if 'x060' in os.environ.get("RWKV_MY_TESTING", "") else None

        if args.tiny_att_dim > 0 and layer_id == args.tiny_att_layer:
            self.tiny_ln = nn.LayerNorm(args.n_embd)
            self.tiny_q = nn.Linear(args.n_embd, args.tiny_att_dim, bias=False)
            self.tiny_k = nn.Linear(args.n_embd, args.tiny_att_dim, bias=False)
            self.tiny_v = nn.Linear(args.n_embd, args.n_embd, bias=False)
            self.register_buffer("tiny_mask", torch.tril(torch.ones(args.ctx_len, args.ctx_len)))

        self.drop0 = nn.Dropout(p=args.dropout) if args.dropout > 0 else None
        self.drop1 = nn.Dropout(p=args.dropout) if args.dropout > 0 else None

    def forward(self, x, x_emb=None):
        args = self.args
        B, T, C = x.size()
        if self.layer_id == 0:
            x = self.ln0(x)
            if args.my_pos_emb > 0:
                x = x + (self.pos_emb_x + self.pos_emb_y).reshape(T + 1, -1)[:-1, :]

        if self.drop0:
            if self.layer_id == 0 and args.pre_ffn > 0:
                x = self.drop0(x + self.ffnPre(self.ln1(x)))
            else:
                x = self.drop0(x + self.att(self.ln1(x)))
            x = self.drop1(x + self.ffn(self.ln2(x)))
        else:
            if self.layer_id == 0 and args.pre_ffn > 0:
                x = x + self.ffnPre(self.ln1(x))
            else:
                x = x + self.att(self.ln1(x))
            x = x + self.ffn(self.ln2(x))

        if args.tiny_att_dim > 0 and self.layer_id == args.tiny_att_layer:
            xx = self.tiny_ln(x)
            q = self.tiny_q(xx)[:, :T, :]
            k = self.tiny_k(xx)[:, :T, :]
            c = (q @ k.transpose(-2, -1)) * (args.tiny_att_dim ** (-0.5))
            c = c.masked_fill(self.tiny_mask[:T, :T] == 0, 0)
            x = x + c @ self.tiny_v(x_emb)
        return x


class L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, y):
        ctx.save_for_backward(y)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]
        factor = 1e-4 / (y.shape[0] * y.shape[1])
        maxx = torch.max(y, -1, keepdim=True).values
        gy = torch.zeros_like(y)
        gy.scatter_(-1, torch.argmax(y, -1, keepdim=True), maxx * factor)
        return grad_output, gy


class RWKV(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # Set default values for missing arguments
        args.dim_att = getattr(args, 'dim_att', args.n_embd)
        args.dim_ffn = getattr(args, 'dim_ffn', int((args.n_embd * (4 if '-f4' in
                                        os.environ.get("RWKV_MY_TESTING", "") else 3.5)) // 32 * 32))
        args.tiny_att_layer = getattr(args, 'tiny_att_layer', -1)
        args.tiny_att_dim = getattr(args, 'tiny_att_dim', -1)

        # Assertions to ensure correct dimensions
        assert args.n_embd % 32 == 0
        assert args.dim_att % 32 == 0
        assert args.dim_ffn % 32 == 0

        self.emb = nn.Embedding(args.vocab_size, args.n_embd)
        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])
        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

        if args.head_qk > 0:
            self.head_q = nn.Linear(args.n_embd, args.head_qk, bias=False)
            self.head_k = nn.Linear(args.n_embd, args.head_qk, bias=False)
            self.register_buffer("copy_mask", torch.tril(torch.ones(args.ctx_len, args.ctx_len)))

        self.drop0 = nn.Dropout(p=args.dropout) if args.dropout > 0 else None

    def configure_optimizers(self):
        args = self.args

        lr_decay = []
        lr_1x = []
        lr_2x = []
        lr_3x = []
        for n, p in self.named_parameters():
            # Optimization flags based on parameter names
            if args.train_type == 'states' and 'time_sta' not in n:
                continue

            if ("_w1" in n or "_w2" in n) and args.layerwise_lr > 0:
                lr_1x.append(n)
            elif "time_sta" in n and args.weight_decay > 0:
                lr_decay.append(n)
            elif ("time_mix" in n or "time_maa" in n) and args.layerwise_lr > 0:
                (lr_2x if args.my_pile_stage == 2 else lr_1x).append(n)
            elif ("time_decay" in n or "time_daaaa" in n) and args.layerwise_lr > 0:
                (lr_3x if args.my_pile_stage == 2 else lr_2x).append(n)
            elif "time_faaaa" in n and args.layerwise_lr > 0:
                (lr_2x if args.my_pile_stage == 2 else lr_1x).append(n)
            elif "time_first" in n and args.layerwise_lr > 0:
                lr_3x.append(n)
            elif len(p.squeeze().shape) >= 2 and args.weight_decay > 0:
                lr_decay.append(n)
            else:
                lr_1x.append(n)

        # Sort parameter names for consistent output
        lr_decay.sort()
        lr_1x.sort()
        lr_2x.sort()
        lr_3x.sort()

        if self.trainer.is_global_zero:
            print('decay', lr_decay, '\n')
            print('1x', lr_1x, '\n')
            print('2x', lr_2x, '\n')
            print('3x', lr_3x, '\n')

        param_dict = {n: p for n, p in self.named_parameters()}

        # Create optimizer groups based on learning rate schedules
        if args.layerwise_lr > 0:
            if args.my_pile_stage == 2:
                optim_groups = [
                    {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                    {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 5.0},
                    {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 5.0},
                ]
            else:
                optim_groups = [
                    {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                    {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 2.0},
                    {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 3.0},
                ]
        else:
            optim_groups = [{"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0}]

        if args.weight_decay > 0:
            optim_groups += [{"params": [param_dict[n] for n in lr_decay],
                              "weight_decay": args.weight_decay, "my_lr_scale": 1.0}]

        # Use DeepSpeedCPUAdam if offloading is enabled, otherwise use FusedAdam
        AdamOptimizer = DeepSpeedCPUAdam if self.deepspeed_offload else FusedAdam

        return AdamOptimizer(
            optim_groups, lr=self.args.lr_init, betas=self.args.betas,
            eps=self.args.adam_eps, bias_correction=True,
            adamw_mode=args.weight_decay > 0, amsgrad=False
        )

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return cfg.get("offload_optimizer") or cfg.get("offload_param")
        return False

    def forward(self, idx):
        args = self.args
        B, T = idx.size()
        assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."

        x = self.emb(idx)
        x_emb = x if args.tiny_att_dim > 0 else None

        x = self.drop0(x) if self.drop0 else x

        for block in self.blocks:
            x = deepspeed.checkpointing.checkpoint(block, x, x_emb) if x_emb is not None else deepspeed.checkpointing.checkpoint(block, x) if args.grad_cp == 1 else block(x, x_emb) if x_emb is not None else block(x)

        x = self.ln_out(x)

        if args.head_qk > 0:
            q = self.head_q(x)[:, :T, :]
            k = self.head_k(x)[:, :T, :]
            c = (q @ k.transpose(-2, -1)) * (1.0 / args.head_qk)
            c = c.masked_fill(self.copy_mask[:T, :T] == 0, 0)

            if "32" in os.environ["RWKV_FLOAT_MODE"]:
                c = c @ F.one_hot(idx, num_classes=args.vocab_size)
            elif os.environ["RWKV_FLOAT_MODE"] == "fp16":
                c = c @ F.one_hot(idx, num_classes=args.vocab_size).half()
            elif os.environ["RWKV_FLOAT_MODE"] == "bf16":
                c = c @ F.one_hot(idx, num_classes=args.vocab_size).bfloat16()

            x = self.head(x) + c
        else:
            x = self.head(x)

        return x
    
    def training_step(self, batch):
        args = self.args
        if args.my_qa_mask != 1:
            idx, targets = batch
            logits = self(idx)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            idx, targets, mask = batch
            mask = mask.view(-1)
            sum_mask = torch.sum(mask).item()

            logits = self(idx)
            if sum_mask == mask.shape[0]:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            else:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none')
                loss = torch.sum(loss * mask) / sum_mask

        return L2Wrap.apply(loss, logits)

    def training_step_end(self, batch_parts):
        if int(pl.__version__[0]) > 1:
            return
        all_losses = self.all_gather(batch_parts)
        if self.trainer.is_global_zero:
            self.trainer.my_loss_all = all_losses

    def generate_init_weight(self):
        print(
            """
############################################################################
#
# Init model weight (slow for large models)...
#
############################################################################
"""
        )
        m = {}
        n_params = 0
        for n, p in self.state_dict().items():
            shape = p.shape
            s0 = str(shape[0]) if len(shape) > 0 else ""
            s1 = str(shape[1]) if len(shape) > 1 else ""
            s2 = str(shape[2]) if len(shape) > 2 else ""
            print(f"{s0.ljust(5)} {s1.ljust(5)} {s2.ljust(5)} {n}", end="")


            if any(k in n for k in ("ln_", ".ln", "time_", "_mask", "pos_emb", '.mask.',
                                    '_w', '_w1', '_w2', '_bias')):
                if 'ln_x.weight' in n:
                    layer_scale = (1 + int(n.split('.')[1])) / self.args.n_layer
                    m[n] = (p * 0.0) + (layer_scale ** 0.7)
                else:
                    m[n] = p.clone()
                print()
            elif n == "emb.weight":
                m[n] = p.clone()
                scale = -1e-4
                nn.init.uniform_(m[n], a=scale, b=-scale)
                print(f" [scale {scale}]")
            elif n == "head.weight":
                m[n] = p.clone()
                scale = 0.5 * math.sqrt(self.args.vocab_size / self.args.n_embd) if self.args.vocab_size > self.args.n_embd else 0.5
                nn.init.orthogonal_(m[n], gain=scale)
                print(f" [scale {scale}]")
            else:
                assert n.endswith('.weight')
                scale = 1.0
                if any(kk in n for kk in (".att.output.", ".ffn.value.", ".ffn.receptance.",
                                        ".ffnPre.value.", ".ffnPre.receptance.",
                                        "head_q.", '.oo.', '.rr.')):
                    scale = 0
                elif "head_k." in n:
                    scale = 0.1
                elif "head_q." in n:
                    scale = 0
                elif ".att.key." in n:
                    scale = 0.1
                elif ".att.gate." in n:
                    scale = 0.1

                print(f" [scale {scale}]")
                m[n] = torch.empty((shape[0], shape[1]))
                if scale == 0:
                    nn.init.zeros_(m[n])
                elif scale < 0:
                    nn.init.uniform_(m[n], a=scale, b=-scale)
                else:
                    nn.init.orthogonal_(m[n], gain=scale)
            m[n] = m[n].to(device=p.device, dtype=p.dtype)
            n_params += m[n].numel()
        print('model params', n_params)
        gc.collect()
        torch.cuda.empty_cache()
        return m
