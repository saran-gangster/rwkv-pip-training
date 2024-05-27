
from torch.utils.cpp_extension import load
import torch
import torch.nn as nn
from model import USE_JIT, RUN_CUDA_RWKV6 , RUN_CUDA_RWKV6_STATE

if USE_JIT:
    MyModule = torch.jit.ScriptModule
    MyFunction = torch.jit.script_method
else:
    MyModule = nn.Module
    MyFunction = lambda x: x  

time_mixing_cuda = load(
    name="rwkv6_timemix", 
    sources=["model/cuda/rwkv6_timemix_op.cpp", "model/cuda/rwkv6_timemix.cu"],
    verbose=True
)

class RWKV_Tmix_x060(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - ratio_0_to_1  # 1 to ~0
            ddd = torch.arange(args.n_embd, dtype=torch.float32, device='cpu').reshape(1, 1, -1) / args.n_embd

            # fancy time_mix
            self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            D_MIX_LORA = 32  # generate TIME_MIX for w,k,v,r
            self.time_maa_w1 = nn.Parameter(torch.zeros(args.n_embd, D_MIX_LORA * 4))
            self.time_maa_w2 = nn.Parameter(torch.zeros(4, D_MIX_LORA, args.n_embd).uniform_(-0.01, 0.01))

            # fancy time_decay
            decay_speed = torch.arange(args.dim_att, dtype=torch.float32, device='cpu')
            decay_speed = -6 + 5 * (decay_speed / (args.dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(1, 1, -1))

            D_DECAY_LORA = 64
            self.time_decay_w1 = nn.Parameter(torch.zeros(args.n_embd, D_DECAY_LORA))
            self.time_decay_w2 = nn.Parameter(torch.zeros(D_DECAY_LORA, args.dim_att).uniform_(-0.01, 0.01))

            tmp = torch.zeros(args.dim_att, dtype=torch.float32, device='cpu')
            for n in range(args.dim_att):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (args.dim_att - 1))) + zigzag
            self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)
        self.ln_x = nn.GroupNorm(self.n_head, args.dim_att, eps=(1e-5) * (args.head_size_divisor ** 2))

    @MyFunction
    def jit_func(self, x):
        xx = self.time_shift(x) - x

        xw = torch.empty_like(x)
        xk = torch.empty_like(x)
        xv = torch.empty_like(x)
        xr = torch.empty_like(x)

        time_mixing_cuda.time_mixing_forward(
            x.size(0), x.size(1), x.size(2),
            x, xx, self.time_maa_x, self.time_maa_w,
            self.time_maa_k, self.time_maa_v, self.time_maa_r, 
            self.time_maa_w1, self.time_maa_w2.transpose(0, 1), # Transpose time_maa_w2 
            xw, xk, xv, xr
        )

        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)

        ww = torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2
        w = self.time_decay + ww

        return r, k, v, w

    @MyFunction
    def jit_func_2(self, x):
        return self.output(self.ln_x(x.view(x.shape[0] * x.shape[1], -1)).view(x.shape))

    def forward(self, x):
        r, k, v, w = self.jit_func(x)
        x = RUN_CUDA_RWKV6(x.shape[0], x.shape[1], self.args.dim_att, self.n_head, r, k, v, w, self.time_faaaa)
        return self.jit_func_2(x)


########################################################################################################

class RWKV_Tmix_x060_state(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - ratio_0_to_1  # 1 to ~0
            ddd = torch.arange(args.n_embd, dtype=torch.float32, device='cpu').reshape(1, 1, -1) / args.n_embd

            # fancy time_mix
            self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            D_MIX_LORA = 32  # generate TIME_MIX for w,k,v,r
            self.time_maa_w1 = nn.Parameter(torch.zeros(args.n_embd, D_MIX_LORA * 4))
            self.time_maa_w2 = nn.Parameter(torch.zeros(4, D_MIX_LORA, args.n_embd).uniform_(-0.01, 0.01))

            # fancy time_decay
            decay_speed = torch.arange(args.dim_att, dtype=torch.float32, device='cpu')
            decay_speed = -6 + 5 * (decay_speed / (args.dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(1, 1, -1))

            D_DECAY_LORA = 64
            self.time_decay_w1 = nn.Parameter(torch.zeros(args.n_embd, D_DECAY_LORA))
            self.time_decay_w2 = nn.Parameter(torch.zeros(D_DECAY_LORA, args.dim_att).uniform_(-0.01, 0.01))

            tmp = torch.zeros(args.dim_att, dtype=torch.float32, device='cpu')
            for n in range(args.dim_att):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (args.dim_att - 1))) + zigzag
            self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))
            self.time_state = nn.Parameter(torch.zeros(self.n_head, self.head_size, self.head_size))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)
        self.ln_x = nn.GroupNorm(self.n_head, args.dim_att, eps=(1e-5) * (args.head_size_divisor ** 2))

    @MyFunction
    def jit_func(self, x):
        xx = self.time_shift(x) - x

        xxx = x + xx * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(x.shape[0] * x.shape[1], 4, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(4, x.shape[0], x.shape[1], -1)
        mw, mk, mv, mr = xxx.unbind(dim=0)

        xw = x + xx * (self.time_maa_w + mw)
        xk = x + xx * (self.time_maa_k + mk)
        xv = x + xx * (self.time_maa_v + mv)
        xr = x + xx * (self.time_maa_r + mr)

        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)

        ww = torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2
        w = self.time_decay + ww

        return r, k, v, w

    @MyFunction
    def jit_func_2(self, x):
        return self.output(self.ln_x(x.view(x.shape[0] * x.shape[1], -1)).view(x.shape))

    def forward(self, x):
        r, k, v, w = self.jit_func(x)
        x = RUN_CUDA_RWKV6_STATE(x.shape[0], x.shape[1], self.args.dim_att, self.n_head,
                                 r, k, v, w, self.time_faaaa, self.time_state)
        return self.jit_func_2(x)


########################################################################################################

class RWKV_CMix_x060(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.arange(args.n_embd, dtype=torch.float32, device='cpu').reshape(1, 1, -1) / args.n_embd
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))

        self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
        self.receptance = nn.Linear(args.n_embd, args.n_embd, bias=False)
        self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

    @MyFunction
    def forward(self, x):
        xx = self.time_shift(x) - x
        xk = x + xx * self.time_maa_k
        xr = x + xx * self.time_maa_r
        return torch.sigmoid(self.receptance(xr)) * self.value(torch.relu(self.key(xk)) ** 2)