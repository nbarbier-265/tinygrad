# LoRA (Low-Rank Adaptation) for Llama2 70B fine-tuning
# ref: https://arxiv.org/abs/2106.09685
from tinygrad import Tensor, nn
from tinygrad.helpers import getenv

class LoRALinear:
  def __init__(self, inf, outf, r=16, alpha=32.0, dp=0.1, bias=False):
    self.r, self.alpha, self.scaling = r, alpha, alpha/r if r>0 else 0.0
    self.enabled, self.merged = True, False
    self.linear = nn.Linear(inf, outf, bias=bias)
    if r > 0:
      self.lora_A, self.lora_B = nn.Linear(inf, r, bias=False), nn.Linear(r, outf, bias=False)
      self.lora_B.weight.assign(Tensor.zeros_like(self.lora_B.weight)) # init B to zero so lora starts as identity
      self.dp = dp
  def __call__(self, x):
    if self.r==0 or not self.enabled: return self.linear(x)
    out = self.linear(x)
    if self.merged: return out
    lora_out = self.lora_A(x)
    if self.dp>0.0: lora_out = lora_out * (Tensor.rand(*lora_out.shape)>self.dp) / (1.0-self.dp) # inverted dropout
    return out + self.lora_B(lora_out) * self.scaling # W = W0 + BA*alpha/r
  def merge(self): # merge lora into base for inference
    if self.r>0 and not self.merged:
      self.linear.weight.assign(self.linear.weight + (self.lora_B.weight @ self.lora_A.weight) * self.scaling)
      self.merged = True
  def unmerge(self):
    if self.r>0 and self.merged:
      self.linear.weight.assign(self.linear.weight - (self.lora_B.weight @ self.lora_A.weight) * self.scaling)
      self.merged = False

def apply_lora(model, r=16, alpha=32.0, dp=0.1, target=None, layers=None):
  target = target or ["wq", "wv", "wk", "wo"] # attention projection matrices
  orig = {}
  for i,layer in enumerate(model.layers):
    if layers is not None and i not in layers: continue
    orig[f"layer_{i}"] = {}
    for name in target:
      if hasattr(layer.attention, name):
        lin = getattr(layer.attention, name)
        inf, outf, bias = lin.weight.shape[1], lin.weight.shape[0], lin.bias is not None
        lora_lin = LoRALinear(inf, outf, r, alpha, dp, bias)
        lora_lin.linear.weight.assign(lin.weight.detach()) # copy pretrained weights
        if bias and lin.bias is not None: lora_lin.linear.bias.assign(lin.bias.detach())
        setattr(layer.attention, name, lora_lin)
        orig[f"layer_{i}"][name] = lin
  return orig

def freeze_base(model): # freeze base model, only train lora adapters
  for layer in model.layers:
    for name in ["wq", "wv", "wk", "wo"]:
      if hasattr(layer.attention, name):
        m = getattr(layer.attention, name)
        if isinstance(m, LoRALinear):
          m.linear.weight.requires_grad = False
          if m.linear.bias is not None: m.linear.bias.requires_grad = False

def get_lora_params(model): # extract trainable lora params (A,B matrices only)
  ret = []
  for layer in model.layers:
    for name in ["wq", "wv", "wk", "wo"]:
      if hasattr(layer.attention, name):
        m = getattr(layer.attention, name)
        if isinstance(m, LoRALinear) and m.r>0:
          ret += [m.lora_A.weight, m.lora_B.weight]
  return ret

def lora_config_from_env():
  return {
    'r': getenv("LORA_R", 16), 'alpha': getenv("LORA_ALPHA", 32.0), 'dp': getenv("LORA_DROPOUT", 0.1),
    'target': getenv("LORA_TARGET_MODULES", "wq,wv,wk,wo").split(','),
  }