#!/usr/bin/env python3
# Llama2 70B LoRA training for MLPerf
import time, json, re, argparse
from pathlib import Path
from collections import Counter
from tinygrad import Device, GlobalCounters, Tensor, TinyJit
from tinygrad.helpers import getenv, diskcache_clear, Context
from tinygrad.nn.state import get_parameters, load_state_dict, safe_load, safe_save
from tinygrad.nn.optim import AdamW
from extra.models.llama import Transformer, convert_from_huggingface, fix_bf16
from examples.mlperf.helpers import get_training_state
from examples.mlperf.llama2_70b_lora.lora import apply_lora, freeze_base, get_lora_params
from examples.mlperf.llama2_70b_lora.dataset import create_loaders, get_tokenizer

try:
  from mlperf_logging import mllog
  import mlperf_logging.mllog.constants as mlc
  MLPERF = True
except ImportError: MLPERF = False

# ** ROUGE scoring (simple implementation) **
# ref: https://aclanthology.org/W04-1013.pdf
def tokenize_text(t): return re.findall(r'\b\w+\b', t.lower())
def get_ngrams(toks, n): return Counter(' '.join(toks[i:i+n]) for i in range(len(toks)-n+1))
def rouge_n(p_toks, r_toks, n): # rouge-N: n-gram overlap
  p_ng, r_ng = get_ngrams(p_toks, n), get_ngrams(r_toks, n)
  if not r_ng: return {"p": 0.0, "r": 0.0, "f": 0.0}
  overlap = sum((p_ng & r_ng).values())
  prec, rec = overlap/max(sum(p_ng.values()),1), overlap/sum(r_ng.values())
  return {"p": prec, "r": rec, "f": (2*prec*rec)/max(prec+rec, 1e-8)}
def rouge_l(p_toks, r_toks):
  def lcs_len(x, y):
    m, n = len(x), len(y)
    if m==0 or n==0: return 0
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1,m+1):
      for j in range(1,n+1): dp[i][j] = dp[i-1][j-1]+1 if x[i-1]==y[j-1] else max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]
  if not r_toks: return {"p": 0.0, "r": 0.0, "f": 0.0}
  lcs = lcs_len(p_toks, r_toks)
  prec, rec = lcs/max(len(p_toks),1), lcs/len(r_toks)
  return {"p": prec, "r": rec, "f": (2*prec*rec)/max(prec+rec, 1e-8)}
def compute_rouge(preds, refs):
  r1, r2, rl = [], [], []
  for p,r in zip(preds, refs):
    pt, rt = tokenize_text(p), tokenize_text(r)
    r1.append(rouge_n(pt,rt,1)), r2.append(rouge_n(pt,rt,2)), rl.append(rouge_l(pt,rt))
  avg = lambda scores: {k: sum(s[k] for s in scores)/len(scores) for k in ["p","r","f"]} if scores else {k:0.0 for k in ["p","r","f"]}
  return {"rouge-1": avg(r1), "rouge-2": avg(r2), "rouge-l": avg(rl)}

# ** MLPerf logging **
class MLLog:
  def __init__(self, cfg):
    self.cfg, self.log = cfg, None
    if cfg.get('mlperf') and MLPERF:
      mllog.config(filename=f"result_llama2_lora_{cfg['seed']}.txt")
      mllog.config(root_dir=Path(__file__).parents[3].as_posix())
      self.log = mllog.get_mllogger()
  def init_start(self):
    if self.log and self.cfg.get('init_mlperf'):
      self.log.event(key=mlc.SUBMISSION_ORG, value="tinycorp")
      self.log.event(key=mlc.SUBMISSION_PLATFORM, value=getenv("SUBMISSION_PLATFORM", "tinybox"))
      self.log.event(key=mlc.SUBMISSION_DIVISION, value=mlc.CLOSED)
      self.log.event(key=mlc.SUBMISSION_STATUS, value=mlc.ONPREM)
      self.log.event(key="submission_benchmark", value="llama2_70b_lora")
      diskcache_clear()
      self.log.event(key=mlc.CACHE_CLEAR, value=True)
      self.log.start(key=mlc.INIT_START)
  def init_end(self):
    if self.log: self.log.end(key=mlc.INIT_END)
  def run_start(self):
    if self.log and self.cfg.get('run_mlperf'):
      self.log.start(key=mlc.RUN_START)
      self.log.event(key=mlc.SEED, value=self.cfg['seed'])
  def run_stop(self, status, step):
    if self.log: self.log.end(key=mlc.RUN_STOP, metadata={"status": status, "step": step})
  def log_rouge(self, score, step):
    if self.log: self.log.event(key="eval_rouge_l_f1", value=score, metadata={"step": step})

# ** model setup **
def setup_model(cfg):
  print("loading llama2 70b...")
  llama_cfg = { # llama2 70b architecture
    "dim": 8192, "hidden_dim": 28672, "n_heads": 64, "n_kv_heads": 8, "n_layers": 80,
    "norm_eps": 1e-5, "vocab_size": 32000, "max_context": cfg['maxlen'], "jit": False
  }
  model = Transformer(**llama_cfg)
  # load weights (handle single file, sharded, or missing)
  mp = Path(cfg['modeldir'])
  if not mp.exists(): print(f"warn: {mp} not found, using random weights")
  elif mp.is_dir():
    st_single, st_index = mp/"model.safetensors", mp/"model.safetensors.index.json"
    if st_single.exists(): weights = safe_load(st_single)
    elif st_index.exists(): # sharded weights case
      idx = json.load(open(st_index))
      wmap = idx.get("weight_map", {})
      shards = sorted({mp/f for f in wmap.values()})
      shard_st = {str(sf.name): safe_load(sf) for sf in shards}
      weights = {n: shard_st[wmap[n]][n] for n in wmap.keys()}
    else: weights = None
  else: weights = safe_load(mp)
  if weights:
    weights = fix_bf16(weights)
    if any('model.layers' in k for k in weights.keys()): weights = convert_from_huggingface(weights, 80, 64, 8) # convert HF format
    load_state_dict(model, weights)
    print(f"loaded weights from {cfg['modeldir']}")
  # apply lora adapters
  print("applying lora...")
  apply_lora(model, cfg['lora_r'], cfg['lora_alpha'], cfg.get('lora_dp',0.1), cfg.get('lora_target'))
  freeze_base(model) # freeze pretrained weights
  lora_params = get_lora_params(model)
  print(f"lora params: {len(lora_params)}")
  # shard across gpus if multi-gpu
  if len(cfg['gpus']) > 1:
    params = get_parameters(model)
    for p in params: p.to_(cfg['gpus'])
    with Context(BEAM=0): Tensor.realize(*params) # realize without beam search opt
  return model, lora_params

# ** train/eval **
@TinyJit
def train_step(inp, labels, model, opt):
  opt.zero_grad()
  logits = model.forward(inp, start_pos=0, temperature=float('nan'))
  sl, slabels = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous() # shift for next token prediction
  sl_flat, slabels_flat = sl.reshape(-1, sl.shape[-1]), slabels.reshape(-1)
  valid = slabels_flat != -100 # mask out prompt tokens (only train on completion)
  if valid.sum() == 0: return Tensor([0.0])
  nc = sl_flat.shape[-1]
  log_p = sl_flat.log_softmax(axis=-1)
  oh = slabels_flat.one_hot(nc)
  per_tok = -(oh * log_p).sum(axis=-1) # cross entropy
  per_tok_masked = per_tok * valid.cast(per_tok.dtype)
  loss = per_tok_masked.sum() / valid.sum().cast(per_tok.dtype)
  loss.backward()
  opt.step()
  loss_cpu = loss.detach().to("CPU")
  Tensor.realize(loss_cpu)
  return loss_cpu

@Tensor.train(mode=False)
def evaluate(model, val, tok, max_eval):
  tot_loss, nb, preds, refs = 0.0, 0, [], []
  print(f"eval on {max_eval} batches...")
  for i,batch in enumerate(val):
    if i >= max_eval: break
    with Tensor.no_grad():
      inp, labels = batch['input_ids'], batch['labels']
      logits = model.forward(inp, start_pos=0, temperature=float('nan'))
      sl, slabels = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
      sl_flat, slabels_flat = sl.reshape(-1, sl.shape[-1]), slabels.reshape(-1)
      valid = slabels_flat != -100
      if valid.sum() > 0:
        loss = sl_flat[valid].sparse_categorical_crossentropy(slabels_flat[valid])
        tot_loss += loss.item()
        nb += 1
      pred_ids = logits.argmax(axis=-1).numpy()
      pred_text = tok.decode(pred_ids[0].tolist())
      ref_ids = labels[0][labels[0]!=-100].numpy().tolist()
      ref_text = tok.decode(ref_ids)
      preds.append(pred_text)
      refs.append(ref_text)
  return tot_loss/max(nb,1), compute_rouge(preds, refs)

# ** checkpointing **
def save_ckpt(model, opt, step, cfg, prefix="ckpt"):
  print(f"saving {prefix}_{step}")
  ckpt = get_training_state(model, opt, None)
  cpu_ckpt = {}
  for k,v in ckpt.items():
    v.realize()
    cpu = v.detach().to("CPU")
    cpu.realize()
    cpu_ckpt[k] = cpu.cast(cpu.dtype.base).contiguous()
  Tensor.realize(*[v for v in cpu_ckpt.values()])
  ckpt_path = Path(cfg['ckptdir']) / f"{prefix}_{step}.safetensors"
  safe_save(cpu_ckpt, ckpt_path)
  print(f"saved to {ckpt_path}")

def load_ckpt(model, opt, cfg):
  if not cfg.get('resume_ckptdir'): return
  print(f"resuming from {cfg['resume_ckptdir']} @ itr {cfg['resume_itr']}")
  ckpt = safe_load(f"{cfg['resume_ckptdir']}/backup_{cfg['resume_itr']}.safetensors")
  for obj,pat in [(model, "model."), (opt, "optimizer.")]:
    sd = {k.split(pat)[1]: v for k,v in ckpt.items() if k.startswith(pat)}
    with Context(DEBUG=1): load_state_dict(obj, sd, strict=False)

# ** main training loop **
def train(cfg):
  print(f"training on {cfg['gpus']}, bs={cfg['bs']}, base_lr={cfg['base_lr']}, lr={cfg['lr']}")
  for d in cfg['gpus']: Device[d]
  Tensor.manual_seed(cfg['seed'])

  ml = MLLog(cfg)
  ml.init_start()

  model, lora_params = setup_model(cfg)
  opt = AdamW(lora_params, lr=cfg['lr'], weight_decay=cfg.get('wd',0.01))

  Path(cfg['ckptdir']).mkdir(parents=True, exist_ok=True)
  load_ckpt(model, opt, cfg)

  tok = get_tokenizer()
  train_dl, val_dl = create_loaders(cfg['datadir'], tok, cfg['bs'], cfg['maxlen'])

  ml.init_end()
  ml.run_start()

  print("training...")
  best_rouge, achieved, gstep = 0.0, False, 0
  for epoch in range(cfg['epochs']):
    print(f"\nepoch {epoch+1}/{cfg['epochs']}")
    model.train()
    for batch in train_dl:
      gstep += 1
      if cfg.get('max_steps') and gstep > cfg['max_steps']: break
      GlobalCounters.reset()
      t1 = time.perf_counter()
      inp, labels = batch['input_ids'], batch['labels']
      if len(cfg['gpus'])>1: # shard batch across gpus
        inp.shard_(cfg['gpus'], axis=0)
        labels.shard_(cfg['gpus'], axis=0)
      loss = train_step(inp, labels, model, opt)
      loss_v = loss.item()
      t2 = time.perf_counter()
      if gstep%10 == 0:
        gf = GlobalCounters.global_ops * 1e-9 / (t2-t1)
        print(f"step {gstep}: {gf:9.2f} GFLOPS, loss: {loss_v:.5f}")
      # eval
      if gstep % cfg['eval_steps'] == 0:
        print(f"\neval @ {gstep}...")
        eloss, rouge = evaluate(model, val_dl, tok, cfg['max_eval'])
        rf = rouge.get('rouge-l',{}).get('f',0.0)
        print(f"eval - loss: {eloss:.4f}, rouge-l f1: {rf:.4f}")
        ml.log_rouge(rf, gstep)
        if rf >= cfg['target_rouge'] and not achieved:
          print(f"🎉 target rouge-l {cfg['target_rouge']} achieved! ({rf:.4f})")
          achieved, best_rouge = True, rf
          ml.run_stop("success", gstep)
          save_ckpt(model, opt, gstep, cfg, "final")
          return
        if rf > best_rouge:
          best_rouge = rf
          save_ckpt(model, opt, gstep, cfg, "best")
      # periodic ckpt
      if gstep % cfg['ckpt_steps'] == 0: save_ckpt(model, opt, gstep, cfg)
    if achieved or (cfg.get('max_steps') and gstep>=cfg['max_steps']): break
  print(f"\ntraining done! best rouge-l: {best_rouge:.4f}, target achieved: {achieved}")
  if not achieved: ml.run_stop("aborted", gstep)

# ** CLI and config **
def get_config_from_env():
  num_gpus = getenv("GPUS", 1)
  return {
    'gpus': [f"{Device.DEFAULT}:{i}" for i in range(num_gpus)],
    'seed': getenv("SEED", 42), 'bs': getenv("BS", 1*num_gpus),
    'base_lr': getenv("LEARNING_RATE", 1e-4), 'wd': getenv("WEIGHT_DECAY", 0.01),
    'maxlen': getenv("MAX_LENGTH", 8192), 'target_rouge': getenv("TARGET_ROUGE", 0.270),
    'max_steps': getenv("MAX_STEPS", 50000), 'eval_steps': getenv("EVAL_STEP_INTERVAL", 500),
    'ckpt_steps': getenv("CKPT_STEP_INTERVAL", 500), 'max_eval': getenv("MAX_EVAL_BATCHES", 100),
    'lora_r': getenv("LORA_R", 16), 'lora_alpha': getenv("LORA_ALPHA", 32.0),
    'lora_target': getenv("LORA_TARGET_MODULES", "wq,wv,wk,wo").split(','),
    'datadir': Path(getenv("DATADIR", "./dataset/govreport")),
    'modeldir': Path(getenv("MODELDIR", "./models/llama-2-70b")),
    'ckptdir': Path(getenv("CKPTDIR", "./checkpoints")),
    'resume_ckptdir': getenv("RESUME_CKPTDIR", ""), 'resume_itr': getenv("RESUME_ITR", 0),
    'mlperf': bool(getenv("LOGMLPERF")), 'init_mlperf': bool(getenv("INITMLPERF")),
    'run_mlperf': bool(getenv("RUNMLPERF")), 'epochs': 3
  }

def get_config_from_args(args):
  gpus = ['CPU:0'] if args.device=='cpu' else [f"{Device.DEFAULT}:{i}" for i in range(args.gpus)]
  return {
    'gpus': gpus, 'seed': args.seed, 'bs': args.batch_size, 'base_lr': args.learning_rate,
    'maxlen': args.max_length, 'target_rouge': args.target_rouge,
    'eval_steps': args.eval_steps, 'ckpt_steps': args.save_steps, 'epochs': args.num_epochs,
    'datadir': Path(args.dataset_path), 'modeldir': Path(args.model_path),
    'ckptdir': Path(args.output_dir), 'max_eval': 100, 'lora_r': 16, 'lora_alpha': 32.0
  }

def main():
  if getenv("GPUS"):
    cfg = get_config_from_env()
    cfg['lr'] = cfg['bs'] * cfg['base_lr']
    train(cfg)
  else:
    parser = argparse.ArgumentParser(description='llama2 70b lora training')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--max_length', type=int, default=8192)
    parser.add_argument('--eval_steps', type=int, default=500)
    parser.add_argument('--save_steps', type=int, default=1000)
    parser.add_argument('--target_rouge', type=float, default=0.270)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--device', type=str, default='auto', choices=['auto','cpu'])
    args = parser.parse_args()
    cfg = get_config_from_args(args)
    cfg['lr'] = cfg['bs'] * cfg['base_lr']
    train(cfg)

if __name__ == "__main__":
  import multiprocessing
  multiprocessing.set_start_method('spawn')
  main()
