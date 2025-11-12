# GovReport dataset for Llama2 summarization task
import json, random
from pathlib import Path
from tinygrad import Tensor

IGNORE_IDX = -100
PROMPT = "Summarize the following government document:\n\n{input}\n\nSummary:"

class Dataset:
  def __init__(self, path, tok, maxlen=8192, split="train"):
    self.path, self.tok, self.maxlen, self.split = Path(path), tok, maxlen, split
    self.data = self._load()
    print(f"loaded {len(self.data)} {split} examples")
  def _load(self):
    f = self.path / f"{self.split}.json"
    if not f.exists(): self._create_dummy()
    return json.load(open(f))
  def _create_dummy(self):
    self.path.mkdir(parents=True, exist_ok=True)
    dummy = [{"input": "Sample government policy report. "*50, "output": "Policy implementation summary.", "id": f"d{i}"} for i in range(10)]
    for s in ["train", "validation", "test"]: json.dump(dummy, open(self.path/f"{s}.json",'w'), indent=2)
    print(f"created dummy data: {len(dummy)} examples/split")
  def __len__(self): return len(self.data)
  def __getitem__(self, i): return self.data[i]
  def tokenize(self, ex): # tokenize with labels for training
    inp_toks = [self.tok.bos_token_id] + self.tok.encode(PROMPT.format(input=ex['input']))
    tgt_toks = self.tok.encode(ex['output'])
    toks = inp_toks + tgt_toks + [self.tok.eos_token_id]
    if len(toks) > self.maxlen: toks = toks[:self.maxlen]
    labels = [IGNORE_IDX]*len(inp_toks) + tgt_toks + [self.tok.eos_token_id] # mask prompt, only train on completion
    if len(labels) > self.maxlen: labels = labels[:self.maxlen]
    alen = min(len(toks), len(labels))
    attn = [1]*alen + [0]*(self.maxlen-alen)
    toks += [self.tok.pad_token_id]*(self.maxlen-len(toks))
    labels += [IGNORE_IDX]*(self.maxlen-len(labels))
    return {'input_ids': toks, 'attention_mask': attn, 'labels': labels}
  def collate(self, batch):
    b = [self.tokenize(ex) for ex in batch]
    return {
      'input_ids': Tensor([x['input_ids'] for x in b], dtype='int32'),
      'attention_mask': Tensor([x['attention_mask'] for x in b], dtype='int32'),
      'labels': Tensor([x['labels'] for x in b], dtype='int32')
    }

class DataLoader:
  def __init__(self, ds, bs=1, shuffle=True, drop_last=True):
    self.ds, self.bs, self.shuffle, self.drop_last = ds, bs, shuffle, drop_last
    self.idxs = list(range(len(ds)))
  def __len__(self): return len(self.ds)//self.bs if self.drop_last else (len(self.ds)+self.bs-1)//self.bs
  def __iter__(self):
    if self.shuffle: random.shuffle(self.idxs)
    for i in range(0, len(self.ds), self.bs):
      bidx = self.idxs[i:i+self.bs]
      if self.drop_last and len(bidx)<self.bs: continue
      yield self.ds.collate([self.ds[j] for j in bidx])

def create_loaders(path, tok, bs=1, maxlen=8192):
  train, val = Dataset(path, tok, maxlen, "train"), Dataset(path, tok, maxlen, "validation")
  return DataLoader(train, bs, True, True), DataLoader(val, bs, False, False)

def get_tokenizer(mp=None):
  from tinygrad import Tensor
  from tinygrad.apps.llm import SimpleTokenizer
  if mp and Path(mp).exists():
    if mp.endswith('.gguf'): model_gguf = Tensor(mp)
    else:
      mdir = Path(mp).parent if Path(mp).is_file() else Path(mp)
      tok_model = mdir / "tokenizer.model"
      if tok_model.exists():
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.load(str(tok_model))
        class SPTok:
          def __init__(self, sp): self.sp, self.pad_token_id, self.bos_token_id, self.eos_token_id = sp, sp.pad_id(), sp.bos_id(), sp.eos_id()
          def encode(self, t): return self.sp.encode(t, out_type=int)
          def decode(self, ids): return self.sp.decode(ids)
        return SPTok(sp)
      else: raise FileNotFoundError(f"no tokenizer in {mdir}")
  else:
    url = "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q2_K.gguf"
    print(f"loading tok from {url}")
    model_gguf = Tensor.from_url(url)
  from tinygrad.nn.state import gguf_load
  kv, _ = gguf_load(model_gguf.to(None))
  tok = SimpleTokenizer.from_gguf_kv(kv)
  tok.pad_token_id, tok.bos_token_id, tok.eos_token_id = kv.get('tokenizer.ggml.padding_token_id',0), kv.get('tokenizer.ggml.bos_token_id',1), kv.get('tokenizer.ggml.eos_token_id',2)
  return tok