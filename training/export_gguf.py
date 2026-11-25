#!/usr/bin/env python3
"""Export a LionAI scratch checkpoint to custom GGUF v3 and verify exact parity."""
import argparse, hashlib, json, struct, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from lionai_scratch import LionAIConfig, build_model, model_name
from training.lionai_gguf_runtime import GGUFReader, load_gguf_model

def s(v):
    b=v.encode("utf-8"); return struct.pack("<Q",len(b))+b
def m(k,v):
    k=s(k)
    if isinstance(v,str): return k+struct.pack("<I",8)+s(v)
    if isinstance(v,bool): return k+struct.pack("<IB",7,v)
    if isinstance(v,int): return k+struct.pack("<II",4,v)
    if isinstance(v,float): return k+struct.pack("<If",6,v)
    if isinstance(v,list) and all(isinstance(x,str) for x in v): return k+struct.pack("<IIQ",9,8,len(v))+b"".join(s(x) for x in v)
    raise TypeError("Unsupported GGUF metadata: "+str(k))
def export(checkpoint_path, output_path):
    ck=torch.load(checkpoint_path,map_location="cpu",weights_only=True)
    if ck.get("initialized_from")!="random": raise ValueError("Expected from-scratch checkpoint")
    cfg=LionAIConfig(**ck["config"]); name=model_name(cfg)
    if ck.get("model")!=name: raise ValueError("Checkpoint architecture identity mismatch")
    labels=ck["labels"]
    if len(labels)!=cfg.num_classes: raise ValueError("Class label count mismatch")
    model=build_model(cfg).cpu().eval(); model.load_state_dict(ck["state_dict"],strict=True)
    metadata=[("general.architecture","lionai"),("general.name",name),("general.file_type",0),("general.alignment",32),
      ("lionai.model",name),("lionai.initialized_from","random"),("lionai.tokenizer",ck.get("tokenizer","utf8-byte-v1")),
      ("lionai.vocab_size",cfg.vocab_size),("lionai.d_model",cfg.d_model),("lionai.n_heads",cfg.n_heads),
      ("lionai.n_layers",cfg.n_layers),("lionai.feedforward",cfg.feedforward),("lionai.max_length",cfg.max_length),
      ("lionai.dropout",float(cfg.dropout)),("lionai.num_classes",cfg.num_classes),("lionai.labels",labels),
      ("lionai.training_examples",int(ck.get("training_examples",0))),("lionai.checkpoint_epoch",int(ck.get("epoch",0))),
      ("lionai.validation_accuracy",float(ck.get("validation_accuracy",0))),("lionai.validation_macro_f1",float(ck.get("validation_macro_f1",0))),
      ("lionai.selection_metric",str(ck.get("selection_metric","unknown"))),("lionai.tensor_dtype","F32")]
    tensors=[]
    for key,val in model.state_dict().items():
        t=val.detach().cpu().float().contiguous(); tensors.append(("lionai."+key,tuple(reversed(t.shape)),t.numpy().astype("<f4",copy=False).tobytes()))
    infos=bytearray(); data=bytearray()
    for key,shape,raw in tensors:
        data.extend(b"\0"*((-len(data))%32)); off=len(data); infos.extend(s(key)); infos.extend(struct.pack("<I",len(shape)))
        infos.extend(struct.pack("<"+"Q"*len(shape),*shape)); infos.extend(struct.pack("<IQ",0,off)); data.extend(raw)
    header=bytearray(b"GGUF"+struct.pack("<IQQ",3,len(tensors),len(metadata)))
    header.extend(b"".join(m(k,v) for k,v in metadata)); header.extend(infos); header.extend(b"\0"*((-len(header))%32))
    output_path.parent.mkdir(parents=True,exist_ok=True); tmp=output_path.with_name(output_path.name+".tmp")
    tmp.write_bytes(header+data); tmp.replace(output_path)
    reader=GGUFReader(output_path); restored,labs,restored_cfg,_=load_gguf_model(output_path,"cpu")
    if len(reader.tensors)!=len(model.state_dict()) or labs!=labels or restored_cfg!=cfg: raise RuntimeError("GGUF structural/config validation failed")
    for key,val in model.state_dict().items(): torch.testing.assert_close(restored.state_dict()[key],val,rtol=0,atol=0)
    ids=torch.tensor([[1,2,3,257,101,102]],dtype=torch.long); mask=torch.ones_like(ids,dtype=torch.bool)
    with torch.inference_mode(): torch.testing.assert_close(restored(ids,mask),model(ids,mask),rtol=0,atol=0)
    report={"model":name,"format":"GGUF v3 / F32 / custom LionAI architecture","checkpoint_epoch":int(ck.get("epoch",0)),
      "training_examples":int(ck.get("training_examples",0)),"validation_accuracy":float(ck.get("validation_accuracy",0)),
      "validation_macro_f1":float(ck.get("validation_macro_f1",0)),"tensor_count":len(reader.tensors),
      "metadata_count":len(reader.metadata),"tensor_round_trip_exact":True,"inference_logits_exact":True,
      "size_bytes":output_path.stat().st_size,"sha256":hashlib.sha256(output_path.read_bytes()).hexdigest(),
      "release_status":"candidate_only_pending_independent_benchmark_acceptance"}
    output_path.with_suffix(output_path.suffix+".json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return report
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--checkpoint",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(); print(json.dumps(export(a.checkpoint,a.output),ensure_ascii=False,indent=2))
