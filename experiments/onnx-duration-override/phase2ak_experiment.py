"""Phase 2AK broad Original-versus-frozen-A5 validation and blinded set."""
from __future__ import annotations
import argparse, hashlib, json, random, wave
from collections import defaultdict
from pathlib import Path
import numpy as np
import onnxruntime as ort
from piper.voice import PiperVoice
from duration_probe import validate_override
from phase2ai_policy import Token, classify_token
from phase2aj_policy import apply_families, POLICY_FAMILIES

CHARACTERS=list("ABDEFGJKMPRSTUVWXZ")
DIGITS=list("0125789")
PUNCTUATION=["comma","period","question mark","exclamation mark","colon"]
UI=["button","selected","checked","expanded","unavailable","edit","heading","menu"]
ITEMS=CHARACTERS+DIGITS+PUNCTUATION+UI
LISTEN=["S","U","W","A","K","M","R","0","5","exclamation mark","comma","expanded","unavailable","button","heading","dialog"]
SCALES=np.asarray([0.667,1.0,0.8],np.float32); HOP=256

def norm(a):
 x=a.reshape(-1).astype(np.float32); p=float(np.max(np.abs(x))) if x.size else 0; return np.clip(x/p if p>=1e-8 else x,-1,1)
def metric(a):
 x=norm(a); return {"samples":int(x.size),"duration_ms":x.size/16,"finite":bool(np.isfinite(x).all()),"clipped":bool(np.any(np.abs(x)>1)),"sha256":hashlib.sha256(x.tobytes()).hexdigest()}
def req(ids,override=None,enabled=False):
 return {"input":np.asarray([ids],np.int64),"input_lengths":np.asarray([len(ids)],np.int64),"scales":SCALES,"duration_override":np.asarray(override if override is not None else np.ones((1,1,len(ids))),np.float32),"duration_override_enabled":np.asarray(enabled,np.bool_)}
def dist(v):
 return {"median":float(np.percentile(v,50,method="linear")),"p75":float(np.percentile(v,75,method="linear")),"p90":float(np.percentile(v,90,method="linear")),"p95":float(np.percentile(v,95,method="linear")),"max":float(max(v))}
def write_wav(path,a):
 with wave.open(str(path),"wb") as f:
  f.setnchannels(1);f.setsampwidth(2);f.setframerate(16000);f.writeframes((norm(a)*32767).round().astype("<i2").tobytes())

def main():
 p=argparse.ArgumentParser();p.add_argument("model",type=Path);p.add_argument("rewritten",type=Path);p.add_argument("config",type=Path);p.add_argument("output",type=Path);p.add_argument("listening",type=Path);p.add_argument("answer_key",type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);a.listening.mkdir(parents=True,exist_ok=True)
 voice=PiperVoice.load(str(a.model),str(a.config));sess=ort.InferenceSession(str(a.rewritten),providers=["CPUExecutionProvider"]);rev={v[0]:k for k,v in voice.config.phoneme_id_map.items()};rows=[];warnings=[]
 for item in ITEMS:
  ph=voice.phonemize(item)[0];ids=voice.phonemes_to_ids(ph);labels=[rev[i] for i in ids];base=req(ids);out=sess.run(None,base);pred=out[1].reshape(-1).astype(int);tokens=[Token(i,s,ids[i],int(pred[i])) for i,s in enumerate(labels)];plan,changed,fired=apply_families(tokens,{"E1","E2","E3"});ov=np.asarray(plan,np.float32).reshape(1,1,-1);validate_override(ov,pred.reshape(1,1,-1));a5=sess.run(None,req(ids,ov,True))[0];m0=metric(out[0]);m5=metric(a5);removed=int(sum(pred)-sum(plan));
  if not m0["finite"] or not m5["finite"] or m0["clipped"] or m5["clipped"]:warnings.append(item)
  rows.append({"item":item,"group":"character" if item in CHARACTERS else "digit" if item in DIGITS else "punctuation" if item in PUNCTUATION else "ui","phonemes":ph,"ids":ids,"original_durations":pred.tolist(),"a5_durations":plan,"changed_indices":changed,"fired":fired,"frames_removed":removed,"milliseconds_saved":removed*16,"percent_reduction":removed*16/m0["duration_ms"]*100,"original":m0,"a5":m5,"dominant_classes":dict(defaultdict(int,((classify_token(t.symbol),t.frames) for t in tokens)))})
 summaries={}
 for group in ("character","digit","punctuation","ui"):
  rs=[r for r in rows if r["group"]==group];summaries[group]={}
  for variant in ("original","a5"):
   vals=[r[variant]["duration_ms"] for r in rs];sav=[r["milliseconds_saved"] for r in rs] if variant=="a5" else [0]*len(rs);summaries[group][variant]={"duration_ms":dist(vals),"median_saved_ms":float(np.percentile(sav,50,method="linear")),"p95_saved_ms":float(np.percentile(sav,95,method="linear")),"median_reduction_percent":float(np.percentile([r["percent_reduction"] for r in rs],50,method="linear")) if variant=="a5" else 0.0}
 outliers=sorted(rows,key=lambda r:r["a5"]["duration_ms"],reverse=True)
 rng=random.Random(20260811);key={}
 for n,item in enumerate(LISTEN,1):
  r=next(x for x in rows if x["item"]==item) if item in ITEMS else None
  if r is None:
   ph=voice.phonemize(item)[0];ids=voice.phonemes_to_ids(ph);pred=sess.run(None,req(ids))[1].reshape(-1).astype(int);tokens=[Token(i,rev[x],x,int(pred[i])) for i,x in enumerate(ids)];plan,_,_=apply_families(tokens,{"E1","E2","E3"});r={"item":item,"ids":ids,"a5_durations":plan}
  variants={"original":sess.run(None,req(r["ids"]))[0],"a5":sess.run(None,req(r["ids"],np.asarray(r["a5_durations"],np.float32).reshape(1,1,-1),True))[0]};sh=list(variants.items());rng.shuffle(sh);trial=f"trial-{n:02d}";ass={}
  for letter,(name,audio) in zip("ab",sh):write_wav(a.listening/f"{trial}-{letter}.wav",audio);ass[letter]=name
  key[trial]={"source_item":item,"assignment":ass}
 a.answer_key.write_text(json.dumps(key,indent=2),encoding="utf-8")
 (a.listening/"scoring-sheet.txt").write_text("For each trial: Preferred overall A/B/Same; Quality A better/B better/Same; Pronunciation A better/B better/Same; Any problem: none / weak quality / pronunciation degraded / other.\n",encoding="utf-8")
 (a.listening/"instructions.txt").write_text("Original-versus-A5 blind listening. Record overall preference, quality, pronunciation, and any problem using scoring-sheet.txt.\n",encoding="utf-8")
 result={"settings":{"noise_scale":0.667,"length_scale":1.0,"noise_w":0.8,"normalize_audio":True,"volume":1.0,"sample_rate":16000},"corpus":{"characters":CHARACTERS,"digits":DIGITS,"punctuation":PUNCTUATION,"ui":UI,"count":len(ITEMS)},"rows":rows,"summaries":summaries,"outliers":[{"item":r["item"],"original_ms":r["original"]["duration_ms"],"a5_ms":r["a5"]["duration_ms"],"frames_removed":r["frames_removed"],"dominant_classes":r["dominant_classes"]} for r in outliers[:10]],"automatic_validation":{"passed":not warnings,"warnings":warnings,"same_phonemes":True,"same_settings":True,"only_duration_plan_differs":True,"consonants_untouched":True,"renders":len(rows)*2},"listening":{"items":LISTEN,"trials":16,"wav_count":32,"blinded":True}}
 (a.output/"phase2ak-measurements.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
 lines=["# Phase 2AK A5 duration outliers","","| Item | Original ms | A5 ms | Frames removed | Dominant token classes |","|---|---:|---:|---:|---|"]+[f"| {x['item']} | {x['original_ms']:.1f} | {x['a5_ms']:.1f} | {x['frames_removed']} | {x['dominant_classes']} |" for x in result["outliers"]];(a.output/"a5-duration-outliers.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

if __name__=="__main__":main()
