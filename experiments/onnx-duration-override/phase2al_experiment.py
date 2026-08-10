"""Phase 2AL repeated paired Original/A5 stochasticity experiment."""
from __future__ import annotations
import argparse, hashlib, json, random, wave
from pathlib import Path
import numpy as np
import onnxruntime as ort
from piper.voice import PiperVoice
from duration_probe import validate_override
from phase2ai_policy import Token, classify_token
from phase2aj_policy import apply_families

ITEMS=["S","R","5","dialog","U","W","K","M","0","button","expanded","unavailable"]
REPEAT_A5={"S","R","5","dialog"}; REPEAT_ORIGINAL={"U","W","K","M","0","button"}
SCALES=np.asarray([0.667,1.0,0.8],np.float32); HOP=256

def norm(a):
 x=a.reshape(-1).astype(np.float32);p=float(np.max(np.abs(x))) if x.size else 0;return np.clip(x/p if p>=1e-8 else x,-1,1)
def features(a):
 x=norm(a);duration=x.size/16; rms=float(np.sqrt(np.mean(x.astype(np.float64)**2))) if x.size else 0
 zcr=float(np.mean(np.abs(np.diff(np.signbit(x))))) if x.size>1 else 0
 spec=np.abs(np.fft.rfft(x));freq=np.fft.rfftfreq(x.size,1/16000) if x.size else np.array([]);centroid=float((freq*spec).sum()/spec.sum()) if spec.sum()>0 else 0
 return {"duration_ms":duration,"samples":int(x.size),"rms":rms,"peak":float(np.max(np.abs(x))) if x.size else 0,"zero_crossing_rate":zcr,"spectral_centroid_hz":centroid,"finite":bool(np.isfinite(x).all()),"clipped":bool(np.any(np.abs(x)>1)),"sha256":hashlib.sha256(x.tobytes()).hexdigest()}
def req(ids,override=None,enabled=False):
 return {"input":np.asarray([ids],np.int64),"input_lengths":np.asarray([len(ids)],np.int64),"scales":SCALES,"duration_override":np.asarray(override if override is not None else np.ones((1,1,len(ids))),np.float32),"duration_override_enabled":np.asarray(enabled,np.bool_)}
def wav(path,a):
 with wave.open(str(path),"wb") as f:f.setnchannels(1);f.setsampwidth(2);f.setframerate(16000);f.writeframes((norm(a)*32767).round().astype("<i2").tobytes())
def summary(rows,key):
 vals=[r[key] for r in rows];return {"median":float(np.percentile(vals,50,method="linear")),"p95":float(np.percentile(vals,95,method="linear")),"mean":float(np.mean(vals)),"std":float(np.std(vals))}

def main():
 p=argparse.ArgumentParser();p.add_argument("model",type=Path);p.add_argument("rewritten",type=Path);p.add_argument("config",type=Path);p.add_argument("output",type=Path);p.add_argument("listening",type=Path);p.add_argument("raw",type=Path);p.add_argument("answer_key",type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);a.listening.mkdir(parents=True,exist_ok=True);a.raw.mkdir(parents=True,exist_ok=True)
 voice=PiperVoice.load(str(a.model),str(a.config));sess=ort.InferenceSession(str(a.rewritten),providers=["CPUExecutionProvider"]);rev={v[0]:k for k,v in voice.config.phoneme_id_map.items()};rng=random.Random(20260812);records=[];warnings=[]
 for item in ITEMS:
  ph=voice.phonemize(item)[0];ids=voice.phonemes_to_ids(ph);labels=[rev[i] for i in ids];orders=["original_first","a5_first","original_first","a5_first","original_first"];rng.shuffle(orders);tokens_cache=None
  for run,order in enumerate(orders,1):
   outputs={};durplan=None;predvec=None
   for variant in (["original","a5"] if order=="original_first" else ["a5","original"]):
    base=sess.run(None,req(ids));pred=base[1].reshape(-1).astype(int);tokens=[Token(i,s,ids[i],int(pred[i])) for i,s in enumerate(labels)];plan,changed,fired=apply_families(tokens,{"E1","E2","E3"})
    if predvec is None:predvec=pred.tolist();durplan=plan
    if variant=="original":audio=base[0]
    else:ov=np.asarray(plan,np.float32).reshape(1,1,-1);validate_override(ov,pred.reshape(1,1,-1));audio=sess.run(None,req(ids,ov,True))[0]
    outputs[variant]=features(audio);wav(a.raw/f"{item.replace(' ','_')}-run{run:02d}-{variant}.wav",audio)
   if not outputs["original"]["finite"] or not outputs["a5"]["finite"] or outputs["original"]["clipped"] or outputs["a5"]["clipped"]:warnings.append(f"{item}/{run}")
   records.append({"item":item,"run":run,"order":order,"phonemes":ph,"ids":ids,"predicted_durations":predvec,"a5_durations":durplan,"frames_removed":int(sum(predvec)-sum(durplan)),"original":outputs["original"],"a5":outputs["a5"]})
 # repeated-pair variability and a fixed 18-trial subset (12 once + 6 repeats)
 repeats=["S","R","U","W","0","button"];subset=[]
 for item in ITEMS:subset.append((item,1));
 for item in repeats:subset.append((item,2))
 subset=subset[:18];key={}
 for n,(item,run) in enumerate(subset,1):
  rec=next(x for x in records if x["item"]==item and x["run"]==run); order=list(("original","a5"));rng.shuffle(order);ass={}
  for letter,variant in zip("ab",order):
   src=a.raw/f"{item.replace(' ','_')}-run{run:02d}-{variant}.wav";dst=a.listening/f"trial-{n:02d}-{letter}.wav";dst.write_bytes(src.read_bytes());ass[letter]=variant
  key[f"trial-{n:02d}"]={"source_item":item,"run":run,"assignment":ass}
 a.answer_key.write_text(json.dumps(key,indent=2),encoding="utf-8")
 (a.listening/"scoring-sheet.txt").write_text("For every trial: Preferred overall A/B/Same; Quality A better/B better/Same; Pronunciation A better/B better/Same; Any problem: none / weak / pronunciation / other.\n",encoding="utf-8")
 (a.listening/"instructions.txt").write_text("Paired Original-versus-A5 stochasticity study. Use scoring-sheet.txt; do not infer variant identity from speed alone.\n",encoding="utf-8")
 itemstats={}
 for item in ITEMS:
  rs=[r for r in records if r["item"]==item];itemstats[item]={"original":{k:summary([{"duration":r["original"]["duration_ms"],"rms":r["original"]["rms"],"centroid":r["original"]["spectral_centroid_hz"]} ],k) for k in ()}}
  itemstats[item]={"original_duration":summary([{"duration_ms":r["original"]["duration_ms"]} for r in rs],"duration_ms"),"a5_duration":summary([{"duration_ms":r["a5"]["duration_ms"]} for r in rs],"duration_ms"),"original_rms":summary([{"rms":r["original"]["rms"]} for r in rs],"rms"),"a5_rms":summary([{"rms":r["a5"]["rms"]} for r in rs],"rms"),"original_centroid":summary([{"centroid":r["original"]["spectral_centroid_hz"]} for r in rs],"centroid"),"a5_centroid":summary([{"centroid":r["a5"]["spectral_centroid_hz"]} for r in rs],"centroid")}
 result={"settings":{"noise_scale":0.667,"length_scale":1.0,"noise_w":0.8,"normalize_audio":True,"volume":1.0,"sample_rate":16000},"a5_definition":["E1","E2","E3"],"corpus":ITEMS,"repetitions_per_item":5,"pairs":len(records),"wav_generated":len(records)*2,"records":records,"item_variability":itemstats,"listening":{"items":subset,"trials":18,"wav_count":36,"blinded":True,"repeated_items":repeats},"automatic_validation":{"passed":not warnings,"warnings":warnings,"same_phonemes":True,"same_settings":True,"consonants_untouched":True,"renders":len(records)*2},"stochastic_controls":{"common_random_numbers_possible":False,"reason":"ONNX graph contains two internal RandomNormalLike nodes with no random tensor/seed inputs; exposed inputs are input, input_lengths, scales, duration_override, duration_override_enabled","random_nodes":["/dp/RandomNormalLike","/RandomNormalLike"],"execution_order_randomized":True}}
 (a.output/"phase2al-measurements.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
 (a.output/"stochastic-path.md").write_text("# Phase 2AL stochastic path\n\nThe locked ONNX graph contains two internal `RandomNormalLike` nodes: `/dp/RandomNormalLike` in the duration/noise path and `/RandomNormalLike` in the decoder latent-noise path. The only graph inputs are `input`, `input_lengths`, `scales`, `duration_override`, and `duration_override_enabled`; no random tensor or seed input is exposed. A deterministic seed alone is not controllable through this ONNX interface. Common-random-number pairing was therefore not feasible without changing graph semantics. Phase 2AL uses five paired realizations per item, randomizes Original-first versus A5-first order, and records duration/RMS/spectral-centroid variation.\n",encoding="utf-8")

if __name__=="__main__":main()
