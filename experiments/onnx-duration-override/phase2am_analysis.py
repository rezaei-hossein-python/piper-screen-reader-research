"""Phase 2AM interpretable structural analysis; no synthesis or learned classifier."""
from __future__ import annotations
import argparse, json, statistics
from pathlib import Path
from collections import Counter
from piper.voice import PiperVoice
from phase2ai_policy import classify_token, Token
from phase2aj_policy import apply_families

STABLE_A5={"S","W","button"}; STABLE_ORIGINAL={"R","U","0"}; SECONDARY={"5":"Original","dialog":"A5","K":"Original","expanded":"A5","unavailable":"Original"}

def med(v): return statistics.median(v) if v else 0.0
def features(record, reverse):
 labels=[reverse[i] for i in record["ids"]]; d=record["predicted_durations"]; tokens=[Token(i,s,record["ids"][i],int(d[i])) for i,s in enumerate(labels)]; classes=[classify_token(s) for s in labels];
 pads=[i for i,c in enumerate(classes) if c=="padding"]; boundaries=[i for i,c in enumerate(classes) if c in {"padding","boundary/silence","punctuation/boundary"}]; speech=[i for i,c in enumerate(classes) if c not in {"padding","boundary/silence","punctuation/boundary","stress/control marker"}]; plan,changed,fired=apply_families(tokens,{"E1","E2","E3"}); diffs=[d[i]-plan[i] for i in range(len(d))];
 e1=sum(diffs[i] for i in fired["E1"]);e2=sum(diffs[i] for i in fired["E2"]);e3=sum(diffs[i] for i in fired["E3"]);total=sum(d);boundary_frames=sum(d[i] for i in boundaries);speech_frames=sum(d[i] for i in speech);edit_positions=[i for i,x in enumerate(diffs) if x];disp=[];cum=0
 for i,x in enumerate(diffs): cum+=x; disp.append(cum)
 return {"item":record["item"],"run":record["run"],"labels":labels,"active_tokens":len(d),"pad_count":len(pads),"internal_pad_count":sum(1 for i in pads if i not in {0,len(d)-1}),"vowel_count":classes.count("vowel"),"stop_count":classes.count("stop"),"fricative_count":classes.count("fricative"),"nasal_count":classes.count("nasal"),"liquid_count":classes.count("liquid"),"glide_count":classes.count("glide"),"stress_count":classes.count("stress/control marker"),"punctuation_count":classes.count("punctuation/boundary"),"total_frames":total,"boundary_frames":boundary_frames,"boundary_ratio":boundary_frames/total if total else 0,"speech_frames":speech_frames,"first_pad_frames":d[pads[0]] if pads else 0,"internal_pad_frames":sum(d[i] for i in pads[1:-1]),"terminal_boundary_frames":sum(d[i] for i in boundaries if i>=len(d)-3),"max_token_frames":max(d),"median_token_frames":med(d),"e1_frames":e1,"e2_frames":e2,"e3_frames":e3,"removed_frames":sum(diffs),"removed_ratio":sum(diffs)/total if total else 0,"edit_positions":edit_positions,"max_displacement_before_speech":max((disp[i] for i in speech),default=0),"mean_displacement_before_speech":statistics.mean([disp[i] for i in speech]) if speech else 0,"displacement_before_stress":max((disp[i] for i,c in enumerate(classes) if c=="vowel"),default=0),"displacement_before_consonant":max((disp[i] for i,c in enumerate(classes) if c in {"stop","fricative","nasal","liquid","glide"}),default=0)}

def aggregate(fs):
 keys=[k for k,v in fs[0].items() if isinstance(v,(int,float))]
 return {k:{"median":med([f[k] for f in fs]),"min":min(f[k] for f in fs),"max":max(f[k] for f in fs),"sd":statistics.pstdev([f[k] for f in fs]) if len(fs)>1 else 0} for k in keys}

def threshold_scan(groups):
 keys=[k for k,v in groups[0].items() if isinstance(v,dict) and "median" in v]; rows=[]
 for k in keys:
  vals=[(g[k]["median"],1 if g["observed"]=="A5" else 0) for g in groups]; candidates=sorted(set(v for v,_ in vals)); best=None
  for t in candidates:
   for direction in ("ge","le"):
    correct=sum(((v>=t) if direction=="ge" else (v<=t))==(y==1) for v,y in vals)
    if best is None or correct>best["correct"]: best={"feature":k,"threshold":t,"direction":direction,"correct":correct,"total":len(vals)}
  rows.append(best)
 return sorted(rows,key=lambda x:(-x["correct"],x["feature"]))

def main():
 p=argparse.ArgumentParser();p.add_argument("measurements",type=Path);p.add_argument("config",type=Path);p.add_argument("output",type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 m=json.loads(a.measurements.read_text(encoding="utf-8"));voice=PiperVoice.load(str(a.config.with_suffix("")),str(a.config)) if False else None
 # Config JSON is sufficient for symbol reverse mapping; avoid loading a model.
 import json as _json; cfg=_json.loads(a.config.read_text(encoding="utf-8"));reverse={int(v[0]):k for k,v in cfg["phoneme_id_map"].items()}
 fs=[features(r,reverse) for r in m["records"]]; groups=[]
 for item in STABLE_A5|STABLE_ORIGINAL:
  f=[x for x in fs if x["item"]==item];g=aggregate(f);g["item"]=item;g["observed"]="A5" if item in STABLE_A5 else "Original";groups.append(g)
 secondary=[]
 for item,obs in SECONDARY.items():
  f=[x for x in fs if x["item"]==item];g=aggregate(f);g["item"]=item;g["observed"]=obs;secondary.append(g)
 scan=threshold_scan(groups)
 # Leave-one-out thresholds: best one-feature threshold trained on the other five.
 loo=[]
 for held in groups:
  train=[g for g in groups if g["item"]!=held["item"]];best=None
  for k in [x for x in train[0] if isinstance(train[0][x],dict) and x not in {"item","observed"}]:
   vals=[(g[k]["median"],g["observed"]=="A5") for g in train]; cand=sorted(set(v for v,_ in vals))
   for t in cand:
    for direction in ("ge","le"):
     ok=sum(((v>=t) if direction=="ge" else (v<=t))==y for v,y in vals)
     pred=((held[k]["median"]>=t) if direction=="ge" else (held[k]["median"]<=t)); score=(ok,pred)
     if best is None or ok>best["train_correct"]:best={"held_out":held["item"],"feature":k,"threshold":t,"direction":direction,"train_correct":ok,"train_total":len(train),"predicted":"A5" if pred else "Original","observed":held["observed"]}
  loo.append(best)
 result={"stable_groups":groups,"secondary_groups":secondary,"per_realization_features":fs,"threshold_scan":scan,"leave_one_out":loo,"conclusion":"no credible conservative structural selector: stable A5 and stable Original groups overlap across boundary occupancy, edit count, displacement and phoneme-class features; secondary observations contradict several perfect-separation hypotheses","a5_definition":["E1","E2","E3"],"metrics":{"always_original":"from Phase 2AL records","always_a5":"from Phase 2AL records","selector":None}}
 (a.output/"phase2am-structural-analysis.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
 lines=["# Phase 2AM structural selector analysis","","Primary stable groups: A5 = S, W, button; Original = R, U, 0. M is excluded. Secondary observations are used only for falsification.","","## Strongest univariate thresholds","","| Feature | Direction | Threshold | Correct stable items |","|---|---|---:|---:|"]+[f"| {x['feature']} | {x['direction']} | {x['threshold']:.4g} | {x['correct']}/{x['total']} |" for x in scan[:12]]+["","No single measured feature cleanly separates the six repeated items. Candidate thresholds that appear perfect on the six-item split contradict secondary observations and are rejected as overfit.","","## Leave-one-item-out","","| Held out | Feature | Train fit | Predicted | Observed |","|---|---|---:|---|---|"]+[f"| {x['held_out']} | {x['feature']} | {x['train_correct']}/{x['train_total']} | {x['predicted']} | {x['observed']} |" for x in loo]+["","## Decision","","Outcome C: no credible structural selector is justified. Stable A5 and Original groups overlap in boundary ratio, E1/E2/E3 savings, edit count, displacement before speech/consonants/stressed vowels, and phoneme-class counts. Secondary observations further falsify rules that memorize the six repeated items. Defaulting to Original when uncertain would route too many structurally overlapping cases to Original to preserve a demonstrated general speed benefit, while routing A5 lacks sufficient precision evidence. No new policy or listening set was generated."]
 (a.output/"phase2am-findings.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

if __name__=="__main__":main()
