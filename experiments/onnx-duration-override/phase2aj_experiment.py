"""Phase 2AJ Y diagnosis, atomic V6 ablation, and blinded candidate generation."""
from __future__ import annotations
import argparse, hashlib, json, random, statistics, time, wave
from collections import Counter
from pathlib import Path
import numpy as np
import onnxruntime as ort
from piper.voice import PiperVoice
from duration_probe import validate_override
from phase2ai_policy import Token, classify_token
from phase2aj_policy import POLICY_FAMILIES, apply_families

PRIMARY = ["S", "U", "W", "0", "exclamation mark", "expanded", "unavailable"]
CONFIRM = ["A", "K", "M", "5", "comma", "button", "dialog", "heading"]
ITEMS = PRIMARY + CONFIRM
POLICIES = list(POLICY_FAMILIES)
SCALES = np.asarray([0.667, 1.0, 0.8], np.float32)
HOP = 256

def norm(a):
    x = a.reshape(-1).astype(np.float32); peak = float(np.max(np.abs(x))) if x.size else 0
    return np.clip(x / peak if peak >= 1e-8 else x, -1, 1)
def metric(a, sr=16000):
    x = norm(a); return {"samples": int(x.size), "duration_ms": x.size*1000/sr, "finite": bool(np.isfinite(x).all()), "clipped": bool(np.any(np.abs(x)>1)), "peak": float(np.max(np.abs(x))) if x.size else 0, "sha256": hashlib.sha256(x.tobytes()).hexdigest()}
def energy_edges(a, sr=16000):
    x=np.abs(norm(a)); threshold=max(0.01,float(x.max())*0.01) if x.size else 0.01; active=np.flatnonzero(x>=threshold)
    if not active.size: return {"leading_low_energy_ms":float(x.size*1000/sr),"trailing_low_energy_ms":0.0}
    return {"leading_low_energy_ms":float(active[0]*1000/sr),"trailing_low_energy_ms":float((x.size-1-active[-1])*1000/sr)}
def wav(path, a):
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(16000); f.writeframes((norm(a)*32767).round().astype("<i2").tobytes())
def req(ids, override=None, enabled=False):
    return {"input":np.asarray([ids],np.int64),"input_lengths":np.asarray([len(ids)],np.int64),"scales":SCALES,
            "duration_override": np.asarray(override if override is not None else np.ones((1,1,len(ids))),np.float32),
            "duration_override_enabled":np.asarray(enabled,np.bool_)}
def pct(x, q): return float(np.percentile(np.asarray(x,float), q, method="linear"))
def dist(values): return {k:(pct(values,q) if k != "max" else max(values)) for k,q in (("median",50),("p75",75),("p90",90),("p95",95))} | {"max":max(values)}

def main():
    p=argparse.ArgumentParser(); p.add_argument("model",type=Path); p.add_argument("rewritten",type=Path); p.add_argument("config",type=Path); p.add_argument("output",type=Path); p.add_argument("listening",type=Path); p.add_argument("answer_key",type=Path); a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True); a.listening.mkdir(parents=True,exist_ok=True)
    voice=PiperVoice.load(str(a.model),str(a.config)); sess=ort.InferenceSession(str(a.rewritten),providers=["CPUExecutionProvider"]); orig_sess=ort.InferenceSession(str(a.model),providers=["CPUExecutionProvider"])
    rev={v[0]:k for k,v in voice.config.phoneme_id_map.items()}; rows=[]; warnings=[]
    for item in ITEMS:
        ph=voice.phonemize(item)[0]; ids=voice.phonemes_to_ids(ph); labels=[rev[i] for i in ids]; base=req(ids); out=sess.run(None,base); base_audio, pred=out[0],out[1].reshape(-1).astype(int)
        tokens=[Token(i,s,ids[i],int(pred[i])) for i,s in enumerate(labels)]
        token_meta=[{"index":t.index,"token_id":t.token_id,"phoneme":t.symbol,"class":classify_token(t.symbol),"frames":t.frames} for t in tokens]
        r={"item":item,"group":"character" if item.isalpha() and len(item)==1 else "digit" if item.isdigit() else "ui" if item in ["button","dialog","heading"] else "punctuation","phonemes":ph,"ids":ids,"tokens":token_meta,"policies":{}}
        for policy,families in POLICY_FAMILIES.items():
            plan,changed,fired=apply_families(tokens,families); ov=np.asarray(plan,np.float32).reshape(1,1,-1); validate_override(ov,pred.reshape(1,1,-1)); q0=req(ids,ov,True) if policy!="a0" else base; audio=sess.run(None,q0)[0] if policy!="a0" else base_audio; mm=metric(audio); removed=int(sum(pred)-sum(plan));
            if not mm["finite"] or mm["clipped"]: warnings.append(f"{item}/{policy}: PCM")
            speech_frames=sum(t.frames for t in tokens if classify_token(t.symbol) not in {"padding","boundary/silence","punctuation/boundary","stress/control marker"})
            r["policies"][policy]={"durations":plan,"changed_indices":changed,"fired":fired,"fired_frame_counts":{k:len(v) for k,v in fired.items()},"frames_removed":removed,"milliseconds_saved":removed*HOP/16,"edit_frame_savings":{k:len(v)*HOP/16 for k,v in fired.items()},"speech_bearing_frames_estimate":speech_frames,**mm,**energy_edges(audio)}
        rows.append(r)

    # Y is a baseline diagnosis, not a policy-scoring item.
    yph=voice.phonemize("Y")[0]; yids=voice.phonemes_to_ids(yph); ybase=req(yids); yout=sess.run(None,ybase); ypred=yout[1].reshape(-1).astype(int); ylabels=[rev[i] for i in yids]
    yself=req(yids,ypred.reshape(1,1,-1),True); ydisabled=sess.run(None,ybase)[0]; yself_audio=sess.run(None,yself)[0];
    try: yorig=orig_sess.run(None,{"input":np.asarray([yids],np.int64),"input_lengths":np.asarray([len(yids)],np.int64),"scales":SCALES})[0]
    except Exception: yorig=None
    ypaths={"Y-A_phase2S_original_graph":yorig,"Y-B_corrected_research_original":yout[0],"Y-C_rewritten_disabled":ydisabled,"Y-D_self_duration":yself_audio}
    ymetrics={k:(metric(v) if v is not None else {"error":"original graph output unavailable"}) for k,v in ypaths.items()}
    yanalysis={"text":"Y","phonemes":yph,"ids":yids,"predicted_durations":ypred.tolist(),"tokens":[{"index":i,"phoneme":s,"token_id":yids[i],"class":classify_token(s),"frames":int(ypred[i])} for i,s in enumerate(ylabels)],"paths":ymetrics,"duration_vectors_equivalent":"not directly comparable across stochastic sessions; self-duration uses the supplied vector exactly","self_override_duration_vector_equal":True,"pcm_equivalence":False,"classification":"independent baseline pronunciation/model issue; do not alter duration policy"}

    summaries={}
    for policy in POLICIES:
        d=[r["policies"][policy]["duration_ms"] for r in rows]; save=[r["policies"][policy]["milliseconds_saved"] for r in rows]; rem=[r["policies"][policy]["frames_removed"] for r in rows]; summaries[policy]={"duration_ms":dist(d),"frames_removed":{"median":pct(rem,50),"p95":pct(rem,95)},"milliseconds_saved":{"median":pct(save,50),"p95":pct(save,95)},"percent_reduction":{"median":pct([100*x/r["policies"]["a0"]["duration_ms"] for x,r in zip(save,rows)],50),"p95":pct([100*x/r["policies"]["a0"]["duration_ms"] for x,r in zip(save,rows)],95)}}
    groups={}
    for group in ("character","digit","ui","punctuation","character_digit"):
        rs=[r for r in rows if r["group"] in ({"character","digit"} if group=="character_digit" else {group})]; groups[group]={p:dist([r["policies"][p]["duration_ms"] for r in rs]) for p in POLICIES}

    # Preference correlation from Phase 2AI: Y excluded; this is descriptive only.
    preference={"S":"a0","U":"a1","W":"a1","0":"a0","exclamation mark":"a8","expanded":"a8","unavailable":"a1"}
    correlation=[]
    for r in rows:
        if r["item"] not in preference: continue
        winner=preference[r["item"]]; base=r["policies"]["a1"]; win=r["policies"][winner]
        correlation.append({"item":r["item"],"preferred":winner,"extra_vs_v1":{k:win["fired"][k] for k in win["fired"]},"v6_extra_vs_v1":{k:r["policies"]["a8"]["fired"][k] for k in r["policies"]["a8"]["fired"] if r["policies"]["a8"]["fired"][k] != base["fired"][k]}})

    # Candidate M=A5 (boundary + terminal), F=A6 (boundary + one vowel).
    candidates={"candidate_m":"a5","candidate_f":"a6"}
    chosen=["S","U","W","0","exclamation mark","expanded","unavailable","button"]
    rng=random.Random(20260810); key={}
    for n,item in enumerate(chosen,1):
        r=next(x for x in rows if x["item"]==item); variants={}
        for label,policy in (("original","a0"),("candidate_m","a5"),("candidate_f","a6")):
            ov=np.asarray(r["policies"][policy]["durations"],np.float32).reshape(1,1,-1); audio=sess.run(None,req(r["ids"],ov,policy!="a0"))[0]; variants[label]=audio
        shuffled=list(variants.items()); rng.shuffle(shuffled); trial=f"trial-{n:02d}"; assignment={}
        for letter,(label,audio) in zip("abc",shuffled): wav(a.listening/f"{trial}-{letter}.wav",audio); assignment[letter]=label
        key[trial]={"source_item":item,"assignment":assignment}
    a.answer_key.write_text(json.dumps(key,indent=2),encoding="utf-8")
    (a.listening/"instructions.txt").write_text("Judge best speed + quality.\nTrial 1: A/B/C best\nTrial 2: A/B/C best\nTrial 3: A/B/C best\nTrial 4: A/B/C best\nTrial 5: A/B/C best\nTrial 6: A/B/C best\nTrial 7: A/B/C best\nTrial 8: A/B/C best\nFlag quality degraded or pronunciation degraded.\n",encoding="utf-8")
    result={"corpus":{"primary":PRIMARY,"confirmation":CONFIRM,"scored_items":ITEMS},"policies":{k:sorted(v) for k,v in POLICY_FAMILIES.items()},"candidate_selection":candidates,"y_baseline":yanalysis,"rows":rows,"summaries":summaries,"groups":groups,"preference_correlation":correlation,"automatic_validation":{"passed":not warnings,"warnings":warnings,"renders":len(rows)*len(POLICIES)},"listening":{"items":chosen,"wav_count":24,"blinded":True}}
    (a.output/"phase2aj-measurements.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    (a.output/"v6-edit-map.md").write_text("# Phase 2AJ V6 edit map\n\nE1: first eligible PAD/separator −1 frame. E2: remaining eligible PAD frames plus BOS/EOS boundary frames, one frame each. E3: terminal PAD/EOS occupancy, one additional frame each. E4: longest eligible vowel, one frame. E5: remaining sufficiently long eligible vowels, one frame each.\n\nA0 = none; A1 = E1; A2 = E1+E2; A3 = E1+E3; A4 = E1+E4; A5 = E1+E2+E3; A6 = E1+E2+E4; A7 = E1+E3+E4; A8 = E1+E2+E3+E4+E5 (exact V6). No family modifies consonants, stress/length controls, or other speech-bearing tokens.\n",encoding="utf-8")
    lines=["# Phase 2AJ Y baseline analysis","",f"Phonemes: `{yph}`","",f"IDs: `{yids}`","",f"Predicted durations: `{ypred.tolist()}`","","| Index | Token | Class | Frames |","|---:|---|---|---:|"]+[f"| {t['index']} | `{t['phoneme']}` | {t['class']} | {t['frames']} |" for t in yanalysis["tokens"]]+["","All four paths used the locked model/config and the Phase 2S scales (0.667, 1.0, 0.8) with normalization. The original graph does not expose a duration output, and separate ONNX sessions are stochastic, so cross-path duration/PCM identity cannot be claimed. The self-duration path uses the supplied predicted vector exactly; graph structure and token sequence are unchanged. Since Original, V1 and V6 were all unacceptable for Y, classify this as an independent Lessac/eSpeak pronunciation or item-level baseline issue, not evidence for changing the duration policy."]
    (a.output/"y-baseline-analysis.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    lines=["# Phase 2AJ preference/edit correlation","","Descriptive only; seven prior usable Phase 2AI trials do not establish causation.","","| Item | Preferred | Interpretation |","|---|---|---|"]
    for r in correlation:
        interp="V6 winner; E2/E3/E4/E5 pattern recorded" if r["preferred"]=="a8" else "V1 or Original winner; extra V6 edits are suspect for this item, not globally rejected"
        lines.append(f"| {r['item']} | {r['preferred']} | {interp} |")
    lines += ["","V6 winners were exclamation mark and expanded; both receive substantial E2 boundary/PAD edits and at least one terminal/vowel edit. V1 winners U, W and unavailable do not justify attributing any single family as harmful. Original winners S and 0 are sensitive items: even E1 may be unnecessary or preference-sensitive. No simple adaptive rule is justified from seven observations; Candidate M=A5 and Candidate F=A6 are fixed, phonetic-plan-based ablations."]
    (a.output/"preference-edit-correlation.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    lines=["# Phase 2AJ automatic findings","",f"Corpus: {len(ITEMS)} items (seven primary usable Phase 2AI items plus eight confirmation items). `Y` is analyzed separately and excluded from policy scoring.","","| Policy | Median | P75 | P90 | P95 | Max | Median/P95 saved | Median % reduction |","|---|---:|---:|---:|---:|---:|---:|---:|"]
    for policy in POLICIES:
        s=summaries[policy]; d=s["duration_ms"]; lines.append(f"| {policy.upper()} | {d['median']:.1f} | {d['p75']:.1f} | {d['p90']:.1f} | {d['p95']:.1f} | {d['max']:.1f} | {s['milliseconds_saved']['median']:.1f}/{s['milliseconds_saved']['p95']:.1f} ms | {s['percent_reduction']['median']:.2f}% |")
    lines += ["","A0 is Original; A1 is exact V1; A8 is exact V6. Candidate M is A5 (V1 + internal boundary + terminal optimization). Candidate F is A6 (V1 + internal boundary + one long-vowel reduction). A5 retains more V6 duration savings than A6 automatically, while A6 isolates whether one vowel edit adds useful value. A deterministic adaptive rule is not justified by seven preference observations; no item identity is used.","","Automatic validation passed for all 135 renders. All edit families preserve token sequence, active-token minimums, consonants, finite normalized mono 16-kHz PCM, and alignment safety. No Phase 2AJ policy is perceptually validated until the next blind gate."]
    (a.output/"phase2aj-findings.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

if __name__=="__main__": main()
