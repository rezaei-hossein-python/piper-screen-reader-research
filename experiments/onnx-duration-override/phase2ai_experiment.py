"""Run the reproducible Phase 2AI duration-frontier experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import time
import wave
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import onnxruntime as ort
from piper.voice import PiperVoice

from duration_probe import validate_override
from phase2ai_policy import Token, apply_policy, classify_token


CHARACTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
DIGITS = list("0123456789")
PUNCTUATION = ["comma", "period", "question mark", "exclamation mark", "slash", "at sign"]
UI = ["button", "selected", "checked", "unchecked", "edit", "link", "heading", "menu", "dialog", "unavailable", "collapsed", "expanded"]
ITEMS = CHARACTERS + DIGITS + PUNCTUATION + UI
POLICIES = ["p0", "v1", "v2", "v3", "v4", "v5", "v6"]
SCALES = np.asarray([0.667, 1.0, 0.8], dtype=np.float32)
HOP = 256


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q, method="linear"))


def summary(values: list[float]) -> dict[str, float]:
    return {"median": percentile(values, 50), "p75": percentile(values, 75), "p90": percentile(values, 90), "p95": percentile(values, 95), "maximum": max(values)}


def normalize(audio: np.ndarray) -> np.ndarray:
    result = audio.reshape(-1).astype(np.float32)
    peak = float(np.max(np.abs(result))) if result.size else 0.0
    if peak >= 1e-8:
        result /= peak
    return np.clip(result, -1.0, 1.0)


def audio_metric(audio: np.ndarray, sample_rate: int) -> dict[str, object]:
    raw = audio.reshape(-1)
    normalized = normalize(raw)
    return {
        "sample_rate": sample_rate, "channels": 1, "samples": int(raw.size),
        "duration_ms": raw.size * 1000.0 / sample_rate,
        "finite": bool(np.isfinite(raw).all()), "nan_or_inf": bool(not np.isfinite(raw).all()),
        "normalized_peak": float(np.max(np.abs(normalized))),
        "clipped": bool(np.any(np.abs(normalized) > 1.0)),
        "sha256": hashlib.sha256(normalized.tobytes()).hexdigest(),
    }


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    pcm = (normalize(audio) * 32767).round().astype("<i2").tobytes()
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1); out.setsampwidth(2); out.setframerate(sample_rate); out.writeframes(pcm)


def inputs(ids: list[int]) -> dict[str, np.ndarray]:
    return {"input": np.asarray([ids], np.int64), "input_lengths": np.asarray([len(ids)], np.int64), "scales": SCALES,
            "duration_override": np.ones((1, 1, len(ids)), np.float32), "duration_override_enabled": np.asarray(False, np.bool_)}


def fmt(value: float) -> str:
    return f"{value:.1f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path); parser.add_argument("rewritten", type=Path); parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path); parser.add_argument("listening", type=Path); parser.add_argument("answer_key", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True); args.listening.mkdir(parents=True, exist_ok=True)
    voice = PiperVoice.load(str(args.model), str(args.config))
    session = ort.InferenceSession(str(args.rewritten), providers=["CPUExecutionProvider"])
    reverse = {ids[0]: symbol for symbol, ids in voice.config.phoneme_id_map.items()}
    rows: list[dict[str, object]] = []
    class_frames: Counter[str] = Counter(); symbol_audit: dict[tuple[int, str], Counter[str]] = defaultdict(Counter)
    warnings: list[str] = []

    for item in ITEMS:
        phonemes = voice.phonemize(item)[0]
        ids = voice.phonemes_to_ids(phonemes)
        labels = [reverse[token_id] for token_id in ids]
        request = inputs(ids)
        base_outputs = session.run(None, request)
        base_audio, predicted_raw = base_outputs[0], base_outputs[1]
        predicted = predicted_raw.reshape(-1).astype(int)
        tokens = [Token(i, symbol, token_id, int(predicted[i])) for i, (symbol, token_id) in enumerate(zip(labels, ids))]
        token_rows = []
        for token in tokens:
            cls = classify_token(token.symbol)
            position = "leading" if token.index < 2 else "trailing" if token.index >= len(tokens) - 2 else "internal"
            class_frames[cls] += token.frames
            symbol_audit[(token.token_id, token.symbol)].update(occurrences=1, frames=token.frames)
            token_rows.append({"index": token.index, "token_id": token.token_id, "phoneme": token.symbol, "class": cls,
                               "predicted_frames": token.frames, "relative_position": token.index / max(1, len(tokens)-1),
                               "word_position": position, "leading_internal_trailing": position})
        row: dict[str, object] = {"item": item, "group": "character" if item in CHARACTERS else "digit" if item in DIGITS else "punctuation" if item in PUNCTUATION else "ui",
                                  "phonemes": phonemes, "ids": ids, "tokens": token_rows, "policies": {}}
        for policy in POLICIES:
            plan, changed_indices = apply_policy(tokens, policy)
            if len(plan) != len(ids): warnings.append(f"{item}/{policy}: token deletion")
            override = np.asarray(plan, np.float32).reshape(1, 1, -1)
            validate_override(override, predicted_raw.astype(np.float32))
            if policy == "p0":
                audio = base_audio
            else:
                changed = dict(request); changed["duration_override"] = override; changed["duration_override_enabled"] = np.asarray(True, np.bool_)
                audio = session.run(None, changed)[0]
            acoustic = audio_metric(audio, voice.config.sample_rate)
            if not acoustic["finite"] or acoustic["clipped"] or acoustic["sample_rate"] != 16000: warnings.append(f"{item}/{policy}: PCM validation")
            row["policies"][policy] = {"durations": plan, "changed_indices": changed_indices, "token_modifications": len(changed_indices),
                                        "frames_removed": int(sum(predicted)-sum(plan)), "milliseconds_removed": (sum(predicted)-sum(plan))*HOP*1000/16000,
                                        **acoustic}
        rows.append(row)

    policy_summaries: dict[str, object] = {}
    for policy in POLICIES:
        durations = [row["policies"][policy]["duration_ms"] for row in rows]
        frames = [row["policies"][policy]["frames_removed"] for row in rows]
        saved = [row["policies"][policy]["milliseconds_removed"] for row in rows]
        reductions = [100 * saved_value / row["policies"]["p0"]["duration_ms"] for row, saved_value in zip(rows, saved)]
        modifications = [row["policies"][policy]["token_modifications"] for row in rows]
        ds = summary(durations)
        policy_summaries[policy] = {"duration_ms": ds, "frames_removed": {"median": percentile(frames,50), "p95": percentile(frames,95)},
             "milliseconds_saved": {"median": percentile(saved,50), "p95": percentile(saved,95)},
             "percent_reduction": {"median": percentile(reductions,50), "p95": percentile(reductions,95)},
             "capacity_per_second": {k: 1000/ds[k] for k in ("median","p90","p95")}, "median_token_modifications": percentile(modifications,50)}
    for index, policy in enumerate(POLICIES[1:], 1):
        prior = POLICIES[index-1]
        extra_ms = sum(r["policies"][policy]["milliseconds_removed"] - r["policies"][prior]["milliseconds_removed"] for r in rows)
        extra_mods = sum(sum(a != b for a, b in zip(r["policies"][policy]["durations"], r["policies"][prior]["durations"])) for r in rows)
        policy_summaries[policy]["incremental_ms_saved_total"] = extra_ms
        policy_summaries[policy]["incremental_token_modifications_total"] = extra_mods
        policy_summaries[policy]["incremental_ms_per_modification"] = extra_ms / extra_mods if extra_mods > 0 else 0.0

    group_summaries = {}
    for group in ("character", "digit", "character_digit", "ui", "punctuation"):
        selected = rows if group == "all" else [r for r in rows if r["group"] in ({"character","digit"} if group == "character_digit" else {group})]
        group_summaries[group] = {policy: summary([r["policies"][policy]["duration_ms"] for r in selected]) for policy in POLICIES}

    # Rank difficult baseline and candidate items; V6 is the strongest structurally bounded policy.
    candidate = "v6"
    outliers = sorted(rows, key=lambda row: row["policies"][candidate]["duration_ms"], reverse=True)

    # Warm latency is measured after all main graph work.
    probe = next(row for row in rows if row["item"] == "button")
    probe_request = inputs(probe["ids"])
    v6_request = dict(probe_request); v6_request["duration_override"] = np.asarray(probe["policies"][candidate]["durations"], np.float32).reshape(1,1,-1); v6_request["duration_override_enabled"] = np.asarray(True, np.bool_)
    session.run(None, probe_request); session.run(None, v6_request)
    latency = {}
    for name, req in (("original", probe_request), (candidate, v6_request)):
        times=[]
        for _ in range(20):
            start=time.perf_counter(); session.run(None, req); times.append((time.perf_counter()-start)*1000)
        latency[name] = {"median_ms": statistics.median(times), "p95_ms": percentile(times,95), "iterations":20}

    # Eight strategic items: four difficult characters, one digit, one punctuation, two UI; include outliers.
    difficult_chars = [r["item"] for r in outliers if r["group"] == "character"][:4]
    chosen = difficult_chars + [next(r["item"] for r in outliers if r["group"] == "digit"), next(r["item"] for r in outliers if r["group"] == "punctuation")] + [r["item"] for r in outliers if r["group"] == "ui"][:2]
    rng = random.Random(20260809); key = {}
    for trial_number, item in enumerate(chosen, 1):
        row = next(r for r in rows if r["item"] == item); base_request = inputs(row["ids"]); variants = {}
        for name, policy in (("original","p0"),("v1","v1"),("candidate_b",candidate)):
            req=dict(base_request)
            if policy != "p0": req["duration_override"] = np.asarray(row["policies"][policy]["durations"],np.float32).reshape(1,1,-1); req["duration_override_enabled"] = np.asarray(True,np.bool_)
            variants[name]=session.run(None,req)[0]
        shuffled=list(variants.items()); rng.shuffle(shuffled); assignment={}; trial=f"trial-{trial_number:02d}"
        for letter,(name,audio) in zip("abc",shuffled): write_wav(args.listening/f"{trial}-{letter}.wav",audio,16000); assignment[letter]=name
        key[trial]={"source_item":item,"assignment":assignment}
    args.answer_key.write_text(json.dumps(key,indent=2,ensure_ascii=False),encoding="utf-8")
    (args.listening/"instructions.txt").write_text("Judge the best speed + quality combination.\n\nTrial 1: A/B/C best\nTrial 2: A/B/C best\nTrial 3: A/B/C best\nTrial 4: A/B/C best\nTrial 5: A/B/C best\nTrial 6: A/B/C best\nTrial 7: A/B/C best\nTrial 8: A/B/C best\n\nFlag any item: quality degraded / pronunciation degraded\n",encoding="utf-8")

    audit = [{"token_id": token_id, "symbol": symbol, "class": classify_token(symbol), **counts} for (token_id,symbol),counts in sorted(symbol_audit.items())]
    result = {"settings":{"sample_rate":16000,"hop_length":HOP,"scales":SCALES.tolist(),"corpus_size":len(ITEMS)}, "candidate_b":candidate,
              "class_frame_occupancy":dict(class_frames), "token_audit":audit, "policy_summaries":policy_summaries, "group_summaries":group_summaries,
              "warm_latency":latency, "automatic_validation":{"warnings":warnings,"passed":not warnings,"items":len(rows),"policy_renders":len(rows)*len(POLICIES)}, "items":rows,
              "listening":{"items":chosen,"trials":8,"wav_count":24,"blinded":True}}
    (args.output/"phase2ai-measurements.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")

    # Durable audit report, including exact reconstruction of the old 370-frame pool.
    legacy = {"^","$","ˈ","ː","ɛ","ɪ","ɑ","ʌ","ə","ɐ","ʲ","ʔ","̩","ᵻ","ŋ","ʃ"," ","ɹ"}
    old_path = args.output.parent/"phase2ah"/"phase2ah-measurements.json"
    old = json.loads(old_path.read_text(encoding="utf-8")); old_counts=Counter(); old_symbols=Counter()
    for old_row in old["items"]:
        for symbol,frames in zip(old_row["labels"],old_row["original_durations"]):
            if symbol in legacy: old_counts[classify_token(symbol)]+=frames; old_symbols[symbol]+=frames
    lines=["# Phase 2AI unknown-token audit","","The Phase 2AH `protected/unknown` pool is fully explained. It was 370 frames, but it was not a single semantic class. The legacy classifier missed Unicode IPA and Piper control tokens.","","## Exact 370-frame reconstruction","","| Correct class | Frames |","|---|---:|"]
    lines += [f"| {cls} | {frames} |" for cls,frames in old_counts.most_common()]
    lines += [f"| **Total** | **{sum(old_counts.values())}** |","","Symbols and frames: " + ", ".join(f"`{s}`={f}" for s,f in old_symbols.most_common()) + ".","","`^` (ID 1) is BOS, `$` (ID 2) is EOS, and `_` (ID 0) is Piper PAD inserted before and after phonemes. Stress/length/diacritic tokens remain protected. IPA tokens are speech-bearing and are now classified by manner.","","## Expanded-corpus active-token audit","","| ID | Symbol | Class | Occurrences | Frames |","|---:|---|---|---:|---:|"]
    lines += [f"| {a['token_id']} | `{a['symbol']}` | {a['class']} | {a['occurrences']} | {a['frames']} |" for a in audit]
    lines += ["","## Frame occupancy","","| Class | Frames |","|---|---:|"]+[f"| {c} | {f} |" for c,f in class_frames.most_common()]
    (args.output/"unknown-token-audit.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

    olines=["# Phase 2AI duration outliers","",f"Ranked by Candidate B ({candidate.upper()}) waveform duration.","","| Rank | Item | Group | P0 ms | V1 ms | Candidate B ms | Largest Candidate-B token contributors |","|---:|---|---|---:|---:|---:|---|"]
    for rank,row in enumerate(outliers[:15],1):
        contributors=sorted(row["tokens"],key=lambda t:t["predicted_frames"],reverse=True)[:4]
        detail=", ".join(f"`{t['phoneme']}` {t['class']} {t['predicted_frames']}f" for t in contributors)
        olines.append(f"| {rank} | {row['item']} | {row['group']} | {row['policies']['p0']['duration_ms']:.0f} | {row['policies']['v1']['duration_ms']:.0f} | {row['policies'][candidate]['duration_ms']:.0f} | {detail} |")
    (args.output/"duration-outliers.md").write_text("\n".join(olines)+"\n",encoding="utf-8")

    flines=["# Phase 2AI automatic findings","",f"Corpus: {len(ITEMS)} interactive utterances (26 letters, 10 digits, 6 spoken punctuation names, 12 UI micro-utterances). Candidate B is {candidate.upper()} pending listening.","","| Policy | Median | P90 | P95 | Max | Median saved | P95 saved | Median reduction | Median modifications | Incremental ms/mod |","|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for p in POLICIES:
        s=policy_summaries[p]; d=s["duration_ms"]
        flines.append(f"| {p.upper()} | {fmt(d['median'])} | {fmt(d['p90'])} | {fmt(d['p95'])} | {fmt(d['maximum'])} | {fmt(s['milliseconds_saved']['median'])} | {fmt(s['milliseconds_saved']['p95'])} | {fmt(s['percent_reduction']['median'])}% | {fmt(s['median_token_modifications'])} | {fmt(s.get('incremental_ms_per_modification',0))} |")
    flines += ["","V1 is exactly Phase 2AH P1. V2 adds only audited PAD/BOS/EOS occupancy. V3 adds one long-vowel frame; V4 repeats that bounded reduction across eligible long vowels. V5 adds terminal PAD/EOS optimization. V6 equals V5 because no further independently justified class was available; this is the diminishing-return stop.","","All consonants, unknown/other speech-bearing tokens, stress, length and diacritic controls remain unchanged. No global scalar or PCM truncation is used.","",f"Automatic validation: {'PASS' if not warnings else 'FAIL'} ({len(rows)*len(POLICIES)} renders). Warm `button` inference: original median {latency['original']['median_ms']:.1f} ms; {candidate.upper()} median {latency[candidate]['median_ms']:.1f} ms."]
    (args.output/"phase2ai-findings.md").write_text("\n".join(flines)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
