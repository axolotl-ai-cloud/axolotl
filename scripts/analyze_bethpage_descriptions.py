import json, re, statistics, argparse, collections, sys, pathlib

def load_records(path):
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                records.append(json.loads(ln))
            except Exception as e:
                print(f"WARN could not parse line: {ln[:80]}... {e}", file=sys.stderr)
    return records

_generic_open_re = re.compile(r'^This par \d+,? ?\d+? yards hole rewards precise placement over brute force', re.I)
_epitomizes_re = re.compile(r'epitomizes Bethpage Black\'s strategic demands', re.I)
_complexity_re = re.compile(r"yards of strategic complexity reward", re.I)

_token_re = re.compile(r"[A-Za-z0-9']+")

def norm(s: str) -> str:
    return re.sub(r'\s+', ' ', s.strip())

def token_set(s: str):
    return set(_token_re.findall(s.lower()))

def jaccard(a, b):
    ta, tb = token_set(a), token_set(b)
    if not (ta or tb):
        return 0.0
    return len(ta & tb) / len(ta | tb)

def analyze(records):
    report = {}
    # Similarity desc3 vs desc4
    sims = []
    high_sim_holes = []
    for r in records:
        d3 = norm(r.get('new_description_3',''))
        d4 = norm(r.get('new_description_4',''))
        sim = jaccard(d3, d4)
        sims.append(sim)
        if sim >= 0.75:
            high_sim_holes.append(r['hole'])
    report['desc3_desc4_avg_jaccard'] = round(statistics.mean(sims), 3) if sims else 0
    report['desc3_desc4_high_similarity_holes'] = high_sim_holes
    report['desc3_desc4_high_similarity_pct'] = round(len(high_sim_holes)/len(records), 3) if records else 0

    # Generic openings counts
    stock3 = stock4 = 0
    epit5 = complex5 = 0
    d5_texts = []
    for r in records:
        d3 = norm(r.get('new_description_3',''))
        d4 = norm(r.get('new_description_4',''))
        d5 = norm(r.get('new_description_5',''))
        d5_texts.append(d5)
        if _generic_open_re.search(d3): stock3 += 1
        if _generic_open_re.search(d4): stock4 += 1
        if _epitomizes_re.search(d5): epit5 += 1
        if _complexity_re.search(d5): complex5 += 1
    report['desc3_generic_openings'] = stock3
    report['desc4_generic_openings'] = stock4
    report['desc5_epitomizes_phrase'] = epit5
    report['desc5_strategic_complexity_phrase'] = complex5

    # Description 5 duplication
    d5_counter = collections.Counter(d5_texts)
    reused = {txt: [r['hole'] for r in records if norm(r.get('new_description_5','')) == txt]
              for txt, ct in d5_counter.items() if ct > 1}
    report['desc5_unique_variants'] = len(d5_counter)
    report['desc5_reused_variant_count'] = len(reused)
    # Display only first 5 reused samples
    report['desc5_reused_examples'] = [
        {'holes': holes, 'preview': txt[:110] + ('...' if len(txt) > 110 else ''), 'length_words': len(txt.split())}
        for txt, holes in list(reused.items())[:5]
    ]

    # Truncation check (no final period)
    report['desc3_truncated'] = sum(1 for r in records if not norm(r.get('new_description_3','')).endswith('.'))
    report['desc4_truncated'] = sum(1 for r in records if not norm(r.get('new_description_4','')).endswith('.'))
    report['desc5_truncated'] = sum(1 for r in records if not norm(r.get('new_description_5','')).endswith('.'))

    # Length stats desc3/4/5
    def lengths(key):
        vals = [len(norm(r.get(key,'' )).split()) for r in records if r.get(key)]
        if not vals: return {'min':0,'avg':0,'max':0}
        return {'min':min(vals), 'avg':round(statistics.mean(vals),1), 'max':max(vals)}
    report['length_stats_desc3'] = lengths('new_description_3')
    report['length_stats_desc4'] = lengths('new_description_4')
    report['length_stats_desc5'] = lengths('new_description_5')

    # Suggest remediation thresholds
    report['recommendations'] = {
        'desc3_desc4_max_jaccard_threshold': 0.6,
        'desc5_min_unique_variants': len(records),
        'target_generic_openings_max': 3,
    }
    return report


def main():
    ap = argparse.ArgumentParser(description='Analyze Bethpage description duplication & similarity.')
    ap.add_argument('--input', required=True, help='Path to data_template_5desc_filled.json')
    ap.add_argument('--json', action='store_true', help='Emit full JSON report')
    args = ap.parse_args()

    recs = load_records(args.input)
    rep = analyze(recs)

    if args.json:
        json.dump(rep, sys.stdout, indent=2)
    else:
        print('=== Bethpage Description Analysis Report ===')
        print(f"Records: {len(recs)}")
        print(f"Avg Jaccard desc3 vs desc4: {rep['desc3_desc4_avg_jaccard']} (high-sim holes: {rep['desc3_desc4_high_similarity_holes']})")
        print(f"High-sim %: {rep['desc3_desc4_high_similarity_pct']*100:.1f}% (>=0.75)")
        print(f"Generic openings desc3: {rep['desc3_generic_openings']} | desc4: {rep['desc4_generic_openings']}")
        print(f"Desc5 unique variants: {rep['desc5_unique_variants']} | reused variant count: {rep['desc5_reused_variant_count']}")
        if rep['desc5_reused_examples']:
            print('  Reused desc5 samples:')
            for ex in rep['desc5_reused_examples']:
                print(f"    Holes {ex['holes']} | words {ex['length_words']} | {ex['preview']}")
        print(f"Truncated desc3/4/5: {rep['desc3_truncated']}/{rep['desc4_truncated']}/{rep['desc5_truncated']}")
        print('Length stats:')
        for k in ('length_stats_desc3','length_stats_desc4','length_stats_desc5'):
            print(f"  {k}: {rep[k]}")
        print('Recommendations:')
        for k,v in rep['recommendations'].items():
            print(f"  {k}: {v}")

if __name__ == '__main__':
    main()
