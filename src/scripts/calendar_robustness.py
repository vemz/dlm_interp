"""Question-level bootstrap robustness checks; exploratory, not confirmatory."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from calendar_swap import wilson


def auc(scores, labels):
    scores, labels = np.asarray(scores), np.asarray(labels, dtype=bool)
    n1, n0 = labels.sum(), (~labels).sum()
    if not n1 or not n0:
        return None
    # Midranks, including exact ties, with O(n log n) complexity.
    order = np.argsort(scores, kind='stable')
    values = scores[order]
    starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
    ends = np.r_[starts[1:], len(scores)]
    ranks = np.empty(len(scores), dtype=float)
    for lo, hi in zip(starts, ends):
        ranks[order[lo:hi]] = (lo + 1 + hi) / 2
    return float((ranks[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ci(values):
    return np.quantile(values, [.025, .975]).tolist()


def run(path, out, draws=5000):
    source_bytes = path.read_bytes()
    source = json.loads(source_bytes)
    rows = source['rows']
    # Verify the original raw records and parser audit pinned by TRACE_ANALYSIS.
    for filename, expected in source['source_sha256'].items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
            raise ValueError('Changed source: ' + filename)
    n = len(rows)
    y = np.array([r['text_change'] for r in rows], bool)
    gap = np.array([r['confidence_gap'] for r in rows])
    order = sorted(range(n), key=lambda i: (gap[i], rows[i]['question_id']))
    low, high = np.array(order[:n//4]), np.array(order[-n//4:])
    rng = np.random.default_rng(20260925)
    delta_draws = [float(y[rng.choice(high, len(high))].mean() - y[rng.choice(low, len(low))].mean()) for _ in range(draws)]
    # Additional features expose possible confounding of gap with absolute confidence.
    features = {
        'larger_gap': gap,
        'lower_advanced_confidence': -np.array([r['advanced_confidence'] for r in rows]),
        'lower_deferred_confidence': -np.array([r['deferred_confidence'] for r in rows]),
        'larger_position_distance': np.array([r['position_distance'] for r in rows]),
        'longer_prompt': np.array([r['prompt_tokens'] for r in rows]),
    }
    paired = np.array([r['both_extracted'] for r in rows])
    endpoints = {'text_all': (np.ones(n, bool), y),
                 'numeric_both_extracted': (paired, np.array([r['numeric_change'] for r in rows], bool))}
    estimates = {}
    for endpoint, (mask, labels) in endpoints.items():
        yy = labels[mask]
        samples = rng.integers(len(yy), size=(draws, len(yy)))
        estimates[endpoint] = {}
        for name, xx in features.items():
            xx = xx[mask]
            boot = [a for idx in samples if (a := auc(xx[idx], yy[idx])) is not None]
            estimates[endpoint][name] = dict(auc=auc(xx, yy), bootstrap_ci95=ci(boot), valid_draws=len(boot))
    text_identical = ~y
    # An identical text with a different numeric label signals an evaluation inconsistency.
    assert not any(r['numeric_change'] and not r['text_change'] for r in rows)
    result = dict(scope='post-hoc descriptive bootstrap; no adjusted causal effect or independent validation',
        bootstrap_draws=draws, seed=20260925,
        trace_analysis_sha256=hashlib.sha256(source_bytes).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        text_divergence=wilson(int(y.sum()), n),
        top_minus_bottom_quartile=dict(difference=float(y[high].mean()-y[low].mean()),
            bootstrap_ci95=ci(delta_draws), low_n=len(low), high_n=len(high),
            note='Membership fixed at observed quartiles; bootstrap within groups.'),
        auc=estimates,
        sensitivity=dict(within_divergent_texts=wilson(sum(r['numeric_change'] for r in rows if r['text_change'] and r['both_extracted']),
                        sum(r['text_change'] and r['both_extracted'] for r in rows)),
            unresolved_pairs=sum(not r['both_extracted'] for r in rows),
            unresolved_text_divergent=sum(not r['both_extracted'] and r['text_change'] for r in rows),
            identical_text_pairs=int(text_identical.sum())),
        limitations=['Fixed GSM8K sample: intervals rely on treating questions as exchangeable.',
                    'Quartiles and feature directions examined after observing data.',
                    'Multiple correlated associations, no family-wise confirmatory claims.',
                    'Numeric endpoint selected by successful extraction; missing not assumed random.'])
    out.mkdir(parents=True, exist_ok=True)
    (out/'ROBUSTNESS.json').write_text(json.dumps(result, indent=2)+'\n')
    lines = ['# Robustesse exploratoire', '',
        f"Bootstrap par question : {draws} réplications, graine 20260925. Intervalles descriptifs ; hypothèse d’échangeabilité des questions, pas une confirmation sur données nouvelles.", '',
        f"Texte différent : {int(y.sum())}/{n}, IC Wilson {result['text_divergence']['ci95']}.",
        f"Différence quartile supérieur − inférieur : {result['top_minus_bottom_quartile']['difference']:.3f}, IC bootstrap {ci(delta_draws)}. Groupes fixés aux quartiles observés.", '',
        '| Score orienté | AUC texte [IC 95 %] | AUC numérique, doubles extractions [IC 95 %] |', '|---|---|---|']
    for feature in features:
        cells = []
        for endpoint in endpoints:
            item = estimates[endpoint][feature]
            cells.append(f"{item['auc']:.3f} [{item['bootstrap_ci95'][0]:.3f}, {item['bootstrap_ci95'][1]:.3f}]")
        lines.append('| '+feature+' | '+' | '.join(cells)+' |')
    lines += ['', 'Ces scores sont corrélés : en particulier, écart de confiance = confiance différée − confiance avancée. Une association plus forte avec la faible confiance de la position avancée empêcherait d’attribuer le signal au seul écart. Aucun modèle ajusté ni effet causal des features n’est estimé.', '',
              'Les paires à texte identique n’apportent pas de changement numérique détecté. Pour les autres, les abstentions ne sont pas imputées comme des réponses stables. Les dénominateurs et intervalles détaillés figurent dans ROBUSTNESS.json.', '']
    (out/'ROBUSTNESS.md').write_text('\n'.join(lines))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('traces', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--draws', type=int, default=5000)
    args = parser.parse_args()
    if args.draws < 100:
        parser.error('Use at least 100 bootstrap draws')
    run(args.traces, args.out, args.draws)
