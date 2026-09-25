"""Descriptive, post-hoc analysis of saved paired calendar interventions (CPU)."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def auc(scores, labels):
    positive = [s for s, y in zip(scores, labels) if y]
    negative = [s for s, y in zip(scores, labels) if not y]
    if not positive or not negative:
        return None
    return sum((a > b) + .5 * (a == b) for a in positive for b in negative) / (len(positive) * len(negative))


def summarize(rows):
    paired = [r for r in rows if r['both_extracted']]
    changed = sum(r['numeric_change'] for r in rows)
    return dict(n=len(rows), text_changes=sum(r['text_change'] for r in rows),
                token_changes=sum(r['token_change'] for r in rows), both_extracted=len(paired),
                numeric_changes=changed, numeric_rate_both=changed / len(paired) if paired else None,
                mean_gap=statistics.mean(r['confidence_gap'] for r in rows),
                median_gap=statistics.median(r['confidence_gap'] for r in rows),
                median_position_distance=statistics.median(r['position_distance'] for r in rows),
                median_hamming=statistics.median(r['hamming'] for r in rows))


def main(directory, audit_path, out):
    from calendar_swap import digest
    manifest = json.loads((directory / 'manifest.json').read_text())
    paths = [directory / f'question_{i:05d}.json' for i in manifest['question_ids']]
    protected = paths + [directory / 'manifest.json', directory / 'summary.json', audit_path]
    hashes = {str(p.resolve()): sha(p) for p in protected}
    raw = [json.loads(p.read_text()) for p in paths]
    audit = json.loads(audit_path.read_text())
    assert digest(raw) == audit['source_records_sha256']
    assert digest(manifest) == audit['source_manifest_sha256']
    decisions = {(d['question_id'], d['arm']): d for d in audit['decisions']}
    rows = []
    for r in raw:
        s = r['swap']
        a, b = r['baseline_ids'], r['treated_ids']
        assert len(a) == len(b) == manifest['protocol']['gen_len']
        assert sum(x != y for x, y in zip(a, b)) == r['hamming']
        assert all(r['checks'].values())
        assert s['deferred_confidence'] >= s['advanced_confidence']
        assert set(s['baseline_bundle']) - set(s['treated_bundle']) == {s['deferred_position']}
        assert set(s['treated_bundle']) - set(s['baseline_bundle']) == {s['advanced_position']}
        ba, ta = [decisions[r['question_id'], arm]['answer'] for arm in ('baseline', 'treated')]
        eos = set(r['eos_ids'])
        def visible(ids):
            return ids[:next((i for i, token in enumerate(ids) if token in eos), len(ids))]
        av, bv = visible(a), visible(b)
        altered = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
        rows.append(dict(question_id=r['question_id'],
            confidence_gap=s['deferred_confidence'] - s['advanced_confidence'],
            deferred_confidence=s['deferred_confidence'], advanced_confidence=s['advanced_confidence'],
            deferred_position=s['deferred_position'], advanced_position=s['advanced_position'],
            position_distance=abs(s['deferred_position'] - s['advanced_position']),
            prompt_tokens=len(r['prompt_ids']), hamming=r['hamming'],
            token_change=a != b, visible_token_change=av != bv,
            text_change=r['baseline']['text'] != r['treated']['text'],
            both_extracted=ba is not None and ta is not None,
            numeric_change=ba is not None and ta is not None and ba != ta,
            first_final_difference=altered[0] if altered else None,
            differences_outside_swapped_positions=sum(i not in (s['deferred_position'], s['advanced_position']) for i in altered)))
    groups = {}
    for feature in ('confidence_gap', 'position_distance'):
        ordered = sorted(rows, key=lambda r: (r[feature], r['question_id']))
        groups[feature] = [dict(bin=i + 1, minimum=chunk[0][feature], maximum=chunk[-1][feature], **summarize(chunk))
                           for i in range(4) if (chunk := ordered[i * len(rows)//4:(i+1)*len(rows)//4])]
    paired = [r for r in rows if r['both_extracted']]
    scores = {}
    for feature, direction in (('confidence_gap', -1), ('position_distance', 1)):
        scores[feature] = dict(direction='smaller' if direction == -1 else 'larger',
            text_auc=auc([direction*r[feature] for r in rows], [r['text_change'] for r in rows]),
            numeric_auc_both=auc([direction*r[feature] for r in paired], [r['numeric_change'] for r in paired]))
    result = dict(scope='Exploratory descriptive associations; no causal identification of mediators or confirmatory testing',
        parser_version=audit['parser_version'], source_sha256=hashes, script_sha256=sha(Path(__file__)),
        total=summarize(rows), by_quartile=groups, directional_auc=scores,
        by_numeric_outcome={label:summarize(group) for label, group in
            [('changed', [r for r in paired if r['numeric_change']]), ('same_extracted', [r for r in paired if not r['numeric_change']])]},
        visible_token_changes=sum(r['visible_token_change'] for r in rows),
        changed_outside_swap=sum(r['differences_outside_swapped_positions'] > 0 for r in rows),
        first_final_difference_histogram=dict(Counter(r['first_final_difference'] for r in rows if r['token_change'])), rows=rows)
    out.mkdir(parents=True, exist_ok=True)
    assert out.resolve() != directory.resolve()
    (out / 'TRACE_ANALYSIS.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    lines = ['# Analyse exploratoire des traces sauvegardées', '',
        '500 questions, une intervention appariée par question. Les features sont mesurées au moment de l’échange ; les divergences sont mesurées à la fin. Aucune trajectoire complète étape par étape ni activation interne n’est disponible dans ces JSON.', '',
        f"Texte différent : {result['total']['text_changes']}/500 ; tokens complets différents : {result['total']['token_changes']}/500 ; tokens avant EOS différents : {result['visible_token_changes']}/500.",
        f"Différences finales hors des deux positions échangées : {result['changed_outside_swap']}/500. Cela montre une propagation dans la sortie, sans dater sa survenue.", '',
        '## Quartiles descriptifs', '',
        'Groupes de 125 questions triées par feature, puis ID pour départager les ex æquo. Une même distance peut chevaucher deux groupes. Aucun seuil n’a été optimisé. Les changements numériques sont conditionnés à une double extraction v4, ce qui introduit une sélection.', '']
    for feature, bins in groups.items():
        lines += [f'### {feature}', '', '| Groupe | Min–max | Texte différent | Numérique différent / deux extraites |', '|---|---|---|---|']
        for b in bins:
            lines.append(f"| {b['bin']} | {b['minimum']:.6g}–{b['maximum']:.6g} | {b['text_changes']}/{b['n']} | {b['numeric_changes']}/{b['both_extracted']} |")
        lines.append('')
    lines += ['## Pouvoir discriminant descriptif', '',
              'AUC = probabilité qu’une paire modifiée reçoive un score supérieur à une paire non modifiée, avec moitié des égalités. 0,5 indique une absence de discrimination. Scores examinés : petit écart de confiance et grande distance spatiale. Ces AUC sont calculées sur le même échantillon exploré, sans validation indépendante.', '']
    for feature, s in scores.items():
        lines.append(f"- {feature} ({s['direction']}) : AUC texte {s['text_auc']:.3f} ; AUC numérique parmi les doubles extractions {s['numeric_auc_both']:.3f}.")
    lines += ['', '## Limites', '',
        'La comparaison appariée documente l’effet de cette intervention déterministe sur ces sorties. Les associations entre écart de confiance, position et divergence ne prouvent pas que ces features causent l’effet : elles ne sont pas randomisées. La distance de Hamming inclut les 256 positions, y compris après EOS ; elle n’est pas une mesure de différence sémantique. La première position différente dans la séquence finale n’est pas la première étape de divergence temporelle.', '',
        'Pas de recherche de seuil optimal, de p-values multiples ou de modèle prédictif ajusté sur les 35 événements. Les abstentions du parseur restent non résolues. Les résultats ne démontrent ni un meilleur décodeur ni un mécanisme interne.', '']
    (out / 'TRACE_ANALYSIS.md').write_text('\n'.join(lines))
    assert hashes == {str(p.resolve()): sha(p) for p in protected}
    print(json.dumps({k: result[k] for k in ('total', 'directional_auc', 'by_numeric_outcome', 'visible_token_changes', 'changed_outside_swap', 'by_quartile')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--audit', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    main(args.directory, args.audit, args.out)
