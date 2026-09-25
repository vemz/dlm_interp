"""Offline oracle ceiling, termination checks and cross-fitted univariate probes.

Exploratory analysis of existing paired outputs; no generation or deployable router.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import expit
from scipy.stats import rankdata

from answer_audit_v4 import extract

FEATURES = ['confidence_rank4', 'confidence_rank5', 'confidence_gap', 'position_distance']
SEED = 20260926


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ceiling(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n = len(a)
    best = max(int(a.sum()), int(b.sum()))
    union = int((a | b).sum())
    return dict(n=n, baseline_correct=int(a.sum()), treated_correct=int(b.sum()),
                baseline_only=int((a & ~b).sum()), treated_only=int((b & ~a).sum()),
                best_fixed_correct=best, oracle_correct=union,
                best_fixed_rate=best/n if n else None, oracle_rate=union/n if n else None,
                margin_count=union-best, margin_pp=100*(union-best)/n if n else None)


def fit_univariate(x, y):
    """Four separate logistic regressions, ridge=1 on slope; no tuned parameters.

    Feature standardization uses training rows only. Intercepts are unpenalized.
    This batched Newton solver never mixes features in the same regression.
    """
    if not len(y) or len(np.unique(y)) < 2:
        raise ValueError('Each training fold must contain both directions')
    mean, scale = x.mean(0), x.std(0)
    scale = np.where(scale > 1e-12, scale, 1.)
    z = (x-mean)/scale
    beta = np.zeros((x.shape[1], 2))
    beta[:, 0] = np.log(y.mean()/(1-y.mean()))
    for _ in range(40):
        p = expit(beta[:, 0] + z*beta[:, 1])
        w = np.maximum(p*(1-p), 1e-10)
        g = np.column_stack(((p-y[:, None]).sum(0),
                             ((p-y[:, None])*z).sum(0)+beta[:, 1]))
        h = np.empty((x.shape[1], 2, 2))
        h[:, 0, 0] = w.sum(0)
        h[:, 0, 1] = h[:, 1, 0] = (w*z).sum(0)
        h[:, 1, 1] = (w*z*z).sum(0)+1.
        change = np.linalg.solve(h, g[..., None])[..., 0]
        beta -= change
        if np.max(np.abs(change)) < 1e-9:
            break
    return mean, scale, beta


def predict(fit, x):
    mean, scale, beta = fit
    return expit(beta[:, 0] + ((x-mean)/scale)*beta[:, 1])


def crossfit(x, labels, folds):
    """Labels are -1 for non-discordant pairs, 0/1 for direction.

    All held-out questions receive a prediction; future discordance is not an input.
    """
    scores = np.zeros_like(x)
    majority = np.zeros(len(x), bool)
    slopes = []
    for fold in sorted(set(folds)):
        test = folds == fold
        train = (~test) & (labels >= 0)
        fitted = fit_univariate(x[train], labels[train])
        scores[test] = predict(fitted, x[test])
        majority[test] = labels[train].mean() >= .5
        slopes.append(fitted[2][:, 1].tolist())
    return scores, majority, slopes


def balanced_accuracy(y, choices):
    return .5*(choices[y == 1].mean(0) + (1-choices[y == 0]).mean(0))


def auc(y, p):
    ranks = rankdata(p, method='average')
    n1, n0 = int(y.sum()), int((1-y).sum())
    return float((ranks[y == 1].sum()-n1*(n1+1)/2)/(n1*n0))


def holm(p):
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    bound = 0.
    for i, j in enumerate(order):
        bound = max(bound, min(1., (len(p)-i)*p[j]))
        adjusted[j] = bound
    return adjusted


def probe(x, a, b, folds, cohorts, permutations, draws):
    event = a != b
    labels = np.where(event, b.astype(int), -1)
    scores, majority, slopes = crossfit(x, labels, folds)
    choices = scores >= .5
    observed = balanced_accuracy(labels[event], choices[event])
    rng = np.random.default_rng(SEED)
    null = np.zeros((permutations, x.shape[1]))
    for j in range(permutations):
        permuted = labels.copy()
        # Preserve discordant membership and the gain/loss totals of each cohort.
        for cohort in sorted(set(cohorts)):
            idx = np.flatnonzero(event & (cohorts == cohort))
            permuted[idx] = rng.permutation(labels[idx])
        pp, _, _ = crossfit(x, permuted, folds)
        null[j] = balanced_accuracy(permuted[event], (pp >= .5)[event])
    pvalues = (1+(null >= observed-1e-12).sum(0))/(permutations+1)
    corrected = holm(pvalues)
    selected = np.where(choices, b[:, None], a[:, None])
    majority_score = np.where(majority, b, a)
    best_fixed = b if b.sum() > a.sum() else a
    delta = selected.astype(float)-best_fixed[:, None]
    # Descriptive uncertainty conditional on these out-of-fold predictions.
    boot = np.empty((draws, x.shape[1]))
    for j in range(draws):
        idx = np.concatenate([rng.choice(np.flatnonzero(cohorts == c),
                              size=int((cohorts == c).sum()), replace=True)
                              for c in sorted(set(cohorts))])
        boot[j] = delta[idx].mean(0)*100
    features = {}
    for j, name in enumerate(FEATURES):
        features[name] = dict(balanced_accuracy=float(observed[j]),
            directional_accuracy=float((choices[event, j] == labels[event]).mean()),
            directional_auc=auc(labels[event], scores[event, j]),
            permutation_p=float(pvalues[j]), holm_p=float(corrected[j]),
            selected_correct=int(selected[:, j].sum()), delta_best_fixed_pp=float(delta[:, j].mean()*100),
            conditional_bootstrap_ci95_pp=np.quantile(boot[:, j], [.025, .975]).tolist(),
            delta_cv_majority_pp=float((selected[:, j].astype(float)-majority_score).mean()*100),
            fold_slopes=[s[j] for s in slopes])
    # Cohort transfer is a supplementary split, not a fresh confirmation.
    transfer = {}
    for test_cohort in sorted(set(cohorts)):
        train = event & (cohorts != test_cohort)
        test = cohorts == test_cohort
        ps = predict(fit_univariate(x[train], labels[train]), x[test])
        event_test = event[test]
        transfer[str(test_cohort)] = dict(n=int(test.sum()), discordant=int(event_test.sum()),
            features={name:dict(balanced_accuracy=float(balanced_accuracy(
                labels[test][event_test], (ps >= .5)[event_test])[j]),
                selected_correct=int(np.where(ps[:, j] >= .5, b[test], a[test]).sum()))
                for j, name in enumerate(FEATURES)})
    return dict(n=len(a), discordant=int(event.sum()), gains=int((b & ~a).sum()),
        losses=int((a & ~b).sum()), cv_majority_correct=int(majority_score.sum()),
        features=features, cohort_transfer=transfer,
        oof=[dict(scores=s.tolist(), choose_treated=c.tolist(), majority=bool(m))
             for s,c,m in zip(scores,choices,majority)])


def report(r):
    lines = ['# Plafond oracle et prédiction du sens : audit exploratoire', '',
        'Audit CPU des 1 000 paires LLaDA déjà collectées. Aucune nouvelle génération. '
        'Les chiffres portent sur la justesse mesurée par le parseur v4 : une abstention '
        'compte comme incorrecte, sans établir que la réponse est sémantiquement fausse.', '',
        '## Décision scientifique', '',
        'Les quatre sondes univariées testées ne valident aucun contrôle utile du sens '
        'du changement. La construction d’un décodeur appris n’est donc pas la prochaine '
        'étape soutenue par ces données. Priorité à la fiabilité de la mesure puis à '
        'la portée des perturbations de calendrier pour les évaluations. Ce résultat '
        'ne ferme pas toutes les architectures, tâches ou politiques possibles.', '',
        '## Plafond best-of-2', '',
        'L’oracle choisit une branche juste dès qu’au moins l’une est juste. Sa marge '
        'est mesurée face à la meilleure branche fixe sur le même ensemble ; cette '
        'référence est descriptive et choisie après observation.', '',
        '| Ensemble | Témoin seul juste | Traité seul juste | Meilleure branche fixe | Oracle | Marge |',
        '|---|---:|---:|---:|---:|---:|']
    for key, label in [('calendar_swap_v2_main_data','Découverte'),
                       ('calendar_replication_v1','Réplication'),('pooled','Ensemble des 1 000')]:
        c=r['ceilings'][key]['v4']
        lines.append(f"| {label} | {c['baseline_only']} | {c['treated_only']} | "
            f"{100*c['best_fixed_rate']:.1f} % | {100*c['oracle_rate']:.1f} % | +{c['margin_pp']:.1f} points |")
    lines += ['', 'Le +4,4 points regroupé n’est pas la moyenne de +3,8 et +3,4 : '
        'le regroupement impose une seule branche fixe aux deux lots, alors que leurs '
        'meilleures branches diffèrent. La marge n’est pas un gain réalisable sans '
        'connaître le sens de l’effet sur chaque question.', '',
        '## Audit des marqueurs de fin et de la limite de longueur', '',
        '| Ensemble (1 000 sorties chacun) | EOS absent | Premier EOS au token 256 | Texte modifié en arrêtant aussi à EOT | Extraction modifiée |',
        '|---|---:|---:|---:|---:|']
    for key,label in [('calendar_swap_v2_main_data','Découverte'),('calendar_replication_v1','Réplication')]:
        c=r['termination_audit'][key]['counts']
        lines.append(f"| {label} | {c['missing_eos']} | {c['eos_at_last_position']} | "
                     f"{c['text_changed_at_eot']} | {c['answer_changed_at_eot']} |")
    lines += ['', 'Le tokenizer local reproduit les 2 000 textes sauvegardés. '
        'Le collecteur coupait à `<|endoftext|>` (126081), mais le modèle peut aussi '
        'émettre `<|eot_id|>` (126348). L’analyse de sensibilité coupe au premier des deux.', '',
        'La paire 623 de réplication perd son extraction de 8 dans les deux bras : '
        'le nombre situé après le marqueur de fin de tour était encore lu. '
        'Le témoin passe à 261/500, le traité à 271/500 et l’oracle à 288/500. '
        'La marge reste +3,4 points. En découverte, les scores sont identiques. '
        'Les étiquettes gain/perte/concordance des 1 000 paires sont inchangées : '
        'les sondes du sens sont donc identiques sous cette correction de découpage.', '',
        'Le premier marqueur de fin se situe dans les deux dernières positions pour '
        'les mêmes 769 et 709 sorties. Cela signale une forte occupation de la fenêtre, '
        'pas une preuve que chacune est tronquée. Inversement, EOS présent ne garantit '
        'pas une conclusion complète. Cet audit technique ne remplace pas une revue '
        'sémantique exhaustive des terminaisons. Le cas 466 reste un faux positif '
        'documenté du parseur figé ; il n’est pas corrigé implicitement ici.', '',
        '## Validation croisée du sens', '',
        'Cinq plis par question, distribués par hash indépendamment des réponses et '
        'équilibrés entre les deux cohortes. Une régression logistique distincte '
        'par caractéristique, pénalité de pente fixée à 1, standardisation sur '
        'l’entraînement seulement. Aucun choix d’hyperparamètre après lecture du test.', '',
        'L’apprentissage utilise les paires discordantes du pli d’entraînement ; '
        'chaque question du pli tenu à part reçoit une décision, sans connaître '
        'sa discordance future. Le seuil est 0,5. L’exactitude équilibrée moyenne '
        'la reconnaissance des gains et celle des pertes. Le test permute leur '
        'sens dans chaque cohorte, conserve les paires discordantes et réentraîne '
        'tous les plis. Correction de Holm sur les quatre caractéristiques.', '',
        '| Caractéristique seule | Exactitude équilibrée du sens | p permutation | p corrigée | Gain vs meilleure branche fixe |',
        '|---|---:|---:|---:|---:|']
    labels=['Confiance rang 4','Confiance rang 5','Écart de confiance','Distance des positions']
    for name,label in zip(FEATURES,labels):
        a=r['analyses']['v4_all']['features'][name]
        lines.append(f"| {label} | {100*a['balanced_accuracy']:.1f} % | {a['permutation_p']:.4f} | "
                     f"{a['holm_p']:.4f} | {a['delta_best_fixed_pp']:+.1f} point |")
    lines += ['', 'Ces résultats portent sur 90 discordances : 46 gains et 44 pertes. '
        'Une majorité apprise sans caractéristique sert aussi de contrôle dans le JSON. '
        'Les scores hors pli peuvent passer sous 50 % par instabilité des ajustements ; '
        'on ne retourne pas les prédictions après observation pour annoncer un gain. '
        'Les permutations incluent cette instabilité. Les intervalles bootstrap '
        'en JSON sont descriptifs, conditionnels aux prédictions hors pli fixées ; '
        'ils n’intègrent pas l’incertitude de réentraînement.', '',
        '## Sensibilité : deux réponses extraites', '',
        '694 paires sont extraites des deux côtés, dont 42 discordantes (21 gains, '
        '21 pertes). Les 48 autres discordances v4 impliquent une transition '
        'd’extraction. Le sous-ensemble doublement extrait est sélectionné par les '
        'sorties ; son score ne remplace pas une mesure globale.', '',
        '| Caractéristique seule | Exactitude équilibrée du sens | p corrigée |',
        '|---|---:|---:|']
    for name,label in zip(FEATURES,labels):
        a=r['analyses']['both_extracted']['features'][name]
        lines.append(f"| {label} | {100*a['balanced_accuracy']:.1f} % | {a['holm_p']:.4f} |")
    lines += ['', 'Le transfert découverte → réplication, et l’inverse, est aussi '
        'rapporté en JSON : aucune des quatre sondes ne dépasse 50 % d’exactitude '
        'équilibrée dans ces comparaisons. Ces contrôles restent exploratoires '
        'puisque les deux ensembles ont déjà été inspectés.', '',
        '## Mise en perspective avec NanoMDLM', '',
        'Le registre `results/RESULTS.md`, section 7, rapporte un plafond '
        'contrefactuel de NELBO sans contrôleur léger validé. Les caractéristiques '
        'locales, le lookahead et les horizons partiels ont échoué ; le pilote '
        'compressed future complete_1 (+0,0240) ne s’est pas confirmé sur '
        '40 nouvelles seeds (−0,00485, IC [−0,02223 ; +0,00947]). '
        'La proposition de routeur LLaDA devait donc commencer par ce diagnostic. '
        'Les métriques diffèrent : NELBO sur NanoMDLM, justesse extraite sur LLaDA.', '',
        '## Incertitude des comparaisons', '',
        '±1,6 point n’est pas un seuil universel. Pour une différence de justesse '
        'appariée, la variance dépend des gains et pertes de cette comparaison '
        'précise. Exemple hypothétique : 13 gains et aucune perte sur 1 319 '
        'questions donnent +0,99 point avec un test exact de McNemar bilatéral '
        'p = 2/2^13 ≈ 0,000244. On peut donc résoudre moins d’un point quand les '
        'discordances sont assez asymétriques. Ici, la raison de ne pas développer '
        'le routeur est l’absence de signal directionnel validé, pas une impossibilité '
        'statistique générale.', '',
        '## Reproduction et intégrité', '',
        '```bash', 'python3 src/scripts/audit_calendar_advantage.py',
        'python3 -m unittest discover -s tests -v', '```', '',
        f"{len(r['source_sha256'])} fichiers sources ont été vérifiés inchangés. "
        f"{r['method']['permutations']} permutations et {r['method']['bootstrap_draws']} bootstraps. "
        'Détails, empreintes, folds et prédictions par question : '
        '`results/calendar_advantage_audit/ADVANTAGE_AUDIT.json`.', '']
    return '\n'.join(lines)


def run(base, tokenizer_dir, out, permutations=1999, draws=5000):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True,
                                              trust_remote_code=False)
    eot = tokenizer.convert_tokens_to_ids('<|eot_id|>')
    protected = {}
    rows, termination, tables = [], {}, {}
    for cohort, folder in enumerate(['calendar_swap_v2_main_data', 'calendar_replication_v1']):
        root = base/folder
        paths = sorted((root/'main').glob('*.json')) + [root/'parser_v4_analysis.json']
        protected.update({str(p): sha(p) for p in paths})
        audit = json.loads((root/'parser_v4_analysis.json').read_text())
        decisions = {(d['question_id'], d['arm']): d for d in audit['decisions']}
        counts, eos_hist, eot_hist, arm_rows = Counter(), Counter(), Counter(), []
        cohort_rows = []
        for path in sorted((root/'main').glob('question_*.json')):
            raw = json.loads(path.read_text())
            answers, scores, short_scores, both = [], [], [], True
            boundary = False
            for arm in ['baseline', 'treated']:
                ids = raw[arm+'_ids']
                end = next((i for i,t in enumerate(ids) if t in raw['eos_ids']), len(ids))
                stop = next((i for i,t in enumerate(ids) if t in set(raw['eos_ids']) | {eot}),len(ids))
                eot_index = next((i for i,t in enumerate(ids) if t == eot), len(ids))
                text = tokenizer.decode(ids[:end], skip_special_tokens=True)
                assert text == raw[arm]['text'], (raw['question_id'], arm, 'decode mismatch')
                pred = extract(text, raw['question']) if end < len(ids) else {'answer':None}
                d = decisions[raw['question_id'], arm]
                assert pred['answer'] == d['answer'], (raw['question_id'],arm,'v4 mismatch')
                correct = pred['answer'] is not None and pred['answer'] == raw['gold']
                assert correct == d['new_correct']
                short = tokenizer.decode(ids[:stop], skip_special_tokens=True)
                full = tokenizer.decode(ids, skip_special_tokens=True)
                sp = extract(short, raw['question']) if stop < len(ids) else {'answer':None}
                fp = extract(full, raw['question'])
                counts['outputs'] += 1
                counts['missing_eos'] += end == len(ids)
                counts['eos_at_last_position'] += end == len(ids)-1
                counts['first_terminator_last_two_positions'] += stop >= len(ids)-2
                counts['eot_before_eos'] += eot_index < end
                counts['text_changed_at_eot'] += text != short
                counts['answer_changed_at_eot'] += pred['answer'] != sp['answer']
                counts['text_changed_full_buffer'] += text != full
                counts['answer_changed_full_buffer'] += pred['answer'] != fp['answer']
                eos_hist[str(end)] += 1
                eot_hist[str(stop)] += 1
                boundary |= stop >= len(ids)-2
                arm_rows.append(dict(question_id=raw['question_id'], arm=arm,
                    first_eos=end, first_terminator=stop, eos_at_last_position=end == len(ids)-1,
                    text_changed_at_eot=text != short, answer_changed_at_eot=pred['answer'] != sp['answer'],
                    v4_answer=pred['answer'], early_terminator_answer=sp['answer'],
                    full_buffer_answer=fp['answer'], final_text_tail=text[-180:]))
                answers.append(pred['answer']); scores.append(correct)
                short_scores.append(sp['answer'] is not None and sp['answer'] == raw['gold'])
                both &= pred['answer'] is not None
            s = raw['swap']
            row = dict(question_id=raw['question_id'], cohort=cohort, both_extracted=both,
                a=scores[0], b=scores[1], a_stop=short_scores[0], b_stop=short_scores[1],
                terminal_boundary=boundary,
                x=[s['deferred_confidence'],s['advanced_confidence'],
                   s['deferred_confidence']-s['advanced_confidence'],
                   abs(s['deferred_position']-s['advanced_position'])])
            rows.append(row); cohort_rows.append(row)
        termination[folder] = dict(counts=dict(counts), first_eos_histogram=dict(eos_hist),
            first_terminator_histogram=dict(eot_hist), outputs=arm_rows)
        tables[folder] = dict(v4=ceiling([r['a'] for r in cohort_rows],[r['b'] for r in cohort_rows]),
            first_eot_or_eos=ceiling([r['a_stop'] for r in cohort_rows],[r['b_stop'] for r in cohort_rows]),
            both_extracted=ceiling([r['a'] for r in cohort_rows if r['both_extracted']],
                                   [r['b'] for r in cohort_rows if r['both_extracted']]))
    assert len(rows) == 1000 and len({r['question_id'] for r in rows}) == 1000
    x = np.array([r['x'] for r in rows])
    a,b = np.array([r['a'] for r in rows]), np.array([r['b'] for r in rows])
    cohorts = np.array([r['cohort'] for r in rows])
    folds = np.zeros(len(rows), int)
    for c in [0,1]:
        indices = sorted(np.flatnonzero(cohorts == c), key=lambda i:hashlib.sha256(
            f'calendar-direction-audit-v1:{rows[i]["question_id"]}'.encode()).hexdigest())
        for j,i in enumerate(indices): folds[i] = j%5
    both = np.array([r['both_extracted'] for r in rows])
    tables['pooled'] = dict(v4=ceiling(a,b), both_extracted=ceiling(a[both],b[both]))
    a_stop,b_stop=np.array([r['a_stop'] for r in rows]),np.array([r['b_stop'] for r in rows])
    termination_labels_unchanged = bool(np.array_equal(b.astype(int)-a.astype(int),
                                                      b_stop.astype(int)-a_stop.astype(int)))
    tables['pooled']['first_eot_or_eos']=ceiling(a_stop,b_stop)
    analyses = {}
    for name, mask in [('v4_all', np.ones(len(rows),bool)), ('both_extracted',both)]:
        print(f'Analysing {name}: {int(mask.sum())} pairs', flush=True)
        analyses[name] = probe(x[mask],a[mask],b[mask],folds[mask],cohorts[mask],permutations,draws)
        for entry, i in zip(analyses[name].pop('oof'),np.flatnonzero(mask)):
            rows[i].setdefault('oof',{})[name] = entry
    for i,r in enumerate(rows):r['fold'] = int(folds[i])
    assert all(sha(Path(p)) == h for p,h in protected.items())
    result = dict(scope='Exploratory offline diagnostic on previously inspected test questions; not confirmation',
        method=dict(seed=SEED,features=FEATURES,folds=5,
            fold_assignment='Hash of question ID, balanced within cohort, independent of labels',
            estimator='One separate ridge logistic model per feature; slope penalty 1, train-fold standardization; probability threshold 0.5',
            training='Only discordant training questions; applied to every held-out question without future information',
            permutation='Refit every fold; permute gain/loss among discordant questions within each cohort',
            permutations=permutations,primary_statistic='OOF balanced accuracy of direction among discordant questions',
            multiplicity='Holm across four features separately for each endpoint; second endpoint is sensitivity only',
            bootstrap_draws=draws,bootstrap='Stratified by cohort, fixed OOF predictions; excludes model-fitting uncertainty'),
        source_sha256=protected,script_sha256=sha(Path(__file__)),
        tokenizer_sha256={str(p):sha(p) for p in tokenizer_dir.glob('*.json')},
        ceilings=tables,termination_audit=termination,
        direction_labels_unchanged_with_eot=termination_labels_unchanged,analyses=analyses,rows=rows,
        limitations=['Missing extraction is scored incorrect in frozen v4, not proven semantically wrong.',
            'Terminator presence and boundary position do not certify semantic completion or prove truncation.',
            'Both-extracted subset is selected on outputs and cannot estimate a population-wide router benefit.',
            'Only these four monotonic one-feature probes and this single intervention are tested.',
            'A negative result is not proof that direction is unpredictable by every possible policy.',
            'The 1000 questions have been inspected; cross-validation does not make this confirmatory.'])
    out.mkdir(parents=True,exist_ok=True)
    (out/'ADVANTAGE_AUDIT.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    (out/'ADVANTAGE_AUDIT.md').write_text(report(result))
    print(json.dumps(dict(ceilings=tables,termination={k:v['counts'] for k,v in termination.items()},
        analyses={k:{a:b for a,b in v.items() if a!='cohort_transfer'} for k,v in analyses.items()}),indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=Path('results'))
    p.add_argument('--tokenizer',type=Path,default=Path('results/llada_tokenizer'))
    p.add_argument('--out',type=Path,default=Path('results/calendar_advantage_audit'))
    p.add_argument('--permutations',type=int,default=1999)
    p.add_argument('--bootstraps',type=int,default=5000)
    args=p.parse_args()
    if args.permutations<99 or args.bootstraps<100:p.error('Use at least 99 permutations and 100 bootstraps')
    run(args.results,args.tokenizer,args.out,args.permutations,args.bootstraps)
