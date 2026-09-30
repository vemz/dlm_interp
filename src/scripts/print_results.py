import argparse
from decimal import Decimal
from fractions import Fraction
import json
import math
from pathlib import Path
import re

NUMBER = r'[+-]?(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)(?:\.[0-9]+)?'

VERSION = 'conclusion-units-v4'
SCALAR = re.compile(r'(?<![\w.,])' + NUMBER + r'(?![\w,]|\.[0-9])')
UNITS = {
    'dollars': ('money', Decimal(1)), 'cents': ('money', Decimal('.01')),
    'hours': ('time', Decimal(3600)), 'minutes': ('time', Decimal(60)),
    'seconds': ('time', Decimal(1)),
    'feet': ('length', Decimal(12)), 'inches': ('length', Decimal(1)),
    'yards': ('length', Decimal(36)),
    'percent': ('percent', Decimal(1)),
}
ALIASES = {'dollar': 'dollars', 'cent': 'cents', 'hour': 'hours', 'minute': 'minutes',
           'second': 'seconds', 'foot': 'feet', 'inch': 'inches', 'yard': 'yards',
           '%': 'percent'}
UNIT_PATTERN = r'dollars?|cents?|hours?|minutes?|seconds?|feet|foot|inches|inch|yards?|percent|%'


def normalized(value):
    if value == 0:
        return '0'
    text = format(value, 'f')
    return text.rstrip('0').rstrip('.') if '.' in text else text


def requested_unit(question):
    # Use the final question clause, not numbers or units from the worked solution.
    clauses = re.findall(r'[^.!?]*\?', question)
    clause = clauses[-1].lower() if clauses else question.lower().split('.')[-1]
    explicit = re.findall(r'(?:how many|in|in total in)\s+(' + UNIT_PATTERN + r')\b', clause)
    targets = {ALIASES.get(u, u) for u in explicit}
    if 'percentage' in clause or 'what percent' in clause:
        targets.add('percent')
    if len(targets) == 1:
        return targets.pop()
    if targets:
        return None
    money = re.search(r'how much.*\b(?:cost|costs|pay|paid|spend|spends|spent|earn|earns|earned|make|makes|made|money|profit|charge|charges)\b', clause)
    if money and ('$' in question or re.search(r'\b(?:dollars?|cents?)\b', question, re.I)):
        return 'dollars'
    if re.search(r'\b(?:price|cost|bill|amount|pay|paid|spend|profit)\b', clause) and ('$' in question or re.search(r'\bdollars?\b', question, re.I)):
        return 'dollars'
    return None


def explicit_sentence(text):
    """A terminal declarative answer, not a bare number or unfinished calculation."""
    if re.match(r'\s*(?:\d+[.)]\s*|[-*]\s*)?(?:determine|calculate|find|identify|subtract|add|multiply|divide)\b', text, re.I):
        return False
    return bool(re.match(r'(?:therefore|thus|so\b|in total\b|the (?:final )?answer\b)', text, re.I)
                or re.search(r'\b(?:is|are|has|have|needs?|paid|fed|costs?|spends?|produces?|remains?)\b', text, re.I))


def extract(text, question=''):
    """Return answer plus provenance, or a reason for abstention. No gold input."""
    markers = list(re.finditer(r'(?<!#)#{3,4}(?!#)[ \t]*', text))
    if markers:
        conclusion = text[markers[-1].end():].strip()
        source = 'final_marker'
        # Only a known empty placeholder permits looking back one paragraph.
        if conclusion.lower() in ('<answer>', '< the>'):
            prefix = text[:markers[-1].start()].strip()
            paragraphs = [p.strip() for p in re.split(r'\n\s*\n', prefix) if p.strip()]
            candidate = paragraphs[-1] if paragraphs else ''
            if explicit_sentence(candidate):
                conclusion, source = candidate, 'before_empty_marker'
    else:
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        conclusion = paragraphs[-1] if paragraphs else ''
        source = 'final_paragraph'
        if not (explicit_sentence(conclusion)
                or r'\boxed{' in conclusion):
            return {'answer': None, 'reason': 'no_explicit_conclusion', 'conclusion': conclusion}
    info = {'answer': None, 'source': source, 'conclusion': conclusion}
    def reject(reason):
        return dict(info, reason=reason)
    if re.match(r'\s*\d+[.)]\s*\D', conclusion):
        return reject('numbered_step')
    if re.search(r'\b(?:not|never|maybe|might|could|probably|uncertain|approximately|about|either|or|at least|at most|up to)\b', conclusion, re.I):
        return reject('qualified_or_alternative')
    # Do not repair missing closing braces or choose among competing boxed values.
    if conclusion.count('{') != conclusion.count('}'):
        return reject('unbalanced_braces')
    for left, right in ((r'\(', r'\)'), (r'\[', r'\]')):
        if conclusion.count(left) != conclusion.count(right):
            return reject('unbalanced_math_delimiters')
    clean = re.sub(r'\\boxed\{([^{}]*)\}', r'\1', conclusion)
    clean = re.sub(r'\\(?:text|mathrm)\{([^{}]*)\}', r' \1 ', clean)
    clean = re.sub(r'<(?:answer|:)>(?:\s*)', '', clean, flags=re.I)
    clean = re.sub(r'\\[\[\]()]', '', clean)
    clean = clean.replace('**', '').replace('\\$', '$').strip()
    # Put a sign preceding the currency next to the number: -$110 -> $-110.
    clean = re.sub(r'([+-])\s*\$\s*(?=\d)', r'$\1', clean)
    # Only remove a trailing time adjunct, never choose among competing answers.
    clean = re.sub(r'\s+in\s+\d+(?:\.\d+)?\s+hours?\.?$', '', clean, flags=re.I)
    clean = re.sub(r'^(Therefore,?\s*)?after\s+\d+(?:\.\d+)?\s+hours?,\s*', '', clean, flags=re.I)
    if '<' in clean or '>' in clean:
        # Known header syntax is cosmetic; arbitrary numeric tags are not.
        clean = re.sub(r'^Answer>\s*', '', clean, flags=re.I)
        if '<' in clean or '>' in clean:
            return reject('unsupported_angle_markup')
    # Remaining TeX commands, fractions, equations and malformed numeric syntax fail.
    if re.search(r'\\|[{}=/^]|\d\s*[+*]|\d\s*-\s*\d', clean):
        return reject('expression_or_unsupported_markup')
    numbers = list(SCALAR.finditer(clean))
    if len(numbers) != 1:
        return reject('multiple_or_missing_numbers')
    match = numbers[0]
    remainder = clean[:match.start()] + clean[match.end():]
    if re.search(r'\d', remainder):
        return reject('malformed_number')
    # Reject commentary after the numeric conclusion rather than accepting any prose.
    raw_suffix = clean[match.end():]
    suffix = raw_suffix.strip()
    if '\n' in raw_suffix and suffix.strip('. \t\n'):
        return reject('trailing_lines')
    if re.search(r'\b(?:per|each)\b', suffix, re.I):
        return reject('rate_or_compound_unit')
    raw_unit = re.match(r'\s*(' + UNIT_PATTERN + r')(?![A-Za-z])', suffix, re.I)
    currency = clean[:match.start()].rstrip().endswith('$')
    if '£' in clean or '€' in clean:
        return reject('unsupported_currency')
    unit = ALIASES.get(raw_unit[1].lower(), raw_unit[1].lower()) if raw_unit else None
    if currency and unit not in (None, 'dollars'):
        return reject('conflicting_units')
    if currency:
        unit = 'dollars'
    target = requested_unit(question)
    value = Decimal(match[0].replace(',', ''))
    info.update(raw_number=normalized(value), unit=unit, target_unit=target)
    if unit:
        if target is None:
            return reject('unit_target_unresolved')
        dimension, scale = UNITS[unit]
        target_dimension, target_scale = UNITS[target]
        if dimension != target_dimension:
            return reject('unit_dimension_mismatch')
        value = value * scale / target_scale
    return dict(info, answer=normalized(value), reason='accepted', converted=bool(unit and unit != target))



def wilson(count, n):
    """95% interval for a detected event; unresolved answers are a separate category."""
    if not n:
        return dict(count=count, n=n, rate=None, ci95=None)
    z = 1.959963984540054
    p, denominator = count / n, 1 + z*z / n
    centre = (p + z*z / (2*n)) / denominator
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denominator
    return dict(count=count, n=n, rate=p, ci95=[max(0, centre-half), min(1, centre+half)])


def summarize(rows, expected_ids):
    """Automatic extraction scores, not a substitute for review of the final answers."""
    ids = [r['question_id'] for r in rows]
    if len(set(ids)) != len(ids) or not set(ids) <= set(expected_ids):
        raise ValueError('Duplicate or unexpected questions')
    pairs = [(r['baseline'], r['treated']) for r in rows]
    comparable = [(b, t) for b, t in pairs if b['answer'] is not None and t['answer'] is not None]
    changes = sum(Fraction(b['answer']) != Fraction(t['answer']) for b, t in comparable)
    gains = sum(not b['correct'] and t['correct'] for b, t in pairs)
    losses = sum(b['correct'] and not t['correct'] for b, t in pairs)
    return dict(
        complete=set(ids) == set(expected_ids), questions_processed=len(rows), questions_planned=len(expected_ids),
        scope='Frozen v4 extraction scores; primary numeric review pending. Missing extractions are not semantic errors.',
        text_changes=wilson(sum(b['text'] != t['text'] for b, t in pairs), len(rows)),
        numeric_changes_all=wilson(changes, len(rows)),
        numeric_changes_both_extracted=wilson(changes, len(comparable)),
        reference_matches={a: sum(r[a]['correct'] for r in rows) for a in ('baseline', 'treated')},
        abstentions={a: sum(r[a]['answer'] is None for r in rows) for a in ('baseline', 'treated')},
        extraction_score_transitions=dict(gains=gains, losses=losses, net=gains-losses),
        actual_forward_calls=sum(r['nfe_actual_pair'] + r['nfe_controls'] for r in rows),
        all_controls_pass=bool(rows) and all(all(r['checks'].get(k) is True for k in
            ('prompt_unchanged', 'fully_unmasked', 'single_swap', 'no_swap_replay', 'independent_baseline_replay')) for r in rows))


def print_review(data):
    print('Réponses finales — revue conservatrice des expériences historiques')
    print(f"{'Échantillon':<26} {'Changements':>12} {'Non résolus':>12} {'Gains':>7} {'Pertes':>7}")
    for label, row in data['experiments'].items():
        print(f"{label:<26} {row['changes']:>5}/{row['n']:<6} {row['unresolved']:>12} {row['gains']:>7} {row['losses']:>7}")
    print('\nMêmes 500 questions :')
    for model, counts in data['matched']['counts'].items():
        print(f"  {model}: {counts['changed']}/500 changements ({counts['changed']/5:.1f} %), "
              f"{counts['unresolved']} non résolus.")
    print('\nChanger la réponse ne garantit pas un gain : une erreur peut devenir une autre erreur.')
    print('Revue par le même assistant, sans validation humaine indépendante.')
    print('Les inconnues restent séparées ; aucun gain moyen de justesse n’est démontré.')
    print('Sources et empreintes : results/reviewed_results.json ; données complètes dans l’archive.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', nargs='?', type=Path,
                        default=Path(__file__).resolve().parents[2]/'results/reviewed_results.json',
                        help='Reviewed JSON, summary JSON, or compact run phase directory')
    parser.add_argument('--json', action='store_true', help='Print the complete machine-readable summary')
    args = parser.parse_args()
    if not args.path.exists():
        parser.error(f'{args.path} absent. Download a run or restore the local research archive (see README).')
    if args.path.is_dir():
        # Import lazily so printing an existing JSON needs only Python’s standard library.
        from llada import load_saved
        manifest = json.loads((args.path/'manifest.json').read_text())
        data = summarize(load_saved(args.path, manifest), [q['id'] for q in manifest['examples']])
    else:
        data = json.loads(args.path.read_text())
    if data.get('format') == 'reviewed-calendar-results-v1' and not args.json:
        print_review(data)
    else:
        print(json.dumps(data, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
