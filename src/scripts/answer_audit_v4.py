"""Separate post-hoc v4 conclusion extraction; frozen v3 is not modified.

Does not alter the frozen v2 collector. Unit targets come from the question,
never the reference answer. Unsupported or ambiguous conclusions stay unresolved.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re

from calendar_swap import NUMBER, analyse, digest, load_rows, save_json

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


def audit(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    originals = load_rows(directory, manifest['question_ids'])
    lookup = {r['question_id']: r for r in originals}
    originals = [lookup[i] for i in manifest['question_ids'] if i in lookup]
    rows = copy.deepcopy(originals)
    decisions = []
    for row in rows:
        for arm in ('baseline', 'treated'):
            old = row[arm]
            result = ({'answer': None, 'reason': 'length_limit'} if old['status'] == 'length_limit'
                      else extract(old['text'], row['question']))
            answer = result['answer']
            decisions.append(dict(result, question_id=row['question_id'], arm=arm,
                                  old_answer=old['answer'], old_correct=old['correct'],
                                  new_correct=answer is not None and answer == row['gold']))
            row[arm] = dict(old, answer=answer, correct=answer is not None and answer == row['gold'],
                            status='valid' if answer is not None else result['reason'])
    transitions = Counter()
    for row in rows:
        a, b = row['baseline'], row['treated']
        if a['correct'] != b['correct']:
            kind = 'both_extracted' if a['answer'] is not None and b['answer'] is not None else 'extraction_transition'
            transitions[('loss_' if a['correct'] else 'gain_') + kind] += 1
    return {'parser_version': VERSION, 'scope': 'post-hoc evaluation audit; no new generation',
            'source_manifest_sha256': digest(manifest), 'source_records_sha256': digest(originals),
            'audit_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'original_summary': analyse(originals, manifest['question_ids']),
            'revised_summary': analyse(rows, manifest['question_ids']),
            'decisions': decisions, 'reasons': dict(Counter(d['reason'] for d in decisions)),
            'correctness_transitions': dict(transitions)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if args.out.resolve().parent == args.directory.resolve():
        parser.error('write the audit outside the original records directory')
    result = audit(args.directory)
    save_json(args.out, result)
    print(json.dumps({k: result[k] for k in ('revised_summary', 'reasons', 'correctness_transitions')}, indent=2))
