"""Recorded source evidence, not a certification of an unmodified client.

Resume snapshots are checked for required fields; native branch reconciliation
is a separate requirement. A remap marked exact does not prove state equality.
"""
from collections import Counter


MUTATING_COMMANDS = set('draw heal win energy remove_card damage ancient'.split())


def console_kind(row):
    # Verified against v0.111.0 DevConsole and AncientConsoleCmd. Unknown mod
    # commands (including mprint) require implementation evidence, not a guess.
    cmd = row.get('cmd')
    if not isinstance(cmd, str):
        return 'unknown_command'
    if cmd == 'help':
        return 'read_only'
    if cmd == 'ancient' and row.get('args', []) == []:
        return 'invalid_no_effect'
    return 'state_mutation_command' if cmd in MUTATING_COMMANDS else 'unknown_command'


def inspect_journal(journal):
    commands = [dict(s=r.get('s'), cmd=r.get('cmd'), args=r.get('args', []),
                     kind=console_kind(r)) for r in journal if r.get('t') == 'console']
    resumes = []
    headers = [r for r in journal if r.get('t') == 'header']
    for i, row in enumerate(journal):
        if row.get('t') != 'resume':
            continue
        header = journal[i-1] if i else {}
        end = journal[i-2] if i > 1 else {}
        missing = []
        if header.get('t') != 'header':
            missing.append('preceding_header')
            header = {}
        for key in ('hp', 'gold', 'deck_size', 'act', 'floor'):
            if type(row.get(key)) is not int:
                missing.append(key)
        if not isinstance(row.get('rng_state'), dict) or not row['rng_state']:
            missing.append('rng_state')
        for key in ('starting_deck', 'starting_relics'):
            if not isinstance(header.get(key), list):
                missing.append(key)
        if type(header.get('starting_max_hp')) is not int:
            missing.append('max_hp')
        for key in ('relics', 'potions'):
            if not isinstance(row.get(key, header.get(key)), list):
                missing.append(key)
        following = []
        for line in journal[i+1:]:
            if line.get('t') not in ('remap', 'deck'):
                break
            following.append(line)
        remap = next((r for r in following if r['t'] == 'remap'), {})
        deck = next((r.get('cards') for r in following if r['t'] == 'deck'), None)
        if remap.get('exact') is not True:
            missing.append('exact_card_remap')
        if not isinstance(deck, list) or len(deck) != row.get('deck_size'):
            missing.append('matching_deck_snapshot')
        elif deck != header.get('starting_deck'):
            missing.append('header_deck_consistency')
        resumes.append(dict(s=row.get('s'), act=row.get('act'), floor=row.get('floor'),
            missing=missing, snapshot_fields_present=not missing,
            previous_end_present=end.get('t') == 'end',
            hp_changed=(end.get('hp') != row.get('hp')) if end.get('t') == 'end' else None,
            previous_hp=end.get('hp') if end.get('t') == 'end' else None,
            resumed_hp=row.get('hp'), remap_exact=remap.get('exact'),
            native_state_verified=False))
    reasons = []
    if any(c['kind'] == 'state_mutation_command' for c in commands):
        reasons.append('console_state_mutation_commands')
    if any(c['kind'] == 'unknown_command' for c in commands):
        reasons.append('console_effect_unknown')
    # Additional headers must belong to a resume boundary. This is structural
    # evidence only; never splice abandoned branches into historical labels.
    if len(headers) != 1 + len(resumes) or any('preceding_header' in r['missing'] for r in resumes):
        reasons.append('resume_segment_structure_invalid')
    if resumes:
        reasons.append('resume_requires_native_reconciliation')
        if any(r['missing'] for r in resumes):
            reasons.append('resume_snapshot_incomplete')
    return dict(console=commands, console_counts=dict(Counter(c['kind'] for c in commands)),
        resumes=resumes, header_count=len(headers), reasons=reasons,
        manifests_changed=any((h.get('mods'), h.get('harmony_owners')) !=
            (headers[0].get('mods'), headers[0].get('harmony_owners')) for h in headers[1:]))
