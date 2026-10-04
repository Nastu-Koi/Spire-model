"""Crystal Sphere is an environment rule, never a policy training label."""
import random

from .protocol import ProtocolError

RULE_VERSION = 'crystal-small-three-v1'


def collection_candidate(frame, rng=None):
    public = frame['public']
    entities = {e.get('ref'): e for e in public['entities'] if e.get('ref')}
    candidates = frame['legal']['candidates']
    if public['phase'] == 'crystal_sphere':
        remaining = next((e.get('remaining_steps') for e in public['entities']
                          if e.get('entity_type') == 'event_state'), None)
        if remaining is not None and not 1 <= remaining <= 3:
            raise ProtocolError('Crystal Sphere is not the three-click branch')
        choices = [c for c in candidates if c['verb'] == 'DIVINE_CELL'
                   and any(entities.get(ref, {}).get('content_id') == 'Small'
                           for ref in c.get('source_refs', [c.get('source_ref')]))]
        if not choices:
            raise ProtocolError('Crystal Sphere has no legal 1x1 cell')
        # Native engine decrements divination count, reveals the cell, and exits
        # after three clicks. Never consult hidden contents or consume game RNG.
        return (rng or random).choice(choices)
    if public['phase'] == 'event' and any(
            e.get('entity_type') == 'event'
            and e.get('content_id', '').split('.')[-1] == 'CRYSTAL_SPHERE'
            for e in public['entities']):
        choices = [c for c in candidates if c['verb'] == 'CHOOSE_EVENT_OPTION']
        first = [c for c in choices if any(ref == 'option:0' for ref in c.get('source_refs', [c.get('source_ref')]))]
        if first:
            return first[0]
        raise ProtocolError('Crystal Sphere has no available event option')
    return None
