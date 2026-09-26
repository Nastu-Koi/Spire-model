"""Schedule early decision histories without merging or pruning game states."""
from collections import defaultdict
from weakref import WeakKeyDictionary

from model.protocol import fingerprint

EARLY_VERBS = frozenset(('CHOOSE_EVENT_OPTION', 'MOVE_TO_NODE', 'TAKE_CARD_REWARD',
                         'SKIP', 'SKIP_REWARDS', 'CHOOSE_REWARD_ALTERNATIVE'))


class RouteDiversity:
    def __init__(self, visits=None):
        self.visits = dict(visits or {})
        self._cache = WeakKeyDictionary()
        self._seen = set()

    @staticmethod
    def _extend(keys, action, act, floor):
        if len(keys) >= 3 or act != 1 or floor > 5 or action.get('verb') not in EARLY_VERBS:
            return keys
        token = fingerprint(action)
        return keys + ((keys[-1] + '/' if keys else '') + token,)

    def keys(self, prefix, pending=None):
        chain, node = [], prefix
        while node is not None and node not in self._cache:
            chain.append(node); node = node.parent
        keys = () if node is None else self._cache[node]
        for node in reversed(chain):
            if node.actor != 'forced':
                keys = self._extend(keys, node.action, node.act, node.floor)
            self._cache[node] = keys
        if pending is not None:
            keys = self._extend(keys, pending, prefix.act if prefix else 1, prefix.floor if prefix else 0)
        return keys

    def begin(self):
        self._seen.clear()

    def observe(self, node):
        for key in self.keys(node[3], node[4]):
            if key not in self._seen:
                self.visits[key] = self.visits.get(key, 0) + 1
                self._seen.add(key)

    def rank(self, node):
        counts = [self.visits.get(k, 0) for k in self.keys(node[3], node[4])]
        counts += [counts[-1] if counts else 0] * (3-len(counts))
        return (*counts, node[0], node[1])


def partition_routes(frontier, lanes):
    """Keep distinct opening subtrees apart; split deeper only to fill idle lanes."""
    diversity = RouteDiversity()
    groups = defaultdict(list)
    for node in sorted(frontier, key=lambda n: (n[0], n[1])):
        keys = diversity.keys(node[3], node[4])
        groups[keys[:1]].append(node)
    groups = list(groups.values())
    # When openings are fewer than workers, split by the next early decision.
    for depth in (2, 3):
        if len(groups) >= lanes: break
        refined = []
        for group in groups:
            parts = defaultdict(list)
            for node in group:
                parts[diversity.keys(node[3], node[4])[:depth]].append(node)
            refined.extend(parts.values())
        groups = refined
    # Identical early routes can still have disjoint later pending branches.
    while len(groups) < lanes:
        index = max(range(len(groups)), key=lambda i: len(groups[i]), default=None)
        if index is None or len(groups[index]) < 2: break
        group = groups.pop(index)
        groups.extend((group[::2], group[1::2]))
    shards = [[] for _ in range(lanes)]
    for group in sorted(groups, key=lambda g: (-len(g), g[0][0], g[0][1])):
        index = min(range(lanes), key=lambda i: (len(shards[i]), i))
        shards[index].extend(group)
    return shards
