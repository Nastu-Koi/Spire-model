"""UCT search over native run histories, with cached combat macros and rollouts.

Nodes never merge public observations. Each evaluated node keeps its exact native
history; simulations reuse that history rather than resampling an old combat plan.
"""
from dataclasses import dataclass, field
import math
from pathlib import Path
import json
import random
import time

from model.protocol import CHARACTERS, action_semantics, fingerprint
from .astar import (Prefix, ReplayWorker, SearchLimit, from_records, load_frontier,
                    player, preference, state_key, write_json)
from .client import InfrastructureError


@dataclass
class Node:
    parent: int | None
    prefix: Prefix | None
    pending: dict | None
    prior: float = 0.
    children: list = field(default_factory=list)
    expanded: bool = False
    closed: bool = False
    terminal: str | None = None
    visits: int = 0
    value_sum: float = 0.
    value: float = 0.
    decision: dict | None = None


class Tree:
    def __init__(self, rng_seed=0, exploration=math.sqrt(2)):
        self.nodes = []
        self.roots = []
        self.rng = random.Random(rng_seed)
        self.exploration = exploration
        self.repair_path = []
        self.repair_due = True
        self.failure_counts = {}

    def add(self, parent, prefix, pending, prior=0.):
        index = len(self.nodes)
        self.nodes.append(Node(parent, prefix, pending, prior))
        (self.roots if parent is None else self.nodes[parent].children).append(index)
        return index

    def choose(self, children, visits):
        available = [i for i in children if not self.nodes[i].closed]
        if not available: return None
        def score(i):
            n = self.nodes[i]
            if not n.visits: return (math.inf, n.prior, self.rng.random())
            uct = n.value_sum/n.visits + self.exploration*math.sqrt(math.log(max(2, visits))/n.visits)
            # A diminishing public-action prior; Q remains a backed-up rollout value.
            uct += .1*math.tanh(n.prior/4)/(n.visits+1)
            return (uct, n.prior, self.rng.random())
        return max(available, key=score)

    def schedule_repair(self, path, act, floor):
        key = f'{act}:{floor}'
        count = self.failure_counts.get(key, 0) + 1
        self.failure_counts[key] = count
        distance = min(8, 1 << min(3, (count-1)//2))
        choices = []
        for offset in range(len(path)-2, -1, -1):
            node = self.nodes[path[offset]]
            if node.decision and any(not self.nodes[i].expanded and not self.nodes[i].closed for i in node.children):
                choices.append((offset, node.decision))
        # Move further back after repeated failures at the same location. Never
        # discard siblings or infer equality from matching public observations.
        eligible = [(i,d) for i,d in choices if d['act'] < act or
                    (d['act'] == act and d['floor'] <= floor-distance)]
        target = next(iter(eligible or choices), None)
        self.repair_path = path[:target[0]+1] if target else []

    def select(self, local_repair=True):
        if local_repair and self.repair_due and self.repair_path:
            path = self.repair_path[:]
            node = self.nodes[path[-1]]
            children = [i for i in node.children if not self.nodes[i].expanded and not self.nodes[i].closed]
            child = self.choose(children, node.visits)
            if child is not None:
                self.repair_due = False
                return path + [child]
            self.repair_path = []
        self.repair_due = True
        index = self.choose(self.roots, sum(self.nodes[i].visits for i in self.roots))
        path = []
        while index is not None:
            path.append(index)
            n = self.nodes[index]
            if not n.expanded or n.terminal is not None: return path
            index = self.choose(n.children, n.visits)
        return path

    def rollout_child(self, index, epsilon, frame=None):
        available = [i for i in self.nodes[index].children if not self.nodes[i].closed]
        if not available: return None
        best = max(self.nodes[i].prior for i in available)
        pool = available
        if frame is not None:
            pool = [i for i in available if self.nodes[i].prior >= best-.75]
            p = player(frame)
            verbs = {self.nodes[i].pending['verb'] for i in available}
            context = (frame['public'].get('selection_context') or {}).get('operation', '').lower()
            entities = frame['public'].get('entities', [])
            current = {e['ref'] for e in entities if e.get('entity_type') == 'map_node' and e.get('current')}
            bosses = {e['ref'] for e in entities if e.get('entity_type') == 'map_node' and e.get('content_id', '').upper() == 'BOSS'}
            boss_next = any(r.get('role') == 'map_edge' and r.get('source') in current and r.get('target') in bosses
                            for r in frame['public'].get('relations', []))
            critical = p['hp']/max(1,p['max_hp']) < .4 or boss_next or 'CHOOSE_REST_OPTION' in verbs or context in ('remove','exhaust')
            if critical: epsilon *= .2
        if self.rng.random() < epsilon: return self.rng.choice(pool)
        return self.rng.choice([i for i in available if self.nodes[i].prior == best])

    def backup(self, path, value):
        for index in reversed(path):
            n = self.nodes[index]
            n.visits += 1
            n.value_sum += value
            if n.expanded and n.children and all(self.nodes[i].closed for i in n.children):
                n.closed = True


def rollout_value(frame, last_player=None):
    if frame['boundary'] == 'terminal' and frame['public']['outcome']['victory']:
        return 1.
    p = next((e for e in frame['public'].get('entities', []) if e.get('entity_type') == 'player'), last_player or {})
    progress = max(0, p.get('act', 1)-1)*20 + p.get('floor', 0)
    health = 0 if frame['boundary'] == 'terminal' else p.get('hp', 0)/max(1, p.get('max_hp', 1))
    # Shaped progress, not an estimate of win probability. Victory is strictly higher.
    return min(.9, max(0., .8*progress/60 + .05*health))


def save_tree(path, tree, identity, stats):
    indices, prefixes = {}, []
    def encode(node):
        chain = []
        while node is not None and id(node) not in indices:
            chain.append(node); node = node.parent
        parent = -1 if node is None else indices[id(node)]
        for node in reversed(chain):
            indices[id(node)] = len(prefixes)
            prefixes.append(dict(parent=parent, before_hash=node.before_hash, action=node.action,
                                 actor=node.actor, length=node.length, phase=node.phase, act=node.act, floor=node.floor))
            parent = len(prefixes)-1
        return parent
    nodes = []
    for n in tree.nodes:
        nodes.append(dict(parent=n.parent, prefix=encode(n.prefix), pending=n.pending,
                          prior=n.prior, children=n.children, expanded=n.expanded, closed=n.closed,
                          terminal=n.terminal, visits=n.visits, value_sum=n.value_sum, value=n.value, decision=n.decision))
    write_json(path, dict(schema='mcts-tree-v1', identity=identity, prefixes=prefixes,
                         nodes=nodes, roots=tree.roots, rng_state=tree.rng.getstate(), stats=stats,
                         repair_path=tree.repair_path, repair_due=tree.repair_due, failure_counts=tree.failure_counts))


def load_tree(path, identity, exploration):
    data = json.loads(Path(path).read_text())
    if data['schema'] != 'mcts-tree-v1' or data['identity'] != identity:
        raise ValueError('MCTS checkpoint identity or policy differs')
    prefixes = []
    for p in data['prefixes']:
        prefixes.append(Prefix(None if p['parent'] == -1 else prefixes[p['parent']],
                               p['before_hash'], p['action'], p['actor'], p['length'],
                               p['phase'], p['act'], p['floor']))
    tree = Tree(exploration=exploration)
    for raw in data['nodes']:
        n = dict(raw); n['prefix'] = None if n['prefix'] == -1 else prefixes[n['prefix']]
        tree.nodes.append(Node(**n))
    tree.roots = data['roots']
    tree.repair_path = data.get('repair_path', [])
    tree.repair_due = data.get('repair_due', True)
    tree.failure_counts = data.get('failure_counts', {})
    def tuples(value): return tuple(tuples(x) for x in value) if isinstance(value, list) else value
    tree.rng.setstate(tuples(data['rng_state']))
    return tree, data.get('stats', {})


def search(config, character, seed, output, *, budget_ms=1000, boss_budget_ms=5000,
           weight=5., max_expansions=2000, max_seconds=3600, max_steps=10000,
           resume=None, prefix_path=None, reuse_turn_plan=False, rollout_decisions=256,
           ascension=10, search_lanes=1, early_route_diversity=True,
           exploration=math.sqrt(2), rollout_epsilon=.1, lane_index=0,
           local_repair=True, risk_aware_rollout=True):
    if character not in CHARACTERS: raise ValueError('Unsupported character')
    if type(ascension) is not int or not 0 <= ascension <= 10: raise ValueError('Invalid ascension')
    if search_lanes not in (1,2,4): raise ValueError('search_lanes must be 1, 2 or 4')
    if any(not math.isfinite(x) or x <= 0 for x in (budget_ms,boss_budget_ms,weight,max_expansions,max_seconds,max_steps,exploration)):
        raise ValueError('Search budgets must be finite and positive')
    if type(boss_budget_ms) is not int or max(budget_ms,boss_budget_ms) > 120000:
        raise ValueError('Invalid combat budget')
    if type(rollout_decisions) is not int or rollout_decisions < 0: raise ValueError('Invalid rollout horizon')
    if not math.isfinite(rollout_epsilon) or not 0 <= rollout_epsilon <= 1: raise ValueError('Invalid rollout epsilon')
    if resume and prefix_path: raise ValueError('Choose either a tree or a prefix')
    options = dict(budget_ms=budget_ms,boss_budget_ms=boss_budget_ms,weight=weight,
                   max_expansions=max_expansions,max_seconds=max_seconds,max_steps=max_steps,
                   reuse_turn_plan=reuse_turn_plan,rollout_decisions=rollout_decisions,
                   ascension=ascension,early_route_diversity=early_route_diversity,
                   exploration=exploration,rollout_epsilon=rollout_epsilon,
                   local_repair=local_repair,risk_aware_rollout=risk_aware_rollout)
    if search_lanes > 1:
        from .mcts_lanes import search_parallel
        return search_parallel(config,character,seed,output,search_lanes=search_lanes,
                               resume=resume,prefix_path=prefix_path,**options)
    identity = dict(character=character,seed=seed,ascension=ascension,
                    exploration=exploration,rollout_epsilon=rollout_epsilon,rollout_decisions=rollout_decisions)
    if local_repair or risk_aware_rollout:
        identity.update(policy='repair_risk_v2',local_repair=local_repair,risk_aware_rollout=risk_aware_rollout)
    algorithm = 'mcts_uct_repair_risk_v2' if local_repair or risk_aware_rollout else 'mcts_uct_history_v1'
    tree = Tree(int(fingerprint([seed,'mcts',lane_index])[:16],16),exploration)
    prior_stats = {}
    if resume:
        schema=json.loads(Path(resume).read_text())['schema']
        if schema == 'mcts-tree-v1':
            tree,prior_stats=load_tree(resume,identity,exploration)
        elif schema == 'astar-frontier-v2':
            queue=load_frontier(resume,character,seed,weight,ascension)
            for f,_,_,prefix,pending in sorted(queue,key=lambda n:(n[0],n[1])):
                tree.add(None,prefix,pending,-f/weight)
        else: raise ValueError('Single-lane MCTS requires a tree or initial frontier')
    elif prefix_path:
        data=json.loads(Path(prefix_path).read_text())
        if (data['character'],data['seed'],data.get('ascension',10)) != (character,seed,ascension):
            raise ValueError('Prefix identity differs')
        tree.add(None,from_records(data['records']),None)
    else: tree.add(None,None,None)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    worker=ReplayWorker(config,character,seed,budget_ms,started+max_seconds,max_steps,reuse_turn_plan,ascension,boss_budget_ms)
    worker.stop_path=output/'STOP'
    expanded=deaths=errors=simulations=0
    best=(0,0);winner=next((n.prefix for n in tree.nodes if n.terminal=='victory'),None)
    status='budget_exhausted';reason=None
    def checkpoint():
        save_tree(output/'tree.json',tree,identity,
                  dict(simulations=prior_stats.get('simulations',0)+simulations,
                       expanded=prior_stats.get('expanded',0)+expanded))
    with (output/'search.jsonl').open('w') as journal:
        def log(event):
            event.update(seed=seed,character=character,ascension=ascension)
            journal.write(json.dumps(event,ensure_ascii=False)+'\n');journal.flush()
        try:
            while winner is None and expanded < max_expansions:
                worker.check()
                path=tree.select(local_repair)
                if not path:
                    status='tree_exhausted_with_unresolved' if any(n.terminal=='unresolved' for n in tree.nodes) else 'tree_exhausted';break
                value=0.
                for depth in range(max(1,rollout_decisions)):
                    index=path[-1];node=tree.nodes[index]
                    if node.terminal=='victory': winner=node.prefix;break
                    if node.expanded:
                        raise RuntimeError('MCTS selected an expanded node without a live child')
                    expanded+=1
                    try:
                        worker.restore(node.prefix)
                        if node.pending is not None: worker.execute(node.pending,'mcts')
                        frame=worker.combat_and_forced()
                        node.prefix=worker.prefix;node.pending=None;node.expanded=True
                        node.value=value=rollout_value(frame,worker.last_player)
                        if frame['boundary']=='terminal':
                            if frame['public']['outcome']['victory']:
                                node.terminal='victory';winner=node.prefix
                            else:
                                node.terminal='defeat';node.closed=True;deaths+=1
                                if local_repair:
                                    tree.schedule_repair(path,worker.last_player.get('act',1),worker.last_player.get('floor',0))
                            log(dict(expanded=expanded,simulation=simulations,result=node.terminal,
                                     act=worker.last_player.get('act'),floor=worker.last_player.get('floor')))
                            break
                        p=player(frame);progress=(p.get('act',1),p.get('floor',0))
                        node.decision=dict(act=progress[0],floor=progress[1],phase=frame['public']['phase'])
                        if progress > best:
                            best=progress
                            write_json(output/'best_prefix.json',dict(character=character,seed=seed,
                                ascension=ascension,records=node.prefix.records() if node.prefix else [],progress=best))
                        observation_key=state_key(frame)
                        candidates=sorted(frame['legal']['candidates'],key=lambda c:fingerprint([seed,observation_key,action_semantics(c)]))
                        for candidate in candidates:
                            if candidate['verb']!='ABANDON_RUN':
                                tree.add(index,node.prefix,action_semantics(candidate),preference(frame,candidate))
                        if not node.children: raise ValueError('Decision has no non-abandon candidates')
                        log(dict(expanded=expanded,simulation=simulations,act=progress[0],floor=progress[1],
                                 hp=p['hp'],phase=frame['public']['phase'],nodes=len(tree.nodes),value=value))
                    except (InfrastructureError,SearchLimit): raise
                    except (RuntimeError,ValueError,KeyError) as exc:
                        errors+=1;node.closed=True;node.terminal='unresolved';value=0.
                        log(dict(expanded=expanded,error=str(exc)))
                        directory=output/'unresolved';directory.mkdir(exist_ok=True)
                        filename=directory/(str(index)+'.json')
                        write_json(filename,dict(character=character,seed=seed,ascension=ascension,
                            records=node.prefix.records() if node.prefix else [],pending_action=node.pending,error=str(exc)))
                        diagnostics=worker.native_diagnostics()
                        if diagnostics: filename.with_suffix('.stderr.log').write_text(diagnostics)
                        worker.close();break
                    if expanded >= max_expansions or depth+1 >= max(1,rollout_decisions): break
                    child=tree.rollout_child(index,rollout_epsilon,frame if risk_aware_rollout else None)
                    if child is None: break
                    path.append(child)
                tree.backup(path,1. if winner is not None else value)
                simulations+=1
                if winner is not None: status='candidate_victory';break
                checkpoint()
        except InfrastructureError as exc:
            status,reason='infrastructure_error',str(exc)
        except (SearchLimit,KeyboardInterrupt) as exc:
            reason=str(exc) or 'interrupted'
            if isinstance(exc,KeyboardInterrupt) or 'STOP' in reason: status='stopped'
        finally:
            worker.close();checkpoint()
    summary=dict(status=status,character=character,seed=seed,ascension=ascension,
                 algorithm=algorithm,local_repair=local_repair,risk_aware_rollout=risk_aware_rollout,expanded=expanded,simulations=simulations,
                 deaths=deaths,unresolved_branches=errors,nodes=len(tree.nodes),best_progress=best,
                 solver_steps=worker.solver_steps,replayed_steps=worker.replayed,
                 budget_ms=budget_ms,boss_budget_ms=boss_budget_ms,reuse_turn_plan=reuse_turn_plan,
                 rollout_decisions=rollout_decisions,exploration=exploration,rollout_epsilon=rollout_epsilon,
                 search_seconds=time.monotonic()-started,stop_reason=reason,optimality_proven=False)
    if winner is not None:
        write_json(output/'winning_prefix.json',dict(character=character,seed=seed,ascension=ascension,
            records=winner.records(),search={k:summary[k] for k in ('algorithm','budget_ms','boss_budget_ms','reuse_turn_plan','rollout_decisions','exploration','rollout_epsilon','local_repair','risk_aware_rollout')}))
        try:
            from .trajectory import verify_and_export
            verify_and_export(config,output/'winning_prefix.json',output)
            summary['status']='verified_victory'
        except Exception as exc:
            summary.update(status='replay_failed',verification_error=str(exc))
    summary['seconds']=time.monotonic()-started
    write_json(output/'summary.json',summary)
    return summary
