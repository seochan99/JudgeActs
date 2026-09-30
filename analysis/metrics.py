"""Prompt-level estimands. Random is exact, including nonlinear event probabilities."""
import numpy as np
from scipy.stats import norm

def candidate_metrics(scores, selected):
    scores = np.asarray(scores, dtype=float)
    best, random = float(scores.max()), float(scores.mean())
    chosen = float(scores[selected])
    return {'utility': chosen, 'regret': best-chosen, 'gain': chosen-random,
            'hsr': float(chosen < random-1e-12), 'hsr05':float(chosen < random-.05-1e-12),
            'human_best':float(chosen >= best-1e-12), 'near_best':float(chosen >= best-.05-1e-12),
            'bottom_half':float(chosen < np.median(scores)-1e-12)}

def random_metrics(scores):
    # E[indicator(H_random < E[H_random])] differs from indicator(E[H_random] < E[H_random]).
    rows = [candidate_metrics(scores, i) for i in range(len(scores))]
    return {key:float(np.mean([r[key] for r in rows])) for key in rows[0]}

def bootstrap_mean(values, resamples=10000, seed=20261002):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return [None, None]
    rng = np.random.default_rng(seed)
    means = []
    for start in range(0,resamples,250):
        ix = rng.integers(len(values), size=(min(250,resamples-start),len(values)))
        means.extend(values[ix].mean(axis=1))
    return np.quantile(means,[.025,.975]).tolist()

def wilson(successes, n):
    if n == 0:
        return [None, None]
    z = norm.ppf(.975); p = successes/n
    center = (p+z*z/(2*n))/(1+z*z/n)
    radius = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [float(center-radius),float(center+radius)]

def choice_consistency(ids, candidate_ids):
    counts = np.asarray([ids.count(i) for i in candidate_ids])
    prob = counts[counts>0]/len(ids)
    winner = candidate_ids[int(np.argmax(counts))]
    return {'majority_id':winner,'agreement':float(counts.max()/len(ids)),
            'entropy':float(-np.sum(prob*np.log(prob))/np.log(len(candidate_ids))),
            'flip':float(len(set(ids))>1)}
