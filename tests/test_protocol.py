import pytest
from analysis.metrics import candidate_metrics, random_metrics, choice_consistency
from src.common import orders, strict_parse
from src.prepare import aggregate

def test_regret_and_random():
    scores=[.2,.6,1.]
    assert candidate_metrics(scores,2)['regret']==0
    assert candidate_metrics(scores,0)['regret']>.0
    assert random_metrics(scores)['utility']==pytest.approx(.6)
    assert random_metrics(scores)['hsr']==pytest.approx(1/3)
    assert candidate_metrics([.5,.5,.5],0)['hsr']==0
    assert candidate_metrics([.5,.5,.5],0)['human_best']==1

def test_permutation_identity():
    group={'prompt_id':'abc','candidates':[{'id':str(i)} for i in range(4)]}
    oo=orders(group)
    assert len({tuple(o) for o in oo})==3
    for order in oo:
        mapping=dict(zip('ABCD',order))
        label=next(k for k,v in mapping.items() if v=='2')
        assert mapping[label]=='2'
        assert set(order)=={'0','1','2','3'}

def test_strict_parser():
    assert strict_parse('{"choice":"B","ranking":["B","A","C"]}',list('ABC'))['choice']=='B'
    for raw in ['{"choice":"A","ranking":["B","A","C"]}', '{"choice":"A","ranking":["A","A","C"]}', '```json\n{}\n```']:
        with pytest.raises(ValueError):strict_parse(raw,list('ABC'))

def test_consistency_identity_and_tie():
    assert choice_consistency(['c','c','c'],['a','b','c'])['agreement']==1
    assert choice_consistency(['a','b','c'],['a','b','c'])['majority_id']=='a'
    assert choice_consistency(['b','a','b'],['a','b','c'])['majority_id']=='b'

def test_human_join_aggregation():
    row={'human_annotations':[{'prompt_alignment':{'score':1,'missing_explicit':False,'missing_implicit':True},
                              'stereotype':{'is_stereotypical':False},'image_quality':{'score':.5},'overall_score':5},
                             {'prompt_alignment':{'score':0,'missing_explicit':True,'missing_implicit':False},
                              'stereotype':{'is_stereotypical':True},'image_quality':{'score':1},'overall_score':1}]}
    out=aggregate(row)
    assert out['utility']==.5 and out['overall']==.5
    assert out['missing_explicit']==.5 and out['stereotype']==.5
    assert out['n_utility']==2
