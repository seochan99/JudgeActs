import pytest
from src.run_smol import parse_selection

def test_exact_choice():
    data,flags=parse_selection('{"choice":"B"}',list('ABC'))
    assert data['choice']=='B' and flags['schema_valid'] and not flags['recovered']

def test_schema_normalization_preserves_raw_status():
    data,flags=parse_selection(' A ',list('ABC'))
    assert data['choice']=='A' and not flags['json_valid'] and flags['recovered']
    data,flags=parse_selection('{"image":"Candidate C"}',list('ABC'))
    assert data['choice']=='C' and flags['json_valid'] and not flags['schema_valid'] and flags['recovered']
    data,flags=parse_selection("{'choice':'A'}",list('ABC'))
    assert data['choice']=='A' and not flags['json_valid'] and flags['recovered']

def test_no_guessing_or_multiple_choices():
    for raw in ['{"choice":"A or B"}','{"choice":"D"}','{"choice":"A","ranking":["B"]}',
                'Candidate A is best.', '{"image":"A because it is clearer"}']:
        with pytest.raises(ValueError):parse_selection(raw,list('ABC'))
