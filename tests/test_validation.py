import pandas as pd
import pytest
from src.validation import aligned, sha256


def test_alignment_order_independent():
    a=pd.DataFrame({'id':[1,2],'time':[10,20]})
    aligned(a,a.iloc[::-1],'id',['time'])

@pytest.mark.parametrize('ids,times', [([1,1],[10,20]),([1,3],[10,20]),([1,2],[10,21])])
def test_alignment_rejects_mismatch(ids,times):
    a=pd.DataFrame({'id':[1,2],'time':[10,20]})
    with pytest.raises((ValueError,AssertionError)):
        aligned(a,pd.DataFrame({'id':ids,'time':times}),'id',['time'])


def test_hash(tmp_path):
    p=tmp_path/'file'; p.write_bytes(b'abc')
    assert sha256(p)=='ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'
