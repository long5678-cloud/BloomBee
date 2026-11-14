"""Structured cache receipts must survive a caller's ordinary log verbosity."""
import json
import logging
from types import SimpleNamespace
from bloombee.server.dynamo_service import _emit
from bloombee.kv_cache.telemetry_paged import _prefill_rows, _layer_skip


def test_cache_receipts_reach_strict_layer_audit_with_warning_logging(capsys):
    root=logging.getLogger();old=root.level;root.setLevel(logging.WARNING)
    try:
        uids=['model.'+str(i)for i in range(32)]
        for uid in uids:
            _emit(SimpleNamespace(name=uid),mode='fallback',telemetry_attempt_id='actual_attempt',
                requested_tokens=292,computed_tokens=292,skipped_tokens=0,returned_tokens=292,
                reason='insufficient bounded page capacity',native_layers_imported=0)
        rows=[]
        for line in capsys.readouterr().out.splitlines():
            _,marker,body=line.partition('DYNAMO_SERVICE ')
            if marker:rows.append(json.loads(body))
        audited=_prefill_rows(rows,uids,'actual_attempt')
        assert len(audited)==32
        assert all(_layer_skip(row,292)==0 for row in audited)
    finally:root.setLevel(old)


def test_one_record_and_newline_stay_together_with_interleaved_warning():
    import contextlib,io
    class Interleaved:
        def __init__(self):self.stream=io.StringIO();self.injected=False
        def write(self,value):
            result=self.stream.write(value)
            if not self.injected and value.startswith('DYNAMO_SERVICE '):
                self.stream.write('ordinary concurrent warning\n');self.injected=True
            return result
        def flush(self):pass
    output=Interleaved()
    with contextlib.redirect_stdout(output):
        _emit(SimpleNamespace(name='model.0'),mode='fallback',telemetry_attempt_id='actual_attempt',
            requested_tokens=292,computed_tokens=292,skipped_tokens=0,returned_tokens=292)
    rows=[json.loads(line.partition('DYNAMO_SERVICE ')[2])for line in output.stream.getvalue().splitlines()if 'DYNAMO_SERVICE 'in line]
    assert len(_prefill_rows(rows,['model.0'],'actual_attempt'))==1
    assert _layer_skip(rows[0],292)==0
