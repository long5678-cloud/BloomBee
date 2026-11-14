"""A local hit must not require a remote deployment; explicit settings stay strict."""
import os
import pytest
from bloombee.server.dynamo_remote import PREFIX,Settings,telemetry_settings


def clean(monkeypatch):
    for key in tuple(os.environ):
        if key.startswith(PREFIX):monkeypatch.delenv(key)


def test_local_without_remote_identity_is_explicitly_unknown(monkeypatch):
    clean(monkeypatch)
    assert telemetry_settings() is None
    monkeypatch.setenv(PREFIX+'MODE','off')
    assert telemetry_settings() is None

@pytest.mark.parametrize('value',['consume','invalid'])
def test_enabled_or_invalid_remote_remains_strict(monkeypatch,value):
    clean(monkeypatch);monkeypatch.setenv(PREFIX+'MODE',value)
    with pytest.raises(ValueError):telemetry_settings()


def test_partial_identity_is_not_silently_ignored(monkeypatch):
    clean(monkeypatch);monkeypatch.setenv(PREFIX+'CONTEXT','actual')
    with pytest.raises(ValueError):telemetry_settings()


def test_complete_local_identity_matches_original_settings(monkeypatch):
    clean(monkeypatch)
    for key,value in dict(MODE='off',CONTEXT='actual',MODEL_ID='model',MODEL_REVISION='b'*40,
        EXECUTION_DIGEST='a'*64,TOKENIZER_DIGEST='c'*64,
        ENDPOINT='http://127.0.0.1:1',ALLOWED_ENDPOINTS='["http://127.0.0.1:1"]').items():monkeypatch.setenv(PREFIX+key,value)
    assert telemetry_settings()==Settings.from_env(identity_only=True)
