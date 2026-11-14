"""Local ordering labels stay stable and do not fabricate Linux boot identity."""
from pathlib import Path
from unittest.mock import patch
import pytest
from bloombee.server import page_control as control


def reset(monkeypatch):
    monkeypatch.setattr(control,'_clock_pid',None)
    monkeypatch.setattr(control,'_clock_domain',None)
    monkeypatch.delenv('BLOOMBEE_CLOCK_DOMAIN',raising=False)


def test_linux_boot_value_wins_and_stays_stable(monkeypatch):
    reset(monkeypatch);monkeypatch.setenv('BLOOMBEE_CLOCK_DOMAIN','explicit-other')
    with patch.object(Path,'read_text',return_value='real-kernel-domain\n'):
        assert control.local_clock_domain()=='real-kernel-domain'
        assert control.local_clock_domain()=='real-kernel-domain'


def test_nonlinux_fallback_is_per_process(monkeypatch):
    reset(monkeypatch)
    with patch.object(Path,'read_text',side_effect=FileNotFoundError):
        with patch.object(control.os,'getpid',return_value=101):
            first=control.local_clock_domain();assert first==control.local_clock_domain()
        with patch.object(control.os,'getpid',return_value=102):
            second=control.local_clock_domain()
    assert first!=second and first.startswith('process_101_') and second.startswith('process_102_')


def test_explicit_nonlinux_domain_and_blank_rejection(monkeypatch):
    reset(monkeypatch);monkeypatch.setenv('BLOOMBEE_CLOCK_DOMAIN',' supplied-domain ')
    with patch.object(Path,'read_text',side_effect=FileNotFoundError):
        assert control.local_clock_domain()=='supplied-domain'
    monkeypatch.setattr(control,'_clock_pid',None);monkeypatch.setenv('BLOOMBEE_CLOCK_DOMAIN','  ')
    with patch.object(Path,'read_text',side_effect=FileNotFoundError),pytest.raises(ValueError):
        control.local_clock_domain()


def test_registry_status_share_target_domain(monkeypatch):
    reset(monkeypatch);monkeypatch.setattr(control,'_bundle_registry',None);monkeypatch.setattr(control,'_states',{})
    assert control.bundle_registry().clock_domain==control.status()['clock_domain']==control.local_clock_domain()
