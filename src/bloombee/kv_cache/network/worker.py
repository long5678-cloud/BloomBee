"""Retain v057 a2 worker protocol/pins; adapt only the page-bundle backend."""
from pathlib import Path
import importlib.util
import json
import os
import threading
import hashlib
import sys

P = Path(__file__).resolve().parent


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--inbox',required=True)
    parser.add_argument('--plugin',required=True)
    parser.add_argument('--shadow-common',required=True)
    parser.add_argument('--common-code',required=True)
    parser.add_argument('--native-code',required=True)
    parser.add_argument('--framework-root',required=True)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    from bloombee.kv_cache.network import worker_protocol as base
    from bloombee.kv_cache.network.joint_runtime import create_joint_runtime
    from bloombee.kv_cache.network.joint_strategy import joint_context

    class JointWorker(base.Worker):
        def __init__(self, parsed):
            super().__init__(parsed)
            # The original init creates only a pure runner and CPU protocol
            # readiness. Backend readiness is separately recorded below.
            self.joint = create_joint_runtime(self.config,emit=self.emit)
            self.build_context = joint_context
            self.runner = self.joint.runner
            original_close = self.runner.close
            close_lock = threading.Lock()
            closed = False
            def close_once():
                nonlocal closed
                with close_lock:
                    if closed:
                        return
                    closed = True
                    if self.joint.http_thread is not None:
                        self.joint.server.shutdown()
                        self.joint.http_thread.join()
                        self.joint.http_thread = None
                    self.joint.server.server_close()
                    self.joint.bank.close()
                    original_close()
                    base.write(self.inbox/'joint_closed.json',dict(
                        bank=self.joint.bank.accounting(),
                        native_staging=self.joint.bridge.retirement_status(),
                        worker_pid=os.getpid()))
            # The original worker closes its runner twice defensively. Preserve
            # that lifecycle while making the added bank/native close exact-once.
            self.runner.close = close_once
            ready = self.joint.start()
            base.write(self.inbox/'joint_ready.json',dict(ready,worker_pid=os.getpid(),
                                                         config_sha256=self.config_sha))

        def run(self):
            try:
                return super().run()
            finally:
                # Base.run drains the existing single warm executor and closes
                # runner resources. Close the additional loopback server here.
                if self.joint.http_thread is not None:
                    self.joint.server.shutdown()
                    self.joint.http_thread.join()
                    self.joint.http_thread = None
                self.joint.server.server_close()

    JointWorker(args).run()


if __name__ == '__main__':
    main()
