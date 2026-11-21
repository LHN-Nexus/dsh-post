"""Delivery recovery against real temporary files; no network or private config.

Run: python tests/test_recovery.py
The SSH seam executes the actual remote shell commands locally. It checks the
transfer protocol, not authentication, network behaviour or a real device.
"""
from __future__ import annotations

import contextlib
import io
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path, PurePosixPath
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bridge import post
from bridge.config import Host, HostError, load_config

TOML = '''
[general]
root = "."
me = "a"
transport = "local"
[device.a]
name = "A"
workspace = "{root}/a"
inbox = "in"
outbox = "out"
[device.b]
name = "B"
workspace = "{root}/b"
inbox = "in"
outbox = "out"
aliases = ["local-shell"]
[device.c]
name = "C"
workspace = "{root}/c"
inbox = "in"
outbox = "out"
[hub]
path = "hub.jsonl"
host = ""
[post]
exts = [".md"]
verify = true
[recipients]
"BC" = ["b", "c"]
[naming]
act_kinds = ["信", "知会"]
'''


class DeliveryRecovery(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="post-recovery-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        config = self.root / "bridge.toml"
        config.write_text(TOML.format(root=self.root.as_posix()), encoding="utf-8")
        self.cfg = load_config(config)
        self.source = Path(self.cfg.device().outbox) / "信_给BC_recovery_20261121.md"
        self.source.parent.mkdir(parents=True)
        self.source.write_bytes("原始正文\n".encode())

    def fail_c_once(self):
        real_write = Host.write_bytes
        calls = []
        failed = False

        def write(host, path, data):
            nonlocal failed
            if Path(path).name == self.source.name and host.device.key in ("b", "c"):
                calls.append((host.device.key, data))
                if host.device.key == "c" and not failed:
                    failed = True
                    raise HostError("injected first-attempt disconnect")
            return real_write(host, path, data)
        return calls, write

    def test_partial_success_survives_next_round_and_consumed_inbox(self):
        calls, write = self.fail_c_once()
        with patch.object(Host, "write_bytes", write):
            self.assertEqual(post.post_once(self.cfg)["failed"], 1)
            Path(self.cfg.devices["b"].inbox, self.source.name).unlink()
            second = post.post_once(self.cfg)
        self.assertEqual([t for t, _ in calls], ["b", "c", "c"])
        self.assertEqual(second["delivered"], 1)
        self.assertFalse(Path(self.cfg.devices["b"].inbox, self.source.name).exists())
        self.assertEqual(Path(self.cfg.devices["c"].inbox, self.source.name).read_bytes(),
                         self.source.read_bytes())
        self.assertEqual(post.load_posted(self.cfg)[self.source.name], post.md5_file(self.source))

    def test_manual_send_uses_same_recovery_receipts(self):
        calls, write = self.fail_c_once()
        with patch.object(Host, "write_bytes", write), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(post.send_one(self.cfg, self.source, None, False), 1)
            Path(self.cfg.devices["b"].inbox, self.source.name).unlink()
            self.assertEqual(post.send_one(self.cfg, self.source, None, False), 0)
        self.assertEqual([t for t, _ in calls], ["b", "c", "c"])

    def test_one_round_freezes_source_and_does_not_archive_changed_bytes(self):
        real_write = Host.write_bytes
        original = self.source.read_bytes()
        changed = "投递中修改后的正文\n".encode()
        seen = []

        def write(host, path, data):
            result = real_write(host, path, data)
            if Path(path).name == self.source.name and host.device.key in ("b", "c"):
                seen.append(data)
                if host.device.key == "b":
                    self.source.write_bytes(changed)
            return result

        with patch.object(Host, "write_bytes", write):
            first = post.post_once(self.cfg)
        self.assertEqual(seen, [original, original])
        self.assertGreater(first["failed"], 0)
        self.assertNotIn(self.source.name, post.load_posted(self.cfg))
        self.assertEqual(post.post_once(self.cfg)["delivered"], 1)
        for target in ("b", "c"):
            self.assertEqual(Path(self.cfg.devices[target].inbox, self.source.name).read_bytes(), changed)
        self.assertEqual(Path(self.cfg.device().sent_dir, self.source.name).read_bytes(), changed)

    def test_dry_run_does_not_create_recovery_journal(self):
        with contextlib.redirect_stdout(io.StringIO()):
            post.post_once(self.cfg, dry=True)
        self.assertFalse((self.source.parent / ".post-progress.json").exists())
        self.assertFalse(post.posted_path(self.cfg).exists())

    def test_requested_digest_verification_cannot_silently_succeed(self):
        with patch.object(Host, "digest", return_value=""):
            result = post.post_once(self.cfg)
        self.assertEqual(result["delivered"], 0)
        self.assertGreater(result["failed"], 0)
        self.assertNotIn(self.source.name, post.load_posted(self.cfg))

    def test_corrupt_journal_returns_failure_without_writes(self):
        damaged = b'{"broken":'
        post.progress_path(self.cfg).write_bytes(damaged)
        with patch.object(Host, "write_bytes", side_effect=AssertionError("must not write")):
            result = post.post_once(self.cfg)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(post.main(["--once", "--config", str(self.cfg.path)]), 1)
        self.assertEqual(result["candidates"], 0)
        self.assertEqual(result["failed"], 1)
        self.assertTrue(result["errors"])
        self.assertEqual(post.progress_path(self.cfg).read_bytes(), damaged)
        self.assertFalse(post.posted_path(self.cfg).exists())

    def test_failed_success_receipt_stops_remaining_targets(self):
        real_write = Host.write_bytes
        calls = []

        def write(host, path, data):
            if Path(path).name == self.source.name:
                calls.append(host.device.key)
            return real_write(host, path, data)

        with patch.object(Host, "write_bytes", write), patch.object(post, "save_progress", return_value=False):
            result = post.post_once(self.cfg)
        self.assertEqual(calls, ["b"])
        self.assertEqual(result["delivered"], 0)
        self.assertEqual(result["failed"], 1)
        self.assertFalse(post.posted_path(self.cfg).exists())

    def test_changed_endpoint_does_not_reuse_previous_success(self):
        calls, write = self.fail_c_once()
        with patch.object(Host, "write_bytes", write):
            post.post_once(self.cfg)
            self.cfg.devices["b"].inbox = self.root / "new-inbox-b"
            second = post.post_once(self.cfg)
        self.assertEqual([t for t, _ in calls], ["b", "c", "b", "c"])
        self.assertEqual(second["delivered"], 1)
        self.assertEqual(Path(self.cfg.devices["b"].inbox, self.source.name).read_bytes(),
                         self.source.read_bytes())


class RemoteAtomicWrite(unittest.TestCase):
    def setUp(self):
        self.shell = shutil.which("sh")
        if not self.shell:
            git_shell = Path("C:/Program Files/Git/bin/bash.exe")
            if git_shell.is_file():
                self.shell = str(git_shell)
        if not self.shell:
            self.skipTest("POSIX shell unavailable; remote transport seam untested")
        self.tmp = tempfile.TemporaryDirectory(prefix="post-atomic-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        config = self.root / "bridge.toml"
        config.write_text(TOML.format(root=self.root.as_posix()).replace('transport = "local"',
                                                                       'transport = "ssh"'),
                          encoding="utf-8")
        self.host = load_config(config).host("b")
        self.target = self.root / "remote ' quoted.md"
        self.remote = PurePosixPath(self.target.as_posix())
        self.data = ("中文 payload\n" * 4096).encode()

    def shell_ssh(self, mode="normal"):
        def run(_host, alias, cmd, stdin=None, timeout=None):
            data = stdin
            if data and mode == "truncate":
                data = data[:len(data) // 2]
            if data and mode == "corrupt":
                data = bytes([data[0] ^ 1]) + data[1:]
            result = subprocess.run([self.shell, "-c", cmd], input=data or b"",
                                    capture_output=True, timeout=15)
            return result.returncode, result.stdout, result.stderr
        return run

    def test_interrupted_transfer_preserves_existing_file(self):
        self.target.write_bytes(b"old complete body")
        unrelated = self.root / ".other.tmp-dshbridge"
        unrelated.write_bytes(b"another delivery")
        with patch.object(Host, "_ssh", self.shell_ssh("truncate")):
            with self.assertRaises(HostError):
                self.host.write_bytes(self.remote, self.data)
        self.assertEqual(self.target.read_bytes(), b"old complete body")
        self.assertEqual(unrelated.read_bytes(), b"another delivery")
        self.assertEqual(set(self.root.glob("*tmp-dshbridge")), {unrelated})

    def test_interrupted_transfer_does_not_publish_new_file(self):
        with patch.object(Host, "_ssh", self.shell_ssh("truncate")):
            with self.assertRaises(HostError):
                self.host.write_bytes(self.remote, self.data)
        self.assertFalse(self.target.exists())

    def test_same_length_corruption_does_not_replace_old_file(self):
        self.target.write_bytes(b"old body")
        with patch.object(Host, "_ssh", self.shell_ssh("corrupt")):
            with self.assertRaises(HostError):
                self.host.write_bytes(self.remote, self.data)
        self.assertEqual(self.target.read_bytes(), b"old body")

    def test_successful_retry_publishes_exact_bytes_and_cleans_own_temp(self):
        self.target.write_bytes(b"old body")
        with patch.object(Host, "_ssh", self.shell_ssh()):
            self.host.write_bytes(self.remote, self.data)
        self.assertEqual(self.target.read_bytes(), self.data)
        self.assertEqual(list(self.root.glob("*tmp-dshbridge")), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
