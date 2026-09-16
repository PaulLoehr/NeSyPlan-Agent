"""Unified, comparable session transcript for every orchestration mode.

Extends the frozen agent/session_log.py writer with a richer, mode-agnostic schema:
on top of the verbatim `messages` list (system/user/assistant-incl-reasoning/tool)
it records the `config` block, a structured `turns` list, and a final `metrics`
block, so every episode -- one-shot or agentic, any cache -- is one JSON file with
the SAME top-level shape and can be compared directly.

    <base_dir>/<mode>/<session_id>.json      (mode = the policy name)

Rewritten after every turn (atomic .tmp + os.replace); all I/O is best-effort, so a
crash leaves a readable partial transcript and logging never breaks a robot run.
Stdlib only.
"""

import json
import os
from datetime import datetime

SCHEMA_VERSION = 1


class SessionLog:
    def __init__(self, base_dir, mode, *, task=None, model=None, base_url=None,
                 config=None, seed=None):
        self.enabled = True
        self.path = None
        self._conversation = None   # the orchestrator's live messages list (for model_view)
        now = datetime.now()
        self.session_id = now.strftime('%Y-%m-%d_%H-%M-%S-') + f'{now.microsecond // 1000:03d}'
        self.data = {
            'schema_version': SCHEMA_VERSION,
            'mode': mode,
            'session_id': self.session_id,
            'config': config or {},
            'started_at': now.isoformat(timespec='seconds'),
            'task': task,
            'seed': seed,
            'model': model,
            'base_url': base_url,
            'messages': [],   # full verbatim conversation (assistant kept raw -> reasoning survives);
                              # includes the cache-summarizer side-calls inline (chronological)
            'model_view': None,  # what the MODEL actually sees: the re-sent conversation after
                              # content-backfill -- no summarizer side-calls, no reasoning traces
            'turns': [],      # structured per-turn records (comparable across modes)
            'metrics': {},    # final per-episode metrics
        }
        try:
            directory = os.path.join(base_dir, mode)
            os.makedirs(directory, exist_ok=True)
            path = os.path.join(directory, self.session_id + '.json')
            counter = 1
            while os.path.exists(path):
                path = os.path.join(directory, f'{self.session_id}_{counter}.json')
                counter += 1
            self.path = path
        except OSError as exc:
            self._disable(exc)
        self._save()

    def add(self, message):
        """Append one conversation message verbatim and persist. Returns the message."""
        self.data['messages'].append(message)
        self._save()
        return message

    def add_turn(self, record):
        """Append one structured turn record (turn#, reasoning, trigger, tool_call, ...)."""
        self.data['turns'].append(record)
        self._save()
        return record

    def set_meta(self, **fields):
        self.data.update(fields)
        self._save()

    def set_metrics(self, metrics):
        self.data['metrics'] = metrics
        self._save()

    def track_conversation(self, messages):
        """Register the orchestrator's live messages list so finish() can snapshot the
        model-facing conversation (post content-backfill) into `model_view`."""
        self._conversation = messages

    def finish(self, **fields):
        self.data.update(fields)
        if self._conversation is not None:   # snapshot what the model actually saw
            self.data['model_view'] = [dict(m) for m in self._conversation]
        self.data['finished_at'] = datetime.now().isoformat(timespec='seconds')
        self._save()

    def _disable(self, exc):
        if self.enabled:
            print(f'[session_log] disabled -- could not write session log: {exc!r}')
        self.enabled = False

    def _save(self):
        if not self.enabled or not self.path:
            return
        try:
            tmp = self.path + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as fh:
                json.dump(self.data, fh, indent=2, ensure_ascii=False, default=str)
            os.replace(tmp, self.path)
        except OSError as exc:
            self._disable(exc)
