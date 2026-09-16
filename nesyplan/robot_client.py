"""Thin HTTP client for an EXTERNAL robot command server.

Used only by --backend sim. The executor it talks to is not part of this repository (see
docs/ARCHITECTURE.md); this client exists so the harness can drive one if you have it.
The in-process default (nesyplan/fake_robot.py) implements the same method surface, which
is what makes the two interchangeable. Stdlib urllib only -- no third-party deps.

The HTTP contract is:
  GET  /state             -> world observation {ok, grid_units, held, cubes, mode?}
  GET  /health            -> {ok, status}
  POST /command {action..} -> feedback {ok, message|error, observation}
"""

import json
import urllib.error
import urllib.request

DEFAULT_URL = 'http://localhost:8100'


class RobotClient:
    def __init__(self, base_url=DEFAULT_URL, timeout=120):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout

    def _request(self, method, path, body=None):
        url = self.base_url + path
        data = json.dumps(body).encode('utf-8') if body is not None else None
        headers = {'Content-Type': 'application/json'} if data is not None else {}
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode('utf-8', 'replace')[:500]
            try:
                return json.loads(detail)
            except ValueError:
                return {'ok': False, 'error': f'HTTP {exc.code} {exc.reason}: {detail}'}
        except urllib.error.URLError as exc:
            raise ConnectionError(
                f'cannot reach the command server at {self.base_url} ({exc.reason}). '
                f'That executor is external to this repository (see docs/ARCHITECTURE.md); '
                f'to run without one, use --backend fake') from exc

    def health(self):
        return self._request('GET', '/health')

    def get_state(self):
        """Current world observation (initial state before any command)."""
        return self._request('GET', '/state')

    def send_command(self, action, **args):
        """Send one command; returns the feedback dict {ok, message|error, observation}."""
        body = {'action': action}
        body.update(args)
        return self._request('POST', '/command', body)
