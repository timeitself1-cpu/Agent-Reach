"""Reach chat: a local web page that shows the latest trends report and a grounded chat agent.

    python -m agent_reach chat                 # opens http://127.0.0.1:8765
    python -m agent_reach chat --no-cloud      # local runs only
"""

from agent_reach.chat.agent import ReachAgent
from agent_reach.chat.snapshot import Snapshot, load_latest

__all__ = ["ReachAgent", "Snapshot", "load_latest"]
