"""Bridge between Paperclip heartbeats and a Paperclip-unaware research CLI.

Paperclip runs ``python -m paperclip_bridge`` through its ``process`` adapter.
The bridge claims one ticket, writes a ``ResearchRequest`` JSON file, runs the
research CLI as ``<cmd> --in request.json --out result.json`` under a
machine-wide GPU lock, reads the ``ResearchResult`` JSON back, and reports it
to Paperclip (issue document, comment, status, optional hand-off ticket).

The research CLI only ever sees the two files in ``contract.py``. It never
imports this package and never talks to Paperclip.
"""
