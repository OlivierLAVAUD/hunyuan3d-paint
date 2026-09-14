"""HTTP layer: routes, probes and error handling.

``probes.py``  - ``/health`` (liveness) and ``/ready`` (readiness)
``v1/``        - the versioned ``/api/v1`` surface
``errors.py``  - exception handlers producing the shared error envelope
"""
