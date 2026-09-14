"""Domain services.

``cuda``      - device state, VRAM reporting, host-OOM recognition
``paint``     - the Hunyuan3D-2 paint model (heavy, GPU, singleton)
``mesh``      - mesh preparation (clean, Taubin, decimate, UV unwrap)
``pipeline``  - composes them into one (mesh, image) -> textured mesh job

Nothing here knows about HTTP: they raise the exceptions in
``app.exceptions`` and the API layer turns those into responses.
"""
