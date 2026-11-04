"""Memory tests: can a fixed-size or compressed memory stand in for attention's
lookup? Track A tests the associative matrix as a standalone store (store.py);
Track B tests memories inside a Hyena model on recall tasks (tasks.py,
layers.py, model.py, train.py). Protocol: memtest/PROTOCOL.md."""
