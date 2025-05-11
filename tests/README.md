# Tests

To execute all tests, run the following:

```bash
PYTHONPATH=. pytest tests/tests.py
```

## Golden Test
The golden test is designed to check if changes in the code have resulted in a difference in model behaviour. It takes a pre-defined input, runs the model, and compares the output to what is expected. If something changed in the model and the outputs are different, the test will fail.

The golden test can be ran with:
```bash
PYTHONPATH=. pytest tests/tests.py::test_golden_<task>
```

where `<task>` is one of `parity`, `qamnist`, `rl`.