Implementation of the time-dependent dual TRILEX theory
=======================================================

Verify proper functioning of the package by running

```bash
    pip install pytest pytest-mpi
    pytest -v             # Skip the unit tests depending on MPI
    pytest -v --with-mpi  # Run the MPI tests as well
```

in its root directory.
