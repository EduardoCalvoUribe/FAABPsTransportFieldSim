# Test Suite

Comprehensive unit tests for the FAABP Payload Simulation codebase.

## Test Structure

- **[physics_utils_tests.py](physics_utils_tests.py)** - Tests for physics utility functions
  - Vector normalization
  - Line segment intersection
  - Point-to-segment distance calculations
  - Periodic boundary distance computations
  - Wall intersection detection

- **[forces_tests.py](forces_tests.py)** - Tests for force computation
  - Particle-particle repulsive forces
  - Wall collision forces
  - Cell list creation for neighbor search

- **[simulation_tests.py](simulation_tests.py)** - Tests for simulation logic
  - Curvity computation from polarity
  - Line of sight calculations
  - Score-weighted Vicsek alignment
  - Polarity computations (toward minimum score particles)
  - Goal-directed polarity
  - Orientation vector updates
  - Complete force computation

- **[main_tests.py](main_tests.py)** - Integration tests
  - Complete single simulation steps
  - Tests with walls
  - Polarity update intervals
  - Periodic boundary behavior

## Running Tests

### Run all tests:
```bash
python run_tests.py
```

### Run all tests with verbose output:
```bash
python run_tests.py -v
```

### Run specific test file:
```bash
pytest tests/physics_utils_tests.py -v
pytest tests/forces_tests.py -v
pytest tests/simulation_tests.py -v
pytest tests/main_tests.py -v
```

### Run specific test:
```bash
python run_tests.py -k test_normalize
python run_tests.py -k test_compute_repulsive_force
```

### Run with coverage:
```bash
pytest tests/ --cov=src --cov-report=html
```

## Test Coverage

The test suite covers:
- ✅ Core physics utilities (normalization, distances, intersections)
- ✅ Force calculations (repulsive, wall forces, cell lists)
- ✅ Polarity and curvity computations
- ✅ Goal-directed behavior
- ✅ Orientation dynamics
- ✅ Complete simulation step integration
- ✅ Periodic boundary conditions
- ✅ Wall interactions
