#!/usr/bin/env python
"""
Run all tests in the tests/ directory.

Usage:
    python run_tests.py           # Run all tests
    python run_tests.py -v        # Run with verbose output
    python run_tests.py -k test_normalize  # Run specific test
"""
import sys
import pytest


if __name__ == "__main__":
    # Default arguments
    args = ["tests/", "-v"]

    # Add any command-line arguments passed to this script
    if len(sys.argv) > 1:
        args = ["tests/"] + sys.argv[1:]

    # Run pytest
    exit_code = pytest.main(args)
    sys.exit(exit_code)
