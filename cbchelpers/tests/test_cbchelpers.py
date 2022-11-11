"""
Unit and regression test for the cbchelpers package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import cbchelpers


def test_cbchelpers_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "cbchelpers" in sys.modules
