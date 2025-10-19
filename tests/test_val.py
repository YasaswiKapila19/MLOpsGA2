import pytest
from validate import validate_data

def test_validate_data_passes():
    result = validate_data("data/test.csv")
    assert result is True, "Size < 10, Data validation should pass for clean data"

