#!/usr/bin/env python
"""Quick test of load_dataset with built-in data."""

from backend.agent.tools.data.load_tools import load_dataset

# Test built-in data loading
result = load_dataset(use_builtin_data=True)

print(f"✅ Status: {result['status']}")
print(f"✅ Rows loaded: {result['rows']:,}")
print(f"✅ Date column: {result['date_column']}")
print(f"✅ Sales column: {result['sales_column']}")
print(f"✅ Data type: {result.get('data_type', 'N/A')}")
