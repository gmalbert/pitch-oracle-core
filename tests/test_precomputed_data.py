"""
Test script to verify precomputed data integrity
"""
import pickle
import numpy as np
import os
import pytest

if not os.path.exists('precomputed/preprocessed_data.pkl'):
    pytest.skip('EPL generated fixtures belong in the consuming league repository', allow_module_level=True)

print('Loading precomputed data...')
with open('precomputed/preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)

print('\n' + '='*60)
print('PRECOMPUTED DATA VERIFICATION')
print('='*60)

# Check what keys are in the data
print('\n📦 Contents:')
for key in data.keys():
    print(f'   • {key}')

# Check array shapes and types
print('\n📊 Array Details:')
print(f'   X_train shape: {data["X_train"].shape}')
print(f'   X_train dtype: {data["X_train"].dtype}')
print(f'   X_test shape: {data["X_test"].shape}')
print(f'   X_test dtype: {data["X_test"].dtype}')
print(f'   y_train shape: {data["y_train"].shape}')
print(f'   y_train dtype: {data["y_train"].dtype}')
print(f'   y_test shape: {data["y_test"].shape}')
print(f'   y_test dtype: {data["y_test"].dtype}')

# Check feature names
print(f'\n🏷️  Feature Names:')
print(f'   Count: {len(data["feature_names"])}')
print(f'   First 5: {data["feature_names"][:5]}')
print(f'   Last 5: {data["feature_names"][-5:]}')

# Check metadata
if 'metadata' in data:
    print(f'\n📋 Metadata:')
    for key, value in data['metadata'].items():
        print(f'   {key}: {value}')

# Check sample data
if 'df_sample' in data:
    df_sample = data['df_sample']
    print(f'\n🔍 DataFrame Sample:')
    print(f'   Rows: {len(df_sample)}')
    print(f'   Columns: {len(df_sample.columns)}')

# Verify data integrity
print(f'\n✅ Data Integrity Checks:')
print(f'   X_train has NaN: {np.isnan(data["X_train"]).any()}')
print(f'   X_test has NaN: {np.isnan(data["X_test"]).any()}')
print(f'   X_train has Inf: {np.isinf(data["X_train"]).any()}')
print(f'   X_test has Inf: {np.isinf(data["X_test"]).any()}')
print(f'   y_train unique values: {np.unique(data["y_train"])}')
print(f'   y_test unique values: {np.unique(data["y_test"])}')

# Check class distribution
print(f'\n📈 Target Distribution:')
train_unique, train_counts = np.unique(data['y_train'], return_counts=True)
test_unique, test_counts = np.unique(data['y_test'], return_counts=True)
print(f'   Training set:')
for val, count in zip(train_unique, train_counts):
    pct = (count / len(data['y_train'])) * 100
    label = ['Home Win', 'Draw', 'Away Win'][int(val)]
    print(f'     {label} ({val}): {count} ({pct:.1f}%)')
print(f'   Test set:')
for val, count in zip(test_unique, test_counts):
    pct = (count / len(data['y_test'])) * 100
    label = ['Home Win', 'Draw', 'Away Win'][int(val)]
    print(f'     {label} ({val}): {count} ({pct:.1f}%)')

# Sample some actual data values
print(f'\n🔢 Sample Data Values (X_train[0, :10]):')
print(f'   {data["X_train"][0, :10]}')

# Check memory size
import sys
total_size = sys.getsizeof(pickle.dumps(data)) / (1024 * 1024)
print(f'\n💾 Memory Size: {total_size:.2f} MB')

print('\n' + '='*60)
print('✅ VERIFICATION COMPLETE')
print('='*60)
