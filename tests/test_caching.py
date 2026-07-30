"""
Test Streamlit caching effectiveness

This script measures:
1. Cold start (no cache)
2. Warm start (with cache)
3. Speedup factor
"""

import time
import sys
import os
import pytest

# Suppress Streamlit warnings during testing
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'

if not os.path.exists('data_files/combined_historical_data_with_calculations_new.csv'):
    pytest.skip('EPL generated fixtures belong in the consuming league repository', allow_module_level=True)

print('='*60)
print('STREAMLIT CACHING PERFORMANCE TEST')
print('='*60)

# Test 1: Simulate cold start (clear cache)
print('\n🥶 Test 1: Cold Start (First Load - No Cache)')
print('   Simulating fresh app load...')

# Import and time the imports
start_import = time.time()
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from os import path
import_time = time.time() - start_import

print(f'   Import time: {import_time:.3f}s')

# Import the cached functions directly
sys.path.insert(0, '.')
import importlib.util
spec = importlib.util.spec_from_file_location("app_shell", "app_shell.py")
prem = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prem)

load_pretrained_models = prem.load_pretrained_models
load_and_process_data = prem.load_and_process_data

# Clear Streamlit cache
st.cache_data.clear()
st.cache_resource.clear()
print('   Cache cleared')

# Time data loading
csv_path = 'data_files/combined_historical_data_with_calculations_new.csv'
start_data = time.time()
try:
    # This simulates what happens in the app
    X_train, X_test, y_train, y_test, feature_names, df = load_and_process_data(csv_path)
    data_time = time.time() - start_data
    print(f'   Data loading: {data_time:.3f}s')
    print(f'   Training samples: {len(X_train):,}')
except Exception as e:
    print(f'   ❌ Data loading failed: {e}')
    data_time = 0

# Time model loading
start_models = time.time()
try:
    models = load_pretrained_models()
    models_time = time.time() - start_models
    print(f'   Model loading: {models_time:.3f}s')
    if models:
        print(f'   Models loaded: {len([k for k, v in models.items() if v is not None and k != "performance"])}')
except Exception as e:
    print(f'   ❌ Model loading failed: {e}')
    models_time = 0

cold_start_total = data_time + models_time
print(f'   \n   ❄️  COLD START TOTAL: {cold_start_total:.3f}s')

# Test 2: Simulate warm start (cache hit)
print('\n🔥 Test 2: Warm Start (Cached Load)')
print('   Simulating app reload with cache...')

# Time data loading (should be cached)
start_data_cached = time.time()
try:
    X_train, X_test, y_train, y_test, feature_names, df = load_and_process_data(csv_path)
    data_time_cached = time.time() - start_data_cached
    print(f'   Data loading (cached): {data_time_cached:.3f}s')
except Exception as e:
    print(f'   ❌ Cached data loading failed: {e}')
    data_time_cached = 0

# Time model loading (should be cached)
start_models_cached = time.time()
try:
    models = load_pretrained_models()
    models_time_cached = time.time() - start_models_cached
    print(f'   Model loading (cached): {models_time_cached:.3f}s')
except Exception as e:
    print(f'   ❌ Cached model loading failed: {e}')
    models_time_cached = 0

warm_start_total = data_time_cached + models_time_cached
print(f'   \n   🚀 WARM START TOTAL: {warm_start_total:.3f}s')

# Calculate speedup
print('\n' + '='*60)
print('📊 PERFORMANCE ANALYSIS')
print('='*60)

if cold_start_total > 0 and warm_start_total > 0:
    data_speedup = data_time / data_time_cached if data_time_cached > 0 else 0
    models_speedup = models_time / models_time_cached if models_time_cached > 0 else 0
    total_speedup = cold_start_total / warm_start_total if warm_start_total > 0 else 0
    
    print(f'\n🔢 Component Speedups:')
    print(f'   Data loading: {data_speedup:.1f}x faster')
    print(f'   Model loading: {models_speedup:.1f}x faster')
    
    print(f'\n⚡ TOTAL SPEEDUP: {total_speedup:.1f}x faster')
    print(f'   Cold start: {cold_start_total:.3f}s')
    print(f'   Warm start: {warm_start_total:.3f}s')
    print(f'   Time saved: {cold_start_total - warm_start_total:.3f}s per reload')
    
    # Estimate savings over time
    daily_visits = 10  # Conservative estimate
    monthly_visits = daily_visits * 30
    time_saved_monthly = (cold_start_total - warm_start_total) * monthly_visits / 60  # minutes
    
    print(f'\n💰 Estimated Monthly Savings:')
    print(f'   Assuming {daily_visits} unique visitors/day:')
    print(f'   Time saved per month: {time_saved_monthly:.1f} minutes')
    print(f'   Better user experience: {monthly_visits} faster loads')
    
    if total_speedup > 5:
        print('\n✅ EXCELLENT: Cache provides >5x speedup!')
    elif total_speedup > 2:
        print('\n✅ GOOD: Cache provides >2x speedup')
    else:
        print('\n⚠️  Cache speedup is minimal - investigate further')
else:
    print('❌ Unable to calculate speedup - tests failed')

print('\n' + '='*60)
