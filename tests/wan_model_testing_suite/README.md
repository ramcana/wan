# WAN Model Testing Suite

Comprehensive testing framework for WAN model implementations (T2V, I2V, TI2V) with unit tests, integration tests, performance benchmarking, and hardware compatibility validation.

## Test Structure

```
tests/wan_model_testing_suite/
├── unit/                    # Unit tests for individual model components
│   ├── test_wan_t2v_a14b.py
│   ├── test_wan_i2v_a14b.py
│   ├── test_wan_ti2v_5b.py
│   └── test_model_components.py
├── integration/             # Integration tests with infrastructure
│   ├── test_model_pipeline_integration.py
│   ├── test_model_service_integration.py
│   └── test_model_api_integration.py
├── performance/             # Performance benchmarking tests
│   ├── test_generation_benchmarks.py
│   ├── test_memory_benchmarks.py
│   └── test_throughput_benchmarks.py
├── hardware/                # Hardware compatibility tests
│   ├── test_rtx4080_optimization.py
│   ├── test_threadripper_optimization.py
│   └── test_hardware_detection.py
├── fixtures/                # Test fixtures and mock data
│   ├── sample_prompts.json
│   ├── sample_images/
│   └── mock_models.py
└── utils/                   # Testing utilities
    ├── test_helpers.py
    ├── benchmark_utils.py
    └── hardware_utils.py
```

## Requirements Addressed

- **4.1**: Unit testing for model implementations
- **5.3**: Integration testing with existing infrastructure
- **6.1**: Performance benchmarking for generation quality
- **8.1**: Hardware compatibility testing for RTX 4080 and Threadripper PRO

## Running Tests

```bash
# Run all WAN model tests
pytest tests/wan_model_testing_suite/ -v

# Run specific test categories
pytest tests/wan_model_testing_suite/unit/ -v
pytest tests/wan_model_testing_suite/integration/ -v
pytest tests/wan_model_testing_suite/performance/ -v
pytest tests/wan_model_testing_suite/hardware/ -v

# Run with coverage
pytest tests/wan_model_testing_suite/ --cov=core.models.wan_models --cov-report=html
```

## Test Configuration

Tests can be configured via environment variables:

- `WAN_TEST_HARDWARE`: Enable hardware-specific tests
- `WAN_TEST_PERFORMANCE`: Enable performance benchmarks
- `WAN_TEST_INTEGRATION`: Enable integration tests
- `WAN_BENCHMARK_ITERATIONS`: Number of benchmark iterations (default: 10)
