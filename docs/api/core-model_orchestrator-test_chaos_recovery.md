---
title: core.model_orchestrator.test_chaos_recovery
category: api
tags: [api, core]
---

# core.model_orchestrator.test_chaos_recovery

Chaos tests for network failures and interruptions in the Model Orchestrator.

## Classes

### NetworkFailureSimulator

Simulates various network failure scenarios.

#### Methods

##### __init__(self: Any)



##### set_failure_rate(self: Any, rate: float)

Set the probability of failure (0.0 to 1.0).

##### set_failure_types(self: Any, failure_types: <ast.Subscript object at 0x000001942FDF0370>)

Set the types of failures to inject.

##### should_fail(self: Any) -> bool

Determine if this call should fail.

##### inject_failure(self: Any)

Inject a random failure.

### DiskFailureSimulator

Simulates disk-related failures.

#### Methods

##### __init__(self: Any)



##### set_no_space_probability(self: Any, probability: float)

Set probability of disk space errors.

##### set_permission_error_probability(self: Any, probability: float)

Set probability of permission errors.

##### set_corruption_probability(self: Any, probability: float)

Set probability of file corruption.

##### check_disk_space(self: Any, path: str, required_bytes: int)

Simulate disk space check with potential failure.

##### write_file(self: Any, path: str, content: bytes)

Simulate file write with potential failures.

### ProcessKillSimulator

Simulates process interruptions and kills.

#### Methods

##### __init__(self: Any)



##### set_kill_probability(self: Any, probability: float)

Set probability of process kill.

##### set_kill_delay_range(self: Any, min_delay: float, max_delay: float)

Set range for kill delay.

##### kill_context(self: Any)

Context manager that may kill the process.

##### _simulate_kill(self: Any)

Simulate process kill by raising an exception.

### TestNetworkFailureRecovery

Test recovery from various network failures.

#### Methods

##### test_transient_network_failure_recovery(self: Any, recovery_manager: Any, network_simulator: Any)

Test recovery from transient network failures.

##### test_rate_limit_backoff(self: Any, recovery_manager: Any, network_simulator: Any)

Test exponential backoff for rate limiting.

##### test_auth_failure_handling(self: Any, recovery_manager: Any, network_simulator: Any)

Test handling of authentication failures.

##### test_mixed_failure_types(self: Any, recovery_manager: Any, network_simulator: Any)

Test handling of mixed failure types.

### TestDiskFailureRecovery

Test recovery from disk-related failures.

#### Methods

##### test_disk_space_failure(self: Any, recovery_manager: Any, disk_simulator: Any, temp_models_dir: Any)

Test handling of disk space failures.

##### test_permission_error_recovery(self: Any, recovery_manager: Any, disk_simulator: Any, temp_models_dir: Any)

Test recovery from permission errors.

##### test_file_corruption_detection(self: Any, recovery_manager: Any, disk_simulator: Any, temp_models_dir: Any)

Test detection and handling of file corruption.

### TestProcessInterruptionRecovery

Test recovery from process interruptions.

#### Methods

##### test_process_kill_during_operation(self: Any, recovery_manager: Any, process_simulator: Any)

Test handling of process kills during operations.

##### test_partial_download_recovery(self: Any, recovery_manager: Any, temp_models_dir: Any)

Test recovery from partial downloads.

### TestConcurrencyFailureRecovery

Test recovery from concurrency-related failures.

#### Methods

##### test_lock_timeout_recovery(self: Any, recovery_manager: Any)

Test recovery from lock timeout failures.

##### test_concurrent_modification_recovery(self: Any, recovery_manager: Any)

Test recovery from concurrent modification errors.

### TestIntegratedChaosScenarios

Test complex scenarios with multiple failure types.

#### Methods

##### test_download_with_multiple_failures(self: Any, recovery_manager: Any, network_simulator: Any, disk_simulator: Any, process_simulator: Any, temp_models_dir: Any)

Test a complex download scenario with multiple failure types.

##### test_retry_configuration_effectiveness(self: Any, recovery_manager: Any)

Test that retry configurations work as expected.

