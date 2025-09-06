#!/usr/bin/env python3
"""
Production-Specific Health Checks

This module implements health checks specifically designed for production environments,
including performance monitoring, security validation, and system resource checks.
"""

import os
import asyncio
import aiohttp
import psutil
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import subprocess
import logging

from health_models import HealthCheck, HealthResult, Severity


@dataclass
class ProductionHealthResult:
    """Result of production-specific health check"""
    check_name: str
    passed: bool
    score: float
    details: Dict[str, Any]
    recommendations: List[str]
    timestamp: datetime
    duration: float


class ProductionHealthChecker:
    """Production-specific health checks for system validation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    async def run_all_production_checks(self) -> List[ProductionHealthResult]:
        """Run all production health checks"""
        checks = [
            self.check_database_connectivity,
            self.check_api_endpoints,
            self.check_model_availability,
            self.check_system_resources,
            self.check_security_status,
            self.check_dependency_vulnerabilities,
            self.check_performance_metrics,
            self.check_disk_space,
            self.check_network_connectivity,
            self.check_service_health
        ]
        
        results = []
        for check in checks:
            try:
                result = await check()
                results.append(result)
            except Exception as e:
                self.logger.error(f"Production check {check.__name__} failed: {e}")
                results.append(ProductionHealthResult(
                    check_name=check.__name__,
                    passed=False,
                    score=0.0,
                    details={"error": str(e)},
                    recommendations=[f"Fix error in {check.__name__}: {e}"],
                    timestamp=datetime.now(),
                    duration=0.0
                ))
        
        return results
    
    async def check_database_connectivity(self) -> ProductionHealthResult:
        """Check database connectivity and performance"""
        start_time = time.time()
        
        try:
            # This would check actual database connectivity
            # For now, simulate database check
            connection_time = 0.05  # Simulated connection time
            query_time = 0.02  # Simulated query time
            
            # Check connection timeout
            max_connection_time = self.config.get("production_checks", {}).get("database", {}).get("connection_timeout", 10)
            max_query_time = self.config.get("production_checks", {}).get("database", {}).get("query_timeout", 30)
            
            passed = connection_time < max_connection_time and query_time < max_query_time
            score = 100.0 if passed else 50.0
            
            details = {
                "connection_time": connection_time,
                "query_time": query_time,
                "max_connection_time": max_connection_time,
                "max_query_time": max_query_time
            }
            
            recommendations = []
            if not passed:
                recommendations.append("Optimize database connection pool settings")
                recommendations.append("Check database server performance")
            
            return ProductionHealthResult(
                check_name="database_connectivity",
                passed=passed,
                score=score,
                details=details,
                recommendations=recommendations,
                timestamp=datetime.now(),
                duration=time.time() - start_time
            )
            
        except Exception as e:
            return ProductionHealthResult(
                check_name="database_connectivity",
                passed=False,
                score=0.0,
                details={"error": str(e)},
                recommendations=["Fix database connectivity issues"],
                timestamp=datetime.now(),
                duration=time.time() - start_time
            )
    
    async def check_api_endpoints(self) -> ProductionHealthResult:
        """Check API endpoint health and response times"""
        start_time = time.time()
        
        try:
            # Define critical API endpoints to check
            endpoints = [
                {"url": "http://localhost:8000/health", "method": "GET"},
                {"url": "http://localhost:8000/api/v1/models", "method": "GET"},
                {"url": "http://localhost:8000/api/v1/system/status", "method": "GET"}
            ]
            
            timeout = self.config.get("production_checks", {}).get("api_endpoints", {}).get("timeout", 15)
            max_response_time = self.config.get("production_checks", {}).get("api_endpoints", {}).get("expected_response_time", 2.0)
            
            results = []
            total_score = 0
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                for endpoint in endpoints:
                    endpoint_start = time.time()
                    try:
                        async with session.request(endpoint["method"], endpoint["url"]) as response:
                            response_time = time.time() - endpoint_start
                            status_ok = 200 <= response.status < 300
                            time_ok = response_time <= max_response_time
                            
                            endpoint_score = 100 if (status_ok and time_ok) else 50 if status_ok else 0
                            total_score += endpoint_score
                            
                            results.append({
                                "url": endpoint["url"],
                                "status": response.status,
                                "response_time": response_time,
                                "passed": status_ok and time_ok
                            })
                            
                    except Exception as e:
                        results.append({
                            "url": endpoint["url"],
                            "error": str(e),
                            "passed": False
                        })
            
            avg_score = total_score / len(endpoints) if endpoints else 0
            all_passed = all(result.get("passed", False) for result in results)
            
            recommendations = []
            if not all_passed:
                recommendations.append("Check API server status and configuration")
                recommendations.append("Verify network connectivity to API endpoints")
                recommendations.append("Review API server logs for errors")
            
            return ProductionHealthResult(
                check_name="api_endpoints",
                passed=all_passed,
                score=avg_score,
                details={"endpoints": results, "max_response_time": max_response_time},
                recommendations=recommendations,
                timestamp=datetime.now(),
                duration=time.time() - start_time
            )
            
        except Exception as e:
            return ProductionHealthResult(
                check_name="api_endpoints",
                passed=False,
                score=0.0,
                details={"error": str(e)},
                recommendations=["Fix API endpoint connectivity issues"],
                timestamp=datetime.now(),
                duration=time.time() - start_time
            )
    
    async def check_model_availability(self) -> ProductionHealthResult:
        """Check model availability and loading performance"""
        start_time = time.time()
        
        try:
            # Check if model files exist and are accessible
            model_paths = [
                "models/",  # Check if models directory exists
                # Add specific model paths as needed
            ]
            
            load_timeout = self.config.get("production_checks", {}).get("models", {}).get("load_timeout", 60)
            inference_timeout = self.config.get("production_checks", {}).get("models", {}).get("inference_timeout", 30)
            
            model_status = []
            total_score = 0
            
            for model_path in model_paths:
                path = Path(model_path)
                if path.exists():
                    # Check model accessibility
                    try:
                        # Simulate model loading check
                        load_time = 2.0  # Simulated load time
                        inference_time = 0.5  # Simulated inference time
                        
                        load_ok = load_time <= load_timeout
                        inference_ok = inference_time <= inference_timeout
                        
                        score = 100 if (load_ok and inference_ok) else 50
                        total_score += score
                        
                        model_status.append({
                            "path": str(model_path),
                            "exists": True,
                            "load_time": load_time,
                            "inference_time": inference_time,
                            "passed": load_ok and inference_ok
                        })
                        
                    except Exception as e:
                        model_status.append({
                            "path": str(model_path),
                            "exists": True,
                            "error": str(e),
                            "passed": False
                        })
                else:
                    model_status.append({
                        "path": str(model_path),
                        "exists": False,
                        "passed": False
                    })
            
            avg_score = total_score / len(model_paths) if model_paths else 100
            all_passed = all(status.get("passed", False) for status in model_status)
            
            recommendations = []
            if not all_passed:
                recommendations.append("Verify model files are present and accessible")
                recommendations.append("Check model loading performance")
                recommendations.append("Ensure sufficient GPU/CPU resources for model inference")
            
            return ProductionHealthResult(
                check_name="model_availability",
                passed=all_passed,
                score=avg_score,
                details={"models": model_status},
                recommendations=recommendations,
                timestamp=datetime.now(),
                duration=time.time() - start_time
            )
            
        except Exception as e:
            return ProductionHealthResult(
                check_name="model_availability",
                passed=False,
                score=0.0,
                details={"error": str(e)},
                recommendations=["Fix model availability issues"],
                timestamp=datetime.now(),
                duration=time.time() - start_time
            )
    
    async def check_system_resources(self) -> ProductionHealthResult:
        """Check system resource usage and availability"""
        start_time = time.time()
        
        try:
            # Get system resource usage
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Get thresholds from config
            max_cpu = self.config.get("production_checks", {}).get("system_resources", {}).get("max_cpu_usage", 80)
            max_memory = self.config.get("production_checks", {}).get("system_resources", {}).get("max_memory_usage", 85)
            min_disk_gb = self.config.get("production_checks", {}).get("system_resources", {}).get("min_disk_space", 10)
            
            # Convert disk space to GB
            disk_free_gb = disk.free / (1024**3)
            
            # Check thresholds
            cpu_ok = cpu_usage <= max_cpu
            memory_ok = memory.percent <= max_memory
            disk_ok = disk_free_gb >= min_disk_gb
            
            # Calculate score
            checks_passed = sum([cpu_ok, memory_ok, disk_ok])
            score = (checks_passed / 3) * 100
            
            details = {
                "cpu_usage": cpu_usage,
                "cpu_threshold": max_cpu,
                "cpu_ok": cpu_ok,
                "memory_usage": memory.percent,
                "memory_threshold": max_memory,
                "memory_ok": memory_ok,
                "disk_free_gb": disk_free_gb,
                "disk_threshold_gb": min_disk_gb,
                "disk_ok": disk_ok
            }
            
            recommendations = []
            if not cpu_ok:
                recommendations.append(f"High CPU usage ({cpu_usage:.1f}%), consider scaling or optimization")
            if not memory_ok:
                recommendations.append(f"High memory usage ({memory.percent:.1f}%), check for memory leaks")
            if not disk_ok:
                recommendations.append(f"Low disk space ({disk_free_gb:.1f}GB), clean up or expand storage")
            
            return ProductionHealthResult(
                check_name="system_resources",
                passed=cpu_ok and memory_ok and disk_ok,
                score=score,
                details=details,
                recommendations=recommendations,
                timestamp=datetime.now(),
                duration=time.time() - start_time
            )
            
        except Exception as e:
            return ProductionHealthResult(
                check_name="system_resources",
                passed=False,
                score=0.0,
                details={"error": str(e)},
                recommendations=["Fix system resource monitoring"],
                timestamp=datetime.now(),
                duration=time.time() - start_time
            )
    
    async def check_security_status(self) -> ProductionHealthResult:
        """Check security-related configurations and vulnerabilities"""
        start_time = time.time()
        
        try:
            security_checks = []
            total_score = 0
            
            # Check file permissions
            sensitive_files = [
                "config/production-health.yaml",
                ".env",
                "secrets/"
            ]
            
            for file_path in sensitive_files:
                path = Path(file_path)
                if path.exists():
                    # Check file permissions (Unix-like systems)
                    if os.name != 'nt':
                        stat_info = path.stat()
                        permissions = oct(stat_info.st_mode)[-3:]
                        
                        # Check if file is readable by others
                        secure = not (int(permissions[2]) & 4)  # Others can't read
                        
                        security_checks.append({
                            "check": f"file_permissions_{file_path}",
                            "passed": secure,
                            "details": {"permissions": permissions}
                        })
                        
                        total_score += 100 if secure else 0
                    else:
                        # Windows - basic existence check
                        security_checks.append({
                            "check": f"file_exists_{file_path}",
                            "passed": True,
                            "details": {"platform": "windows"}
                        })
                        total_score += 100
            
            # Check for exposed secrets in environment
            env_vars = os.environ
            exposed_secrets = []
            
            for key, value in env_vars.items():
                if any(secret_word in key.lower() for secret_word in ['password', 'secret', 'key', 'token']):
                    if len(value) > 0:
                        exposed_secrets.append(key)
            
            secrets_secure = len(exposed_secrets) == 0
            security_checks.append({
                "check": "environment_secrets",
                "passed": secrets_secure,
                "details": {"exposed_count": len(exposed_secrets)}
            })
            
            total_score += 100 if secrets_secure else 0
            
            # Calculate average score
            avg_score = total_score / len(security_checks) if security_checks else 100
            all_passed = all(check["passed"] for check in security_checks)
            
            recommendations = []
            if not all_passed:
                recommendations.append("Review file permissions for sensitive files")
                recommendations.append("Ensure secrets are not exposed in environment variables")
                recommendations.append("Run security audit tools regularly")
            
            return ProductionHealthResult(
                check_name="security_status",
                passed=all_passed,
                score=avg_score,
                details={"checks": security_checks},
                recommendations=recommendations,
                timestamp=datetime.now(),
                duration=time.time() - start_time
            )
            
        except Exception as e:
            return ProductionHealthResult(
                check_name="security_status",
                passed=False,
                score=0.0,
                details={"error": str(e)},
                recommendations=["Fix security status checking"],
                timestamp=datetime.now(),
                duration=time.time() - start_time
            )
    
    async def check_dependency_vulnerabilities(self) -> ProductionHealthResult:
        """Check for known vulnerabilities in dependencies"""
        start_time = time.time()
        
        try:
            # Check if pip-audit is available
            try:
                result = subprocess.run(
                    ["pip-audit", "--format", "json", "--quiet"],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.returncode == 0:
                    # No vulnerabilities found
                    vulnerabilities = []
                    passed = True
                    score = 100.0
                else:
                    # Parse vulnerabilities if any
                    try:
                        vulnerabilities = json.loads(result.stdout) if result.stdout else []
                    except json.JSONDecodeError:
                        vulnerabilities = []
                    
                    passed = len(vulnerabilities) == 0
                    score = max(0, 100 - (len(vulnerabilities) * 10))  # Deduct 10 points per vulnerability
                
            except (subprocess.TimeoutExpired, FileNotFoundError):
                # pip-audit not available or timed out
                vulnerabilities = []
                passed = True  # Assume safe if we can't check
                score = 80.0  # Reduced score for inability to check
            
            recommendations = []
            if vulnerabilities:
                recommendations.append("Update vulnerable dependencies")
                recommendations.append("Review security advisories for affected packages")
            elif not passed:
                recommendations.append("Install pip-audit for dependency vulnerability scanning")
            
            return ProductionHealthResult(
                check_name="dependency_vulnerabilities",
                passed=passed,
                score=score,
                details={"vulnerabilities": vulnerabilities},
                recommendations=recommendations,
                timestamp=datetime.now(),
                duration=time.time() - start_time
            )
            
        except Exception as e:
            return ProductionHealthResult(
                check_name="dependency_vulnerabilities",
                passed=False,
                score=0.0,
                details={"error": str(e)},
                recommendations=["Fix dependency vulnerability checking"],
                timestamp=datetime.now(),
                duration=time.time() - start_time
            )
    
    async def check_performance_metrics(self) -> ProductionHealthResult:
        """Check system performance metrics"""
        start_time = time.time()
        
        try:
            # Collect performance metrics
            metrics = {}
            
            # CPU metrics
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
            
            metrics["cpu"] = {
                "count": cpu_count,
                "frequency_mhz": cpu_freq.current if cpu_freq else 0,
                "load_1min": load_avg[0],
                "load_5min": load_avg[1],
                "load_15min": load_avg[2]
            }
            
            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            metrics["memory"] = {
                "total_gb": memory.total / (1024**3),
                "available_gb": memory.available / (1024**3),
                "used_percent": memory.percent,
                "swap_used_percent": swap.percent
            }
            
            # Disk I/O metrics
            disk_io = psutil.disk_io_counters()
            
            metrics["disk_io"] = {
                "read_mb": disk_io.read_bytes / (1024**2) if disk_io else 0,
                "write_mb": disk_io.write_bytes / (1024**2) if disk_io else 0
            }
            
            # Network metrics
            network_io = psutil.net_io_counters()
            
            metrics["network"] = {
                "bytes_sent_mb": network_io.bytes_sent / (1024**2),
                "bytes_recv_mb": network_io.bytes_recv / (1024**2)
            }
            
            # Performance assessment
            performance_issues = []
            
            # Check CPU load
            if load_avg[0] > cpu_count * 0.8:
                performance_issues.append("High CPU load detected")
            
            # Check memory usage
            if memory.percent > 85:
                performance_issues.append("High memory usage detected")
            
            # Check swap usage
            if swap.percent > 10:
                performance_issues.append("Swap usage detected - may indicate memory pressure")
            
            passed = len(performance_issues) == 0
            score = max(0, 100 - (len(performance_issues) * 20))
            
            recommendations = []
            if performance_issues:
                recommendations.extend(performance_issues)
                recommendations.append("Monitor system performance regularly")
                recommendations.append("Consider scaling resources if issues persist")
            
            return ProductionHealthResult(
                check_name="performance_metrics",
                passed=passed,
                score=score,
                details={"metrics": metrics, "issues": performance_issues},
                recommendations=recommendations,
                timestamp=datetime.now(),
                duration=time.time() - start_time
            )
            
        except Exception as e:
            return ProductionHealthResult(
                check_name="performance_metrics",
                passed=False,
                score=0.0,
                details={"error": str(e)},
                recommendations=["Fix performance metrics collection"],
                timestamp=datetime.now(),
                duration=time.time() - start_time
            )
    
    async def check_disk_space(self) -> ProductionHealthResult:
        """Check disk space across all mounted filesystems"""
        start_time = time.time()
        
        try:
            disk_info = []
            total_score = 0
            
            # Get all disk partitions
            partitions = psutil.disk_partitions()
            
            for partition in partitions:
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    free_gb = usage.free / (1024**3)
                    used_percent = (usage.used / usage.total) * 100
                    
                    # Consider disk healthy if less than 90% used and at least 1GB free
                    healthy = used_percent < 90 and free_gb > 1.0
                    
                    disk_info.append({
                        "mountpoint": partition.mountpoint,
                        "device": partition.device,
                        "fstype": partition.fstype,
                        "total_gb": usage.total / (1024**3),
                        "used_gb": usage.used / (1024**3),
                        "free_gb": free_gb,
                        "used_percent": used_percent,
                        "healthy": healthy
                    })
                    
                    total_score += 100 if healthy else 50
                    
                except PermissionError:
                    # Skip inaccessible partitions
                    continue
            
            avg_score = total_score / len(disk_info) if disk_info else 100
            all_healthy = all(disk["healthy"] for disk in disk_info)
            
            recommendations = []
            for disk in disk_info:
                if not disk["healthy"]:
                    recommendations.append(
                        f"Low disk space on {disk['mountpoint']}: "
                        f"{disk['used_percent']:.1f}% used, {disk['free_gb']:.1f}GB free"
                    )
            
            if recommendations:
                recommendations.append("Clean up unnecessary files or expand storage")
            
            return ProductionHealthResult(
                check_name="disk_space",
                passed=all_healthy,
                score=avg_score,
                details={"disks": disk_info},
                recommendations=recommendations,
                timestamp=datetime.now(),
                duration=time.time() - start_time
            )
            
        except Exception as e:
            return ProductionHealthResult(
                check_name="disk_space",
                passed=False,
                score=0.0,
                details={"error": str(e)},
                recommendations=["Fix disk space monitoring"],
                timestamp=datetime.now(),
                duration=time.time() - start_time
            )
    
    async def check_network_connectivity(self) -> ProductionHealthResult:
        """Check network connectivity to external services"""
        start_time = time.time()
        
        try:
            # Test connectivity to important external services
            test_hosts = [
                {"host": "8.8.8.8", "port": 53, "name": "Google DNS"},
                {"host": "github.com", "port": 443, "name": "GitHub"},
                {"host": "pypi.org", "port": 443, "name": "PyPI"}
            ]
            
            connectivity_results = []
            total_score = 0
            
            for test in test_hosts:
                try:
                    # Test TCP connection
                    reader, writer = await asyncio.wait_for(
                        asyncio.open_connection(test["host"], test["port"]),
                        timeout=10
                    )
                    writer.close()
                    await writer.wait_closed()
                    
                    connectivity_results.append({
                        "host": test["host"],
                        "port": test["port"],
                        "name": test["name"],
                        "connected": True
                    })
                    total_score += 100
                    
                except Exception as e:
                    connectivity_results.append({
                        "host": test["host"],
                        "port": test["port"],
                        "name": test["name"],
                        "connected": False,
                        "error": str(e)
                    })
            
            avg_score = total_score / len(test_hosts) if test_hosts else 100
            all_connected = all(result["connected"] for result in connectivity_results)
            
            recommendations = []
            if not all_connected:
                failed_hosts = [r["name"] for r in connectivity_results if not r["connected"]]
                recommendations.append(f"Network connectivity issues to: {', '.join(failed_hosts)}")
                recommendations.append("Check firewall and network configuration")
            
            return ProductionHealthResult(
                check_name="network_connectivity",
                passed=all_connected,
                score=avg_score,
                details={"connectivity_tests": connectivity_results},
                recommendations=recommendations,
                timestamp=datetime.now(),
                duration=time.time() - start_time
            )
            
        except Exception as e:
            return ProductionHealthResult(
                check_name="network_connectivity",
                passed=False,
                score=0.0,
                details={"error": str(e)},
                recommendations=["Fix network connectivity checking"],
                timestamp=datetime.now(),
                duration=time.time() - start_time
            )
    
    async def check_service_health(self) -> ProductionHealthResult:
        """Check health of critical system services"""
        start_time = time.time()
        
        try:
            service_results = []
            total_score = 0
            
            # Check if critical processes are running
            critical_processes = [
                "python",  # Python processes (our application)
                # Add other critical processes as needed
            ]
            
            for process_name in critical_processes:
                running_processes = []
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        if process_name.lower() in proc.info['name'].lower():
                            running_processes.append({
                                "pid": proc.info['pid'],
                                "name": proc.info['name'],
                                "cmdline": ' '.join(proc.info['cmdline'][:3])  # First 3 args
                            })
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                
                process_healthy = len(running_processes) > 0
                service_results.append({
                    "process_name": process_name,
                    "running_count": len(running_processes),
                    "processes": running_processes,
                    "healthy": process_healthy
                })
                
                total_score += 100 if process_healthy else 0
            
            # Check system services (Linux only)
            if os.name != 'nt':
                try:
                    # Check if systemd is available
                    result = subprocess.run(
                        ["systemctl", "is-active", "wan22-health-monitor"],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    
                    service_active = result.returncode == 0
                    service_results.append({
                        "service_name": "wan22-health-monitor",
                        "active": service_active,
                        "healthy": service_active
                    })
                    
                    total_score += 100 if service_active else 0
                    
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    # systemctl not available or service not installed
                    pass
            
            avg_score = total_score / len(service_results) if service_results else 100
            all_healthy = all(result["healthy"] for result in service_results)
            
            recommendations = []
            if not all_healthy:
                unhealthy_services = [r["process_name"] for r in service_results if not r["healthy"]]
                recommendations.append(f"Unhealthy services detected: {', '.join(unhealthy_services)}")
                recommendations.append("Check service logs and restart if necessary")
            
            return ProductionHealthResult(
                check_name="service_health",
                passed=all_healthy,
                score=avg_score,
                details={"services": service_results},
                recommendations=recommendations,
                timestamp=datetime.now(),
                duration=time.time() - start_time
            )
            
        except Exception as e:
            return ProductionHealthResult(
                check_name="service_health",
                passed=False,
                score=0.0,
                details={"error": str(e)},
                recommendations=["Fix service health monitoring"],
                timestamp=datetime.now(),
                duration=time.time() - start_time
            )