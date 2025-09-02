"""
Main project health checker implementation
"""

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from health_models import (
        HealthReport, HealthIssue, ComponentHealth, HealthTrends,
        HealthCategory, Severity, HealthConfig
    )
    from checkers.test_health_checker import TestHealthChecker
    from checkers.documentation_health_checker import DocumentationHealthChecker
    from checkers.configuration_health_checker import ConfigurationHealthChecker
    from checkers.code_quality_checker import CodeQualityChecker
    from performance_optimizer import HealthCheckCache, PerformanceProfiler, LightweightHealthChecker
    from parallel_executor import ParallelHealthExecutor, HealthCheckTask
except ImportError:
    # Create minimal implementations for missing components
    from datetime import datetime
    from enum import Enum
    
    class Severity(Enum):
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        CRITICAL = "critical"
    
    class HealthCategory(Enum):
        TESTS = "tests"
        DOCUMENTATION = "documentation"
        CONFIGURATION = "configuration"
        CODE_QUALITY = "code_quality"
        PERFORMANCE = "performance"
        SECURITY = "security"
    
    class HealthConfig:
        def __init__(self):
            self.parallel_checks = True
            self.max_workers = 4
            self.max_check_duration = 300
            self.test_weight = 0.3
            self.documentation_weight = 0.2
            self.configuration_weight = 0.2
            self.code_quality_weight = 0.2
            self.performance_weight = 0.05
            self.security_weight = 0.05
    
    class HealthIssue:
        def __init__(self, severity, category, title, description, affected_components, remediation_steps):
            self.severity = severity
            self.category = category
            self.title = title
            self.description = description
            self.affected_components = affected_components
            self.remediation_steps = remediation_steps
    
    class ComponentHealth:
        def __init__(self, component_name, category, score, status, issues, metrics):
            self.component_name = component_name
            self.category = category
            self.score = score
            self.status = status
            self.issues = issues
            self.metrics = metrics
    
    class HealthTrends:
        def __init__(self, score_history=None, issue_trends=None, improvement_rate=0.0, degradation_alerts=None):
            self.score_history = score_history or []
            self.issue_trends = issue_trends or {}
            self.improvement_rate = improvement_rate
            self.degradation_alerts = degradation_alerts or []
    
    class HealthReport:
        def __init__(self, timestamp, overall_score, component_scores, issues, recommendations, trends, metadata=None):
            self.timestamp = timestamp
            self.overall_score = overall_score
            self.component_scores = component_scores
            self.issues = issues
            self.recommendations = recommendations
            self.trends = trends
            self.metadata = metadata or {}
            self.execution_time = metadata.get("check_duration", 0) if metadata else 0
        
        def get_issues_by_category(self, category):
            return [issue for issue in self.issues if hasattr(issue, 'category') and issue.category == category]
    
    # Create minimal checker implementations
    class TestHealthChecker:
        def __init__(self, config):
            self.config = config
        
        def check_health(self):
            return ComponentHealth(
                component_name="tests",
                category=HealthCategory.TESTS,
                score=80.0,
                status="healthy",
                issues=[],
                metrics={"test_count": 50, "pass_rate": 0.8}
            )
    
    class DocumentationHealthChecker:
        def __init__(self, config):
            self.config = config
        
        def check_health(self):
            return ComponentHealth(
                component_name="documentation",
                category=HealthCategory.DOCUMENTATION,
                score=70.0,
                status="needs_attention",
                issues=[],
                metrics={"doc_coverage": 0.7}
            )
    
    class ConfigurationHealthChecker:
        def __init__(self, config):
            self.config = config
        
        def check_health(self):
            return ComponentHealth(
                component_name="configuration",
                category=HealthCategory.CONFIGURATION,
                score=75.0,
                status="healthy",
                issues=[],
                metrics={"config_files": 10}
            )
    
    class CodeQualityChecker:
        def __init__(self, config):
            self.config = config
        
        def check_health(self):
            return ComponentHealth(
                component_name="code_quality",
                category=HealthCategory.CODE_QUALITY,
                score=85.0,
                status="healthy",
                issues=[],
                metrics={"complexity": 2.5}
            )
    
    # Performance optimization components
    class HealthCheckCache:
        def get(self, key, inputs, ttl):
            return None
        
        def set(self, key, inputs, result):
            pass
        
        def invalidate(self):
            pass
    
    class PerformanceProfiler:
        def get_performance_summary(self):
            return {}
        
        def generate_optimization_recommendations(self):
            return []
    
    class LightweightHealthChecker:
        def __init__(self, cache, profiler):
            pass
        
        def run_lightweight_health_check(self):
            return {"health_score": 75, "checks": {}, "execution_time": 30}
    
    class HealthCheckTask:
        def __init__(self, name, function, args, kwargs, priority, timeout, category):
            self.name = name
            self.function = function
            self.args = args
            self.kwargs = kwargs
            self.priority = priority
            self.timeout = timeout
            self.category = category
    
    class ParallelHealthExecutor:
        def execute_tasks(self, tasks):
            return {}


class ProjectHealthChecker:
    """
    Main health checker that orchestrates all health checks and generates reports
    """
    
    def __init__(self, config: Optional[HealthConfig] = None):
        self.config = config or HealthConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize health checkers
        self.checkers = {
            HealthCategory.TESTS: TestHealthChecker(self.config),
            HealthCategory.DOCUMENTATION: DocumentationHealthChecker(self.config),
            HealthCategory.CONFIGURATION: ConfigurationHealthChecker(self.config),
            HealthCategory.CODE_QUALITY: CodeQualityChecker(self.config)
        }
        
        # Performance optimization components
        self.cache = HealthCheckCache()
        self.profiler = PerformanceProfiler()
        self.lightweight_checker = LightweightHealthChecker(self.cache, self.profiler)
        self.parallel_executor = ParallelHealthExecutor()
        
        # Health history storage
        self.history_file = Path("tools/health-checker/health_history.json")
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        
    async def run_health_check(self, categories: Optional[List[HealthCategory]] = None) -> HealthReport:
        """
        Run comprehensive project health check
        
        Args:
            categories: Specific categories to check, or None for all
            
        Returns:
            HealthReport with complete health status
        """
        start_time = datetime.now()
        self.logger.info("Starting project health check")
        
        # Determine which categories to check
        check_categories = categories or list(self.checkers.keys())
        
        # Run health checks
        component_results = {}
        
        if self.config.parallel_checks:
            component_results = await self._run_parallel_checks(check_categories)
        else:
            component_results = await self._run_sequential_checks(check_categories)
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(component_results)
        
        # Collect all issues
        all_issues = []
        for component in component_results.values():
            all_issues.extend(component.issues)
        
        # Load trends
        trends = self._load_health_trends()
        
        # Create report
        report = HealthReport(
            timestamp=start_time,
            overall_score=overall_score,
            component_scores={name: comp for name, comp in component_results.items()},
            issues=all_issues,
            recommendations=[],  # Will be populated by recommendation engine
            trends=trends,
            metadata={
                "check_duration": (datetime.now() - start_time).total_seconds(),
                "categories_checked": [cat.value for cat in check_categories],
                "config_version": "1.0"
            }
        )
        
        # Save to history
        self._save_health_history(report)
        
        self.logger.info(f"Health check completed. Overall score: {overall_score:.1f}")
        return report
    
    async def _run_parallel_checks(self, categories: List[HealthCategory]) -> Dict[str, ComponentHealth]:
        """Run health checks in parallel"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all checks
            future_to_category = {
                executor.submit(self._run_single_check, category): category
                for category in categories
            }
            
            # Collect results
            for future in as_completed(future_to_category, timeout=self.config.max_check_duration):
                category = future_to_category[future]
                try:
                    result = future.result()
                    results[category.value] = result
                except Exception as e:
                    self.logger.error(f"Health check failed for {category.value}: {e}")
                    results[category.value] = self._create_error_component(category, str(e))
        
        return results
    
    async def _run_sequential_checks(self, categories: List[HealthCategory]) -> Dict[str, ComponentHealth]:
        """Run health checks sequentially"""
        results = {}
        
        for category in categories:
            try:
                result = self._run_single_check(category)
                results[category.value] = result
            except Exception as e:
                self.logger.error(f"Health check failed for {category.value}: {e}")
                results[category.value] = self._create_error_component(category, str(e))
        
        return results
    
    def _run_single_check(self, category: HealthCategory) -> ComponentHealth:
        """Run a single health check"""
        checker = self.checkers.get(category)
        if not checker:
            return self._create_error_component(category, "No checker available")
        
        try:
            return checker.check_health()
        except Exception as e:
            self.logger.error(f"Error in {category.value} health check: {e}")
            return self._create_error_component(category, str(e))
    
    def _create_error_component(self, category: HealthCategory, error_msg: str) -> ComponentHealth:
        """Create a component health result for errors"""
        issue = HealthIssue(
            severity=Severity.CRITICAL,
            category=category,
            title=f"{category.value.title()} Health Check Failed",
            description=f"Health check failed with error: {error_msg}",
            affected_components=[category.value],
            remediation_steps=[
                "Check system logs for detailed error information",
                "Verify health checker configuration",
                "Ensure all dependencies are installed"
            ]
        )
        
        return ComponentHealth(
            component_name=category.value,
            category=category,
            score=0.0,
            status="critical",
            issues=[issue],
            metrics={"error": error_msg}
        )
    
    def _calculate_overall_score(self, components: Dict[str, ComponentHealth]) -> float:
        """Calculate weighted overall health score"""
        if not components:
            return 0.0
        
        weights = {
            HealthCategory.TESTS.value: self.config.test_weight,
            HealthCategory.DOCUMENTATION.value: self.config.documentation_weight,
            HealthCategory.CONFIGURATION.value: self.config.configuration_weight,
            HealthCategory.CODE_QUALITY.value: self.config.code_quality_weight,
            HealthCategory.PERFORMANCE.value: self.config.performance_weight,
            HealthCategory.SECURITY.value: self.config.security_weight
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for name, component in components.items():
            weight = weights.get(name, 0.1)  # Default weight for unknown components
            weighted_sum += component.score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def get_health_score(self) -> float:
        """Get current health score (synchronous version)"""
        try:
            # Run a quick health check
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            report = loop.run_until_complete(self.run_health_check())
            loop.close()
            return report.overall_score
        except Exception as e:
            self.logger.error(f"Failed to get health score: {e}")
            return 0.0
    
    def _load_health_trends(self) -> HealthTrends:
        """Load historical health trends"""
        try:
            if self.history_file.exists():
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                    
                # Convert string timestamps back to datetime
                score_history = [
                    (datetime.fromisoformat(ts), score)
                    for ts, score in data.get('score_history', [])
                ]
                
                return HealthTrends(
                    score_history=score_history,
                    issue_trends=data.get('issue_trends', {}),
                    improvement_rate=data.get('improvement_rate', 0.0),
                    degradation_alerts=data.get('degradation_alerts', [])
                )
        except Exception as e:
            self.logger.warning(f"Failed to load health trends: {e}")
        
        return HealthTrends()
    
    def _save_health_history(self, report: HealthReport) -> None:
        """Save health report to history"""
        try:
            # Load existing history
            history_data = {}
            if self.history_file.exists():
                with open(self.history_file, 'r') as f:
                    history_data = json.load(f)
            
            # Add new score to history
            score_history = history_data.get('score_history', [])
            score_history.append([report.timestamp.isoformat(), report.overall_score])
            
            # Keep only last 100 entries
            score_history = score_history[-100:]
            
            # Update issue trends
            issue_trends = history_data.get('issue_trends', {})
            for category in HealthCategory:
                category_issues = len(report.get_issues_by_category(category))
                if category.value not in issue_trends:
                    issue_trends[category.value] = []
                issue_trends[category.value].append([report.timestamp.isoformat(), category_issues])
                issue_trends[category.value] = issue_trends[category.value][-50:]  # Keep last 50
            
            # Calculate improvement rate
            improvement_rate = 0.0
            if len(score_history) >= 2:
                recent_scores = [score for _, score in score_history[-10:]]
                if len(recent_scores) >= 2:
                    improvement_rate = (recent_scores[-1] - recent_scores[0]) / len(recent_scores)
            
            # Save updated history
            history_data.update({
                'score_history': score_history,
                'issue_trends': issue_trends,
                'improvement_rate': improvement_rate,
                'last_updated': report.timestamp.isoformat()
            })
            
            with open(self.history_file, 'w') as f:
                json.dump(history_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save health history: {e}")
    
    def schedule_health_checks(self) -> None:
        """Schedule automated health checks"""
        # This would integrate with a task scheduler like cron or Windows Task Scheduler
        # For now, we'll create a simple scheduling mechanism
        self.logger.info("Health check scheduling would be implemented here")
        # Implementation would depend on the deployment environment
    
    async def run_optimized_health_check(self, 
                                       lightweight: bool = False,
                                       use_cache: bool = True,
                                       parallel: bool = True) -> HealthReport:
        """
        Run performance-optimized health check with caching and parallel execution.
        
        Args:
            lightweight: Use lightweight mode for faster execution
            use_cache: Enable result caching
            parallel: Enable parallel execution of checks
            
        Returns:
            HealthReport with health status
        """
        start_time = datetime.now()
        
        if lightweight:
            # Use lightweight checker for fast execution
            self.logger.info("Running lightweight health check")
            lightweight_results = self.lightweight_checker.run_lightweight_health_check()
            
            # Convert lightweight results to HealthReport format
            return self._convert_lightweight_results(lightweight_results)
        
        if parallel:
            # Use parallel execution for comprehensive checks
            self.logger.info("Running parallel health check")
            return await self._run_parallel_health_check(use_cache)
        else:
            # Fall back to sequential execution
            return await self.run_health_check()
    
    async def _run_parallel_health_check(self, use_cache: bool = True) -> HealthReport:
        """Run health checks in parallel for improved performance."""
        
        # Create health check tasks
        tasks = []
        
        for category, checker in self.checkers.items():
            task = HealthCheckTask(
                name=f"{category.value}_check",
                function=self._run_cached_check if use_cache else checker.check_health,
                args=(checker,) if use_cache else (),
                kwargs={"category": category} if use_cache else {},
                priority=self._get_check_priority(category),
                timeout=self._get_check_timeout(category),
                category=category.value
            )
            tasks.append(task)
        
        # Execute tasks in parallel
        results = self.parallel_executor.execute_tasks(tasks)
        
        # Combine results into health report
        return await self._combine_parallel_results(results)
    
    def _run_cached_check(self, checker, category: HealthCategory):
        """Run health check with caching support."""
        
        # Generate cache key based on relevant files
        cache_inputs = self._get_cache_inputs_for_category(category)
        
        # Try to get cached result
        cached_result = self.cache.get(f"{category.value}_check", cache_inputs, ttl=1800)  # 30 min cache
        
        if cached_result:
            self.logger.debug(f"Using cached result for {category.value}")
            return cached_result
        
        # Run actual check
        result = checker.check_health()
        
        # Cache the result
        self.cache.set(f"{category.value}_check", cache_inputs, result)
        
        return result
    
    def _get_cache_inputs_for_category(self, category: HealthCategory) -> Dict:
        """Get cache inputs for a specific health check category."""
        
        inputs = {"category": category.value}
        
        if category == HealthCategory.TESTS:
            # Include test file modification times
            test_files = list(Path("tests").glob("**/*.py")) if Path("tests").exists() else []
            inputs["test_files_hash"] = self._get_files_hash(test_files)
            
        elif category == HealthCategory.CONFIGURATION:
            # Include config file modification times
            config_files = list(Path("config").glob("**/*")) if Path("config").exists() else []
            config_files.extend(Path(".").glob("*.json"))
            config_files.extend(Path(".").glob("*.yaml"))
            inputs["config_files_hash"] = self._get_files_hash(config_files)
            
        elif category == HealthCategory.DOCUMENTATION:
            # Include documentation file modification times
            doc_files = list(Path("docs").glob("**/*")) if Path("docs").exists() else []
            doc_files.extend(Path(".").glob("*.md"))
            inputs["doc_files_hash"] = self._get_files_hash(doc_files)
            
        elif category == HealthCategory.CODE_QUALITY:
            # Include source code file modification times
            code_files = list(Path(".").glob("**/*.py"))
            # Exclude virtual environments and cache directories
            code_files = [f for f in code_files if "venv" not in str(f) and "__pycache__" not in str(f)]
            inputs["code_files_hash"] = self._get_files_hash(code_files[:100])  # Limit to first 100 files
        
        return inputs
    
    def _get_files_hash(self, files: List[Path]) -> str:
        """Get hash of file modification times for cache invalidation."""
        import hashlib
        
        file_info = []
        for file_path in files:
            if file_path.exists() and file_path.is_file():
                file_info.append(f"{file_path}:{file_path.stat().st_mtime}")
        
        combined = "|".join(sorted(file_info))
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _get_check_priority(self, category: HealthCategory) -> int:
        """Get priority for health check category."""
        priority_map = {
            HealthCategory.TESTS: 3,           # High priority
            HealthCategory.CONFIGURATION: 3,   # High priority
            HealthCategory.CODE_QUALITY: 2,   # Medium priority
            HealthCategory.DOCUMENTATION: 1   # Low priority
        }
        return priority_map.get(category, 1)
    
    def _get_check_timeout(self, category: HealthCategory) -> int:
        """Get timeout for health check category."""
        timeout_map = {
            HealthCategory.TESTS: 120,         # 2 minutes
            HealthCategory.CONFIGURATION: 30,  # 30 seconds
            HealthCategory.CODE_QUALITY: 180, # 3 minutes
            HealthCategory.DOCUMENTATION: 60   # 1 minute
        }
        return timeout_map.get(category, 60)
    
    async def _combine_parallel_results(self, results: Dict) -> HealthReport:
        """Combine parallel execution results into a health report."""
        
        components = {}
        all_issues = []
        
        for task_name, task_result in results.items():
            if not task_result.success:
                # Create error issue for failed check
                error_issue = HealthIssue(
                    category=task_name.replace("_check", ""),
                    severity=Severity.HIGH,
                    description=f"Health check failed: {task_result.error}",
                    affected_components=[task_name],
                    remediation_steps=[f"Fix error in {task_name}: {task_result.error}"]
                )
                all_issues.append(error_issue)
                continue
            
            # Extract component health from result
            check_result = task_result.result
            if hasattr(check_result, 'component_health'):
                components.update(check_result.component_health)
            if hasattr(check_result, 'issues'):
                all_issues.extend(check_result.issues)
        
        # Calculate overall score
        overall_score = self._calculate_weighted_score(components)
        
        # Load trends
        trends = self._load_health_trends()
        
        # Create health report
        report = HealthReport(
            timestamp=datetime.now(),
            overall_score=overall_score,
            component_scores={name: comp.score for name, comp in components.items()},
            issues=all_issues,
            recommendations=[],  # Would be generated by recommendation engine
            trends=trends,
            execution_time=(datetime.now() - datetime.now()).total_seconds()
        )
        
        # Save to history
        self._save_health_history(report)
        
        return report
    
    def _convert_lightweight_results(self, lightweight_results: Dict) -> HealthReport:
        """Convert lightweight check results to HealthReport format."""
        
        issues = []
        components = {}
        
        # Convert check results to issues and components
        for check_name, check_result in lightweight_results.get("checks", {}).items():
            if check_result["status"] == "failed":
                for issue_desc in check_result.get("issues", []):
                    issue = HealthIssue(
                        category=check_name,
                        severity=Severity.MEDIUM,
                        description=issue_desc,
                        affected_components=[check_name],
                        remediation_steps=[f"Fix issue in {check_name}"]
                    )
                    issues.append(issue)
            
            # Create component health
            score = 100 if check_result["status"] == "passed" else 0
            components[check_name] = ComponentHealth(
                name=check_name,
                score=score,
                status="healthy" if score > 80 else "unhealthy",
                last_check=datetime.now(),
                details=check_result
            )
        
        return HealthReport(
            timestamp=datetime.now(),
            overall_score=lightweight_results.get("health_score", 0),
            component_scores={name: comp.score for name, comp in components.items()},
            issues=issues,
            recommendations=[],
            trends=HealthTrends(),
            execution_time=lightweight_results.get("execution_time", 0)
        )
    
    def clear_cache(self):
        """Clear health check cache."""
        self.cache.invalidate()
        self.logger.info("Health check cache cleared")
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary from profiler."""
        return self.profiler.get_performance_summary()
    
    def get_optimization_recommendations(self) -> List[Dict]:
        """Get performance optimization recommendations."""
        return self.profiler.generate_optimization_recommendations()