#!/usr/bin/env python3
"""
Integration tests for the Unified CLI Tool

Tests all major functionality including tool integration,
workflow automation, team collaboration, and IDE integration.
"""

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import sys

# Add the tools directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from unified_cli.cli import UnifiedCLI, WorkflowContext, TeamCollaborationConfig
from unified_cli.workflow_automation import WorkflowAutomation, AutomationRule
from unified_cli.ide_integration import IDEIntegration, QualityIssue


class TestUnifiedCLI(unittest.TestCase):
    """Test the main UnifiedCLI functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        # Create .kiro directory
        self.kiro_dir = Path(self.temp_dir) / ".kiro"
        self.kiro_dir.mkdir(exist_ok=True)
        
        self.cli = UnifiedCLI()
    
    def tearDown(self):
        """Clean up test environment"""
        os.chdir(self.original_cwd)
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_tool_registry(self):
        """Test that all expected tools are registered"""
        expected_tools = [
            'test-audit', 'test-coverage', 'test-runner',
            'config', 'config-analyzer', 'structure',
            'cleanup', 'quality', 'review', 'health',
            'monitor', 'maintenance', 'report', 'docs',
            'dev-env', 'feedback', 'onboarding'
        ]
        
        for tool in expected_tools:
            self.assertIn(tool, self.cli.tools, f"Tool {tool} not registered")
    
    def test_workflow_contexts(self):
        """Test that all workflow contexts are configured"""
        expected_contexts = [
            WorkflowContext.PRE_COMMIT,
            WorkflowContext.POST_COMMIT,
            WorkflowContext.DAILY_MAINTENANCE,
            WorkflowContext.WEEKLY_CLEANUP,
            WorkflowContext.RELEASE_PREP,
            WorkflowContext.ONBOARDING,
            WorkflowContext.DEBUGGING
        ]
        
        for context in expected_contexts:
            self.assertIn(context, self.cli.workflow_configs,
                         f"Workflow context {context} not configured")
    
    @patch('subprocess.run')
    def test_git_context_detection(self, mock_run):
        """Test git context detection"""
        # Test staged changes (pre-commit)
        mock_run.side_effect = [
            Mock(returncode=0, stdout=''),  # git status
            Mock(returncode=0, stdout='file.py\n')  # git diff --cached
        ]
        
        context = self.cli.get_context_from_git_status()
        self.assertEqual(context, WorkflowContext.PRE_COMMIT)
        
        # Test no staged changes (post-commit)
        mock_run.side_effect = [
            Mock(returncode=0, stdout=''),  # git status
            Mock(returncode=0, stdout=''),  # git diff --cached
            Mock(returncode=0, stdout='abc123 Latest commit')  # git log
        ]
        
        context = self.cli.get_context_from_git_status()
        self.assertEqual(context, WorkflowContext.POST_COMMIT)
    
    def test_team_collaboration_setup(self):
        """Test team collaboration setup"""
        team_name = "Test Team"
        standards = {
            'code_style': 'pep8',
            'max_complexity': 8
        }
        
        self.cli.setup_team_collaboration(team_name, standards)
        
        # Check that config was created
        config_file = self.kiro_dir / "team-config.json"
        self.assertTrue(config_file.exists())
        
        # Check config content
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        self.assertEqual(config_data['team_name'], team_name)
        self.assertEqual(config_data['shared_standards']['code_style'], 'pep8')
        self.assertEqual(config_data['shared_standards']['max_complexity'], 8)
    
    def test_standards_sharing(self):
        """Test sharing and importing team standards"""
        # Set up team
        self.cli.setup_team_collaboration("Test Team")
        
        # Share standards
        standards_file = "test_standards.json"
        self.cli.share_standards(standards_file)
        
        # Check file was created
        self.assertTrue(Path(standards_file).exists())
        
        # Import standards to new CLI instance
        new_cli = UnifiedCLI()
        new_cli.import_standards(standards_file)
        
        # Check standards were imported
        self.assertIsNotNone(new_cli.team_config)
        self.assertEqual(new_cli.team_config.team_name, "Test Team")
    
    def test_quality_gates(self):
        """Test quality gates functionality"""
        # Set up team with quality gates
        self.cli.setup_team_collaboration("Test Team")
        
        # Mock tool results
        with patch.object(self.cli, 'run_tools_sync') as mock_run:
            mock_run.return_value = [
                Mock(success=True, tool_name='quality'),
                Mock(success=True, tool_name='test-audit')
            ]
            
            result = self.cli.check_quality_gates('pre_commit')
            self.assertTrue(result)
            
            # Test failure case
            mock_run.return_value = [
                Mock(success=True, tool_name='quality'),
                Mock(success=False, tool_name='test-audit')
            ]
            
            result = self.cli.check_quality_gates('pre_commit')
            self.assertFalse(result)
    
    def test_team_notifications(self):
        """Test team notification system"""
        # Set up team
        self.cli.setup_team_collaboration("Test Team")
        
        # Test console notification
        with patch('builtins.print') as mock_print:
            self.cli.notify_team("Test message", "info")
            mock_print.assert_called()
            
            # Check message format
            call_args = mock_print.call_args[0][0]
            self.assertIn("TEAM NOTIFICATION", call_args)
            self.assertIn("Test message", call_args)
            self.assertIn("INFO", call_args)
    
    def test_team_report_generation(self):
        """Test team report generation"""
        # Set up team
        self.cli.setup_team_collaboration("Test Team")
        
        # Mock tool results
        with patch.object(self.cli, 'run_tool') as mock_run:
            mock_run.return_value = Mock(
                success=True,
                details={
                    'test_coverage': 75,
                    'quality_score': 7,
                    'documentation_coverage': 60,
                    'recent_operations': []
                }
            )
            
            report = self.cli.generate_team_report()
            
            self.assertEqual(report['team_name'], "Test Team")
            self.assertIn('timestamp', report)
            self.assertIn('recommendations', report)
            
            # Check recommendations are generated
            self.assertGreater(len(report['recommendations']), 0)


class TestWorkflowAutomation(unittest.TestCase):
    """Test workflow automation functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        self.automation = WorkflowAutomation(Path(self.temp_dir))
    
    def tearDown(self):
        """Clean up test environment"""
        os.chdir(self.original_cwd)
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_default_rules_loaded(self):
        """Test that default automation rules are loaded"""
        rule_names = [rule.name for rule in self.automation.rules]
        
        expected_rules = [
            'python_code_changes',
            'test_file_changes',
            'config_changes',
            'documentation_changes',
            'frontend_changes',
            'critical_file_changes'
        ]
        
        for rule_name in expected_rules:
            self.assertIn(rule_name, rule_names)
    
    def test_pattern_matching(self):
        """Test file pattern matching"""
        # Test simple pattern
        self.assertTrue(self.automation.matches_pattern("test.py", "*.py"))
        self.assertFalse(self.automation.matches_pattern("test.js", "*.py"))
        
        # Test directory pattern
        self.assertTrue(self.automation.matches_pattern("tests/test_file.py", "tests/**/*.py"))
        self.assertFalse(self.automation.matches_pattern("src/file.py", "tests/**/*.py"))
    
    def test_file_type_detection(self):
        """Test file type detection"""
        test_cases = [
            ("test.py", "python"),
            ("test_file.py", "test"),
            ("config.json", "config"),
            ("README.md", "doc"),
            ("app.js", "frontend"),
            ("other.txt", "other")
        ]
        
        for file_path, expected_type in test_cases:
            result = self.automation.get_file_type(file_path)
            self.assertEqual(result, expected_type, f"Wrong type for {file_path}")
    
    def test_rule_matching(self):
        """Test finding matching rules for files"""
        # Test Python file
        rules = self.automation.find_matching_rules("src/main.py")
        rule_names = [rule.name for rule in rules]
        self.assertIn("python_code_changes", rule_names)
        
        # Test test file
        rules = self.automation.find_matching_rules("tests/test_main.py")
        rule_names = [rule.name for rule in rules]
        self.assertIn("test_file_changes", rule_names)
        
        # Test config file
        rules = self.automation.find_matching_rules("config.json")
        rule_names = [rule.name for rule in rules]
        self.assertIn("config_changes", rule_names)
    
    def test_debounce_logic(self):
        """Test debounce logic for rule execution"""
        rule = self.automation.rules[0]  # First rule
        
        # Should execute first time
        self.assertTrue(self.automation.should_execute_rule(rule))
        
        # Mark as executed
        from datetime import datetime
        self.automation.last_execution[rule.name] = datetime.now()
        
        # Should not execute immediately after
        self.assertFalse(self.automation.should_execute_rule(rule))
    
    def test_custom_rules_loading(self):
        """Test loading custom automation rules"""
        # Create custom rules file
        kiro_dir = Path(self.temp_dir) / ".kiro"
        kiro_dir.mkdir(exist_ok=True)
        
        custom_rules = {
            "rules": [
                {
                    "name": "custom_rule",
                    "trigger_patterns": ["*.custom"],
                    "workflow_context": "pre-commit",
                    "delay_seconds": 1,
                    "debounce_seconds": 3
                }
            ]
        }
        
        rules_file = kiro_dir / "automation-rules.json"
        with open(rules_file, 'w') as f:
            json.dump(custom_rules, f)
        
        # Create new automation instance
        automation = WorkflowAutomation(Path(self.temp_dir))
        
        # Check custom rule was loaded
        rule_names = [rule.name for rule in automation.rules]
        self.assertIn("custom_rule", rule_names)


class TestIDEIntegration(unittest.TestCase):
    """Test IDE integration functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        self.integration = IDEIntegration(Path(self.temp_dir))
    
    def tearDown(self):
        """Clean up test environment"""
        os.chdir(self.original_cwd)
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_syntax_checking(self):
        """Test syntax error detection"""
        # Valid Python code
        valid_code = "def hello():\n    print('Hello, World!')"
        issues = asyncio.run(self.integration.check_syntax("test.py", valid_code))
        self.assertEqual(len(issues), 0)
        
        # Invalid Python code
        invalid_code = "def hello(\n    print('Hello, World!')"
        issues = asyncio.run(self.integration.check_syntax("test.py", invalid_code))
        self.assertGreater(len(issues), 0)
        self.assertEqual(issues[0].severity, 'error')
        self.assertEqual(issues[0].rule, 'syntax_error')
    
    def test_style_checking(self):
        """Test style issue detection"""
        # Code with style issues
        code_with_issues = "def hello():    \n    x = 'very long line that exceeds the maximum line length limit and should trigger a warning'\n    return x"
        
        issues = asyncio.run(self.integration.check_style("test.py", code_with_issues))
        
        # Should find line length and trailing whitespace issues
        issue_rules = [issue.rule for issue in issues]
        self.assertIn('line_length', issue_rules)
        self.assertIn('trailing_whitespace', issue_rules)
    
    def test_complexity_checking(self):
        """Test complexity analysis"""
        # Complex function
        complex_code = '''
def complex_function():
    for i in range(10):
        if i > 5:
            for j in range(i):
                if j % 2 == 0:
                    if j > 3:
                        print(j)
                    else:
                        continue
                else:
                    break
        else:
            continue
'''
        
        issues = asyncio.run(self.integration.check_complexity("test.py", complex_code))
        
        # Should find complexity issues
        complexity_issues = [issue for issue in issues if 'complexity' in issue.rule]
        self.assertGreater(len(complexity_issues), 0)
    
    def test_metrics_calculation(self):
        """Test file metrics calculation"""
        code = '''
def function1():
    """A simple function"""
    return 1

class TestClass:
    """A test class"""
    
    def method1(self):
        return 2
'''
        
        metrics = asyncio.run(self.integration.calculate_metrics("test.py", code))
        
        self.assertIn('lines_of_code', metrics)
        self.assertIn('functions', metrics)
        self.assertIn('classes', metrics)
        self.assertEqual(metrics['functions'], 2)  # function1 + method1
        self.assertEqual(metrics['classes'], 1)
    
    def test_feedback_formatting(self):
        """Test feedback formatting for different IDE formats"""
        from unified_cli.ide_integration import RealTimeFeedback
        from datetime import datetime
        
        # Create sample feedback
        issues = [
            QualityIssue(
                file_path="test.py",
                line=1,
                column=1,
                severity="error",
                message="Test error",
                rule="test_rule",
                tool="test_tool"
            )
        ]
        
        feedback = RealTimeFeedback(
            file_path="test.py",
            issues=issues,
            metrics={},
            timestamp=datetime.now()
        )
        
        # Test LSP format
        lsp_format = self.integration.format_feedback_for_ide(feedback, 'lsp')
        self.assertIn('uri', lsp_format)
        self.assertIn('diagnostics', lsp_format)
        self.assertEqual(len(lsp_format['diagnostics']), 1)
        
        # Test VS Code format
        vscode_format = self.integration.format_feedback_for_ide(feedback, 'vscode')
        self.assertIn('file', vscode_format)
        self.assertIn('issues', vscode_format)


class TestIntegrationWorkflows(unittest.TestCase):
    """Test end-to-end integration workflows"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        # Create .kiro directory
        self.kiro_dir = Path(self.temp_dir) / ".kiro"
        self.kiro_dir.mkdir(exist_ok=True)
    
    def tearDown(self):
        """Clean up test environment"""
        os.chdir(self.original_cwd)
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_complete_workflow_integration(self):
        """Test complete workflow from setup to execution"""
        cli = UnifiedCLI()
        
        # 1. Set up team collaboration
        cli.setup_team_collaboration("Integration Test Team")
        
        # 2. Verify team config exists
        config_file = self.kiro_dir / "team-config.json"
        self.assertTrue(config_file.exists())
        
        # 3. Test workflow automation setup
        automation = WorkflowAutomation(Path(self.temp_dir))
        self.assertGreater(len(automation.rules), 0)
        
        # 4. Test IDE integration setup
        integration = IDEIntegration(Path(self.temp_dir))
        self.assertIsNotNone(integration.thresholds)
        
        # 5. Test tool execution (mocked)
        with patch.object(cli, 'run_tool') as mock_run:
            mock_run.return_value = Mock(success=True, details={})
            
            # Test quality gates
            result = cli.check_quality_gates('pre_commit')
            self.assertTrue(result)
    
    def test_configuration_persistence(self):
        """Test that configurations persist across instances"""
        # Create and configure first CLI instance
        cli1 = UnifiedCLI()
        cli1.setup_team_collaboration("Persistence Test")
        
        # Create second CLI instance
        cli2 = UnifiedCLI()
        
        # Check configuration was loaded
        self.assertIsNotNone(cli2.team_config)
        self.assertEqual(cli2.team_config.team_name, "Persistence Test")
    
    def test_error_handling(self):
        """Test error handling in various scenarios"""
        cli = UnifiedCLI()
        
        # Test invalid tool
        result = asyncio.run(cli.run_tool('nonexistent-tool'))
        self.assertFalse(result.success)
        self.assertIn("Unknown tool", result.message)
        
        # Test invalid standards file
        cli.import_standards("nonexistent_file.json")
        # Should not crash, just print error message
        
        # Test quality gates without team config
        result = cli.check_quality_gates('pre_commit')
        self.assertTrue(result)  # Should return True when no config


def run_integration_tests():
    """Run all integration tests"""
    print("Running Unified CLI Integration Tests...")
    print("=" * 50)
    
    # Create test suite
    test_classes = [
        TestUnifiedCLI,
        TestWorkflowAutomation,
        TestIDEIntegration,
        TestIntegrationWorkflows
    ]
    
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_integration_tests()
    sys.exit(0 if success else 1)
