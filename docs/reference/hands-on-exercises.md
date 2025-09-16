---
category: reference
last_updated: '2025-09-15T22:49:59.991388'
original_path: docs\training\onboarding\hands-on-exercises.md
tags:
- configuration
- api
- troubleshooting
- installation
title: Hands-on Exercises
---

# Hands-on Exercises

## Overview

These interactive exercises provide practical experience with the WAN22 project's cleanup and quality improvement tools. Complete all exercises to gain confidence using the tools in real scenarios.

## Prerequisites

- [ ] Completed [Development Environment Setup](development-setup.md)
- [ ] All tools installed and verified
- [ ] Basic understanding of the project structure

## Exercise 1: Health Check and System Validation (15 minutes)

### Objective

Learn to assess project health and identify issues using the health checker.

### Steps

1. **Run Initial Health Check**

   ```bash
   python tools/health-checker/cli.py --quick
   ```

   **Expected Output**: Summary of project health status

   **Questions**:

   - What components are healthy?
   - Are there any warnings or errors?
   - What does the overall health score indicate?

2. **Comprehensive Health Analysis**

   ```bash
   python tools/health-checker/cli.py --full --export-report health-report.json
   ```

   **Task**: Review the generated report and identify:

   - Top 3 health issues
   - Recommended actions
   - Priority levels

3. **Component-Specific Health Check**

   ```bash
   python tools/health-checker/cli.py --component tests
   python tools/health-checker/cli.py --component configuration
   python tools/health-checker/cli.py --component documentation
   ```

   **Task**: Compare component health scores and note differences.

### Verification

- [ ] Successfully ran all health check commands
- [ ] Generated and reviewed health report
- [ ] Identified key health issues
- [ ] Understood health scoring system

## Exercise 2: Test Suite Analysis and Repair (20 minutes)

### Objective

Learn to identify and fix test suite issues using the test auditor.

### Steps

1. **Audit Test Suite**

   ```bash
   python tools/test-auditor/cli.py audit --comprehensive
   ```

   **Task**: Review the audit results and identify:

   - Number of broken tests
   - Common failure patterns
   - Missing test coverage areas

2. **Fix Broken Tests**

   ```bash
   python tools/test-auditor/cli.py fix --auto --backup
   ```

   **Task**:

   - Monitor the fixing process
   - Review the backup created
   - Check which tests were automatically fixed

3. **Run Test Suite**

   ```bash
   python tools/test-runner/cli.py --all --timeout 300
   ```

   **Task**: Compare results before and after fixes:

   - Test pass rate improvement
   - Execution time changes
   - Remaining issues

4. **Analyze Test Coverage**

   ```bash
   python tools/test-quality/cli.py coverage --generate-report
   ```

   **Task**: Review coverage report and identify:

   - Overall coverage percentage
   - Uncovered code areas
   - Critical paths without tests

### Verification

- [ ] Successfully audited test suite
- [ ] Fixed broken tests with backups
- [ ] Improved test pass rate
- [ ] Generated coverage report
- [ ] Identified coverage gaps

## Exercise 3: Configuration Analysis and Consolidation (20 minutes)

### Objective

Learn to analyze and consolidate scattered configuration files.

### Steps

1. **Analyze Configuration Landscape**

   ```bash
   python tools/config-analyzer/cli.py analyze --export config-analysis.json
   ```

   **Task**: Review the analysis and identify:

   - Number of configuration files found
   - Duplicate settings across files
   - Configuration conflicts
   - Consolidation opportunities

2. **Create Unified Configuration**

   ```bash
   python tools/config-manager/cli.py unify --dry-run --preview
   ```

   **Task**: Review the unification preview:

   - Proposed unified structure
   - Migration plan
   - Potential issues

3. **Validate Configuration Consistency**

   ```bash
   python tools/config-manager/cli.py validate --all --detailed
   ```

   **Task**: Check validation results:

   - Configuration errors found
   - Inconsistencies between environments
   - Missing required settings

4. **Test Configuration Migration**

   ```bash
   python tools/config-manager/cli.py migrate --test-mode --backup
   ```

   **Task**: Verify migration process:

   - Backup creation
   - Migration steps
   - Rollback capability

### Verification

- [ ] Analyzed configuration landscape
- [ ] Identified consolidation opportunities
- [ ] Validated configuration consistency
- [ ] Tested migration process
- [ ] Understood rollback procedures

## Exercise 4: Code Quality Assessment and Improvement (25 minutes)

### Objective

Learn to assess and improve code quality using automated tools.

### Steps

1. **Run Code Quality Analysis**

   ```bash
   python tools/code-quality/cli.py analyze --comprehensive --export quality-report.json
   ```

   **Task**: Review quality report and identify:

   - Overall quality score
   - Most common violations
   - Files with lowest quality scores
   - Improvement recommendations

2. **Automated Code Formatting**

   ```bash
   python tools/code-quality/cli.py format --preview --diff
   ```

   **Task**: Review formatting changes:

   - Files to be modified
   - Types of formatting changes
   - Impact on code readability

3. **Apply Quality Improvements**

   ```bash
   python tools/code-quality/cli.py fix --auto --safe --backup
   ```

   **Task**: Monitor improvement process:

   - Changes applied automatically
   - Manual fixes required
   - Quality score improvement

4. **Validate Documentation**

   ```bash
   python tools/code-quality/cli.py validate-docs --generate-missing
   ```

   **Task**: Check documentation status:

   - Functions without documentation
   - Generated documentation templates
   - Documentation quality score

5. **Check Type Hints**

   ```bash
   python tools/code-quality/cli.py validate-types --add-missing
   ```

   **Task**: Review type hint coverage:

   - Functions missing type hints
   - Automatically added hints
   - Type checking results

### Verification

- [ ] Analyzed code quality comprehensively
- [ ] Applied automated formatting
- [ ] Fixed quality violations
- [ ] Improved documentation coverage
- [ ] Enhanced type hint coverage

## Exercise 5: Documentation Generation and Validation (15 minutes)

### Objective

Learn to generate and maintain project documentation automatically.

### Steps

1. **Generate Project Documentation**

   ```bash
   python tools/doc-generator/cli.py generate --all --include-diagrams
   ```

   **Task**: Review generated documentation:

   - Project structure documentation
   - Component relationship diagrams
   - API documentation
   - Configuration documentation

2. **Validate Documentation Links**

   ```bash
   python tools/doc-generator/cli.py validate-links --fix-broken
   ```

   **Task**: Check link validation results:

   - Broken links found
   - Automatically fixed links
   - External link status

3. **Update Documentation Index**

   ```bash
   python tools/doc-generator/cli.py update-index --create-search
   ```

   **Task**: Verify documentation index:

   - Navigation structure
   - Search functionality
   - Cross-references

4. **Check Documentation Freshness**

   ```bash
   python tools/doc-generator/cli.py check-freshness --update-stale
   ```

   **Task**: Review freshness report:

   - Outdated documentation
   - Recently updated content
   - Maintenance recommendations

### Verification

- [ ] Generated comprehensive documentation
- [ ] Validated and fixed broken links
- [ ] Created searchable documentation index
- [ ] Identified and updated stale content

## Exercise 6: Codebase Cleanup and Organization (20 minutes)

### Objective

Learn to clean up and organize the codebase using automated tools.

### Steps

1. **Detect Duplicate Files**

   ```bash
   python tools/codebase-cleanup/cli.py detect-duplicates --similarity 0.8
   ```

   **Task**: Review duplicate detection results:

   - Exact duplicates found
   - Similar files identified
   - Consolidation recommendations

2. **Analyze Dead Code**

   ```bash
   python tools/codebase-cleanup/cli.py analyze-dead-code --include-imports
   ```

   **Task**: Check dead code analysis:

   - Unused functions and classes
   - Unused imports
   - Unreachable code paths

3. **Standardize Naming Conventions**

   ```bash
   python tools/codebase-cleanup/cli.py standardize-naming --preview
   ```

   **Task**: Review naming standardization:

   - Inconsistent naming patterns
   - Proposed naming changes
   - Impact on codebase

4. **Safe Cleanup Operations**

   ```bash
   python tools/codebase-cleanup/cli.py cleanup --safe --backup --dry-run
   ```

   **Task**: Review cleanup plan:

   - Files to be removed
   - Refactoring operations
   - Safety measures in place

### Verification

- [ ] Detected and analyzed duplicates
- [ ] Identified dead code
- [ ] Reviewed naming standardization
- [ ] Planned safe cleanup operations

## Exercise 7: Unified CLI and Workflow Integration (15 minutes)

### Objective

Learn to use the unified CLI for integrated workflow management.

### Steps

1. **Explore Unified CLI**

   ```bash
   python tools/unified-cli/cli.py --help
   python tools/unified-cli/cli.py status
   python tools/unified-cli/cli.py metrics
   ```

   **Task**: Familiarize yourself with:

   - Available commands
   - Current system status
   - Quality metrics dashboard

2. **Run Integrated Workflow**

   ```bash
   python tools/unified-cli/cli.py workflow daily-check
   ```

   **Task**: Observe the integrated workflow:

   - Tools executed in sequence
   - Results aggregation
   - Summary report

3. **Configure Personal Preferences**

   ```bash
   python tools/unified-cli/cli.py config set user.name "Your Name"
   python tools/unified-cli/cli.py config set workflow.auto_fix true
   python tools/unified-cli/cli.py config set notifications.email "your@email.com"
   ```

   **Task**: Customize your environment:

   - Personal settings
   - Workflow preferences
   - Notification configuration

4. **Test Pre-commit Integration**

   ```bash
   python tools/unified-cli/cli.py install-hooks
   # Make a small change to a file
   git add .
   git commit -m "Test commit for pre-commit hooks"
   ```

   **Task**: Verify pre-commit integration:

   - Hooks installation
   - Automatic quality checks
   - Commit process enhancement

### Verification

- [ ] Explored unified CLI capabilities
- [ ] Ran integrated workflows
- [ ] Configured personal preferences
- [ ] Tested pre-commit integration

## Exercise 8: Troubleshooting and Recovery (10 minutes)

### Objective

Learn to troubleshoot issues and recover from problems.

### Steps

1. **Use Interactive Troubleshooting**

   ```bash
   python tools/unified-cli/cli.py troubleshoot
   ```

   **Task**: Follow the interactive troubleshooting wizard:

   - Describe a hypothetical issue
   - Follow diagnostic steps
   - Review suggested solutions

2. **Create Support Package**

   ```bash
   python tools/unified-cli/cli.py create-support-package --include-logs
   ```

   **Task**: Review support package contents:

   - System information
   - Configuration files
   - Recent logs
   - Error reports

3. **Test Rollback Functionality**

   ```bash
   python tools/unified-cli/cli.py rollback --list
   python tools/unified-cli/cli.py rollback --preview --last
   ```

   **Task**: Understand rollback capabilities:

   - Available rollback points
   - Rollback preview
   - Recovery procedures

### Verification

- [ ] Used interactive troubleshooting
- [ ] Created comprehensive support package
- [ ] Understood rollback procedures

## Final Integration Exercise (15 minutes)

### Objective

Integrate all learned skills in a comprehensive workflow.

### Steps

1. **Complete Daily Workflow**

   ```bash
   # Morning routine
   python tools/unified-cli/cli.py workflow morning-check

   # Development workflow
   python tools/unified-cli/cli.py workflow dev-ready

   # Pre-commit workflow
   python tools/unified-cli/cli.py workflow pre-commit

   # End-of-day workflow
   python tools/unified-cli/cli.py workflow end-of-day
   ```

2. **Generate Comprehensive Report**

   ```bash
   python tools/unified-cli/cli.py generate-report --comprehensive --export
   ```

3. **Plan Maintenance Activities**
   ```bash
   python tools/maintenance-scheduler/cli.py plan-weekly-maintenance
   ```

### Verification

- [ ] Completed full daily workflow
- [ ] Generated comprehensive report
- [ ] Planned maintenance activities

## Exercise Completion

### Self-Assessment Checklist

#### Knowledge Check

- [ ] I understand the purpose of each tool category
- [ ] I can run basic health checks and interpret results
- [ ] I know how to fix common test issues
- [ ] I can analyze and consolidate configurations
- [ ] I understand code quality assessment and improvement
- [ ] I can generate and maintain documentation
- [ ] I know how to clean up and organize code
- [ ] I can use the unified CLI effectively
- [ ] I understand troubleshooting and recovery procedures

#### Practical Skills

- [ ] I can identify project health issues
- [ ] I can fix broken tests automatically
- [ ] I can consolidate scattered configurations
- [ ] I can improve code quality systematically
- [ ] I can generate comprehensive documentation
- [ ] I can clean up duplicate and dead code
- [ ] I can integrate tools into my workflow
- [ ] I can troubleshoot and recover from issues

#### Confidence Level

Rate your confidence (1-5) in using each tool category:

- Health Checker: \_\_\_/5
- Test Management: \_\_\_/5
- Configuration Management: \_\_\_/5
- Code Quality: \_\_\_/5
- Documentation: \_\_\_/5
- Codebase Cleanup: \_\_\_/5
- Unified CLI: \_\_\_/5
- Troubleshooting: \_\_\_/5

### Next Steps

1. **Practice Regularly**: Use tools in your daily development workflow
2. **Explore Advanced Features**: Dive deeper into tool customization
3. **Share Knowledge**: Help other team members learn the tools
4. **Provide Feedback**: Suggest improvements to tools and documentation
5. **Stay Updated**: Keep up with tool updates and new features

### Getting Help

If you encountered issues during exercises:

- **Review Documentation**: Check tool-specific documentation
- **Check FAQ**: Visit [Frequently Asked Questions](../troubleshooting/faq.md)
- **Use Troubleshooting Guide**: Follow [Troubleshooting Guide](../troubleshooting/README.md)
- **Ask for Help**: Contact team members or support

### Feedback

Please provide feedback on these exercises:

- Which exercises were most helpful?
- What was confusing or unclear?
- What additional exercises would be valuable?
- How can we improve the learning experience?

**Congratulations!** You've completed the hands-on exercises and are ready to use the WAN22 project's cleanup and quality improvement tools effectively.
