# Common Workflows

## Overview

This guide documents the most common workflows for using the WAN22 project's cleanup and quality improvement tools in daily development, maintenance, and collaboration scenarios.

## Daily Development Workflows

### 🌅 Morning Startup Routine (5 minutes)

**Purpose**: Start your day with a clean, healthy development environment.

**Steps**:

1. **Update Repository**:

   ```bash
   git pull origin main
   ```

2. **Quick Health Check**:

   ```bash
   python tools/health-checker/cli.py --quick
   ```

3. **Check for Tool Updates**:

   ```bash
   python tools/unified-cli/cli.py check-updates
   ```

4. **Review Overnight Reports**:

   ```bash
   python tools/maintenance-reporter/cli.py recent-reports --since yesterday
   ```

5. **Validate Environment**:
   ```bash
   python tools/unified-cli/cli.py validate-environment
   ```

**Expected Outcome**: Clean development environment ready for productive work.

---

### 💻 Active Development Workflow

**Purpose**: Maintain quality while developing new features or fixing bugs.

#### Starting New Work

```bash
# 1. Create feature branch
git checkout -b feature/new-feature

# 2. Run baseline health check
python tools/health-checker/cli.py --component-specific --export baseline.json

# 3. Start real-time quality monitoring
python tools/code-quality/cli.py watch --auto-fix &
```

#### During Development

```bash
# Run tests for changed files
python tools/test-runner/cli.py --changed-files --fast

# Check code quality incrementally
python tools/code-quality/cli.py check --changed-files

# Generate documentation for new code
python tools/doc-generator/cli.py --incremental --new-files
```

#### Before Committing

```bash
# Run comprehensive pre-commit checks
python tools/unified-cli/cli.py pre-commit

# Ensure adequate test coverage
python tools/test-quality/cli.py coverage --minimum 80 --changed-files

# Update documentation
python tools/doc-generator/cli.py --update-changed

# Validate commit message
python tools/unified-cli/cli.py validate-commit-message
```

---

### 🌙 End-of-Day Routine (5 minutes)

**Purpose**: Clean up work environment and prepare for next day.

**Steps**:

1. **Comprehensive Health Check**:

   ```bash
   python tools/health-checker/cli.py --full --export daily-report.json
   ```

2. **Clean Temporary Files**:

   ```bash
   python tools/codebase-cleanup/cli.py --temp-files --safe
   ```

3. **Update Personal Metrics**:

   ```bash
   python tools/quality-monitor/cli.py update-personal-metrics
   ```

4. **Schedule Maintenance if Needed**:

   ```bash
   python tools/maintenance-scheduler/cli.py schedule-if-needed
   ```

5. **Backup Important Work**:
   ```bash
   python tools/unified-cli/cli.py backup --incremental
   ```

---

## Code Review Workflows

### 📝 Preparing Code for Review

**Purpose**: Ensure code meets quality standards before review.

**Steps**:

1. **Run Full Quality Check**:

   ```bash
   python tools/code-quality/cli.py analyze --comprehensive --export review-report.json
   ```

2. **Generate Review Checklist**:

   ```bash
   python tools/code-review/cli.py generate-checklist --pr-branch feature/new-feature
   ```

3. **Run Security Analysis**:

   ```bash
   python tools/code-quality/cli.py security-check --detailed
   ```

4. **Validate Test Coverage**:

   ```bash
   python tools/test-quality/cli.py coverage --detailed --export coverage-report.html
   ```

5. **Generate Documentation**:
   ```bash
   python tools/doc-generator/cli.py --pr-documentation --include-examples
   ```

---

### 👀 Reviewing Code

**Purpose**: Systematically review code changes using automated assistance.

**Steps**:

1. **Get Automated Review**:

   ```bash
   python tools/code-review/cli.py review --pr-number 123 --detailed
   ```

2. **Check Quality Metrics**:

   ```bash
   python tools/code-review/cli.py metrics --pr-number 123 --compare-baseline
   ```

3. **Validate Tests**:

   ```bash
   python tools/test-auditor/cli.py validate-pr-tests --pr-number 123
   ```

4. **Review Documentation**:

   ```bash
   python tools/doc-generator/cli.py validate-pr-docs --pr-number 123
   ```

5. **Generate Review Summary**:
   ```bash
   python tools/code-review/cli.py summary --pr-number 123 --export review-summary.md
   ```

---

## Maintenance Workflows

### 🔧 Weekly Maintenance (30 minutes)

**Purpose**: Perform regular maintenance to keep the project healthy.

**Schedule**: Every Monday morning or Friday afternoon.

**Steps**:

1. **Comprehensive Health Analysis**:

   ```bash
   python tools/health-checker/cli.py --comprehensive --export weekly-health.json
   ```

2. **Test Suite Maintenance**:

   ```bash
   python tools/test-auditor/cli.py audit --comprehensive --fix-safe-issues
   ```

3. **Code Quality Review**:

   ```bash
   python tools/code-quality/cli.py analyze --full --generate-report
   ```

4. **Configuration Validation**:

   ```bash
   python tools/config-manager/cli.py validate-all --check-drift
   ```

5. **Documentation Updates**:

   ```bash
   python tools/doc-generator/cli.py update-all --validate-links
   ```

6. **Cleanup Operations**:

   ```bash
   python tools/codebase-cleanup/cli.py safe-cleanup --backup
   ```

7. **Performance Analysis**:
   ```bash
   python tools/quality-monitor/cli.py performance-report --weekly
   ```

---

### 🚀 Release Preparation (60 minutes)

**Purpose**: Ensure code is ready for production deployment.

**Steps**:

1. **Pre-release Health Check**:

   ```bash
   python tools/health-checker/cli.py --release-ready --strict
   ```

2. **Comprehensive Test Validation**:

   ```bash
   python tools/test-runner/cli.py --all --timeout 600 --strict
   ```

3. **Security Audit**:

   ```bash
   python tools/code-quality/cli.py security-audit --comprehensive
   ```

4. **Performance Benchmarking**:

   ```bash
   python tools/quality-monitor/cli.py benchmark --compare-baseline
   ```

5. **Documentation Finalization**:

   ```bash
   python tools/doc-generator/cli.py finalize-release --version v2.2.0
   ```

6. **Configuration Validation**:

   ```bash
   python tools/config-manager/cli.py validate-production --strict
   ```

7. **Generate Release Report**:
   ```bash
   python tools/unified-cli/cli.py generate-release-report --version v2.2.0
   ```

---

## Troubleshooting Workflows

### 🔍 Issue Investigation

**Purpose**: Systematically investigate and resolve issues.

**Steps**:

1. **Initial Diagnosis**:

   ```bash
   python tools/unified-cli/cli.py diagnose --issue-type [test-failure|config-error|performance]
   ```

2. **Collect System Information**:

   ```bash
   python tools/unified-cli/cli.py system-info --detailed
   ```

3. **Analyze Recent Changes**:

   ```bash
   python tools/unified-cli/cli.py analyze-changes --since [date|commit]
   ```

4. **Run Interactive Troubleshooting**:

   ```bash
   python tools/unified-cli/cli.py troubleshoot --interactive
   ```

5. **Generate Support Package**:
   ```bash
   python tools/unified-cli/cli.py create-support-package --include-logs
   ```

---

### 🔄 Recovery Procedures

**Purpose**: Recover from issues using rollback and restoration.

**Steps**:

1. **Assess Damage**:

   ```bash
   python tools/health-checker/cli.py --emergency-check
   ```

2. **Check Rollback Options**:

   ```bash
   python tools/unified-cli/cli.py rollback --list --detailed
   ```

3. **Perform Safe Rollback**:

   ```bash
   python tools/unified-cli/cli.py rollback --to-date [date] --preview
   python tools/unified-cli/cli.py rollback --to-date [date] --confirm
   ```

4. **Validate Recovery**:

   ```bash
   python tools/health-checker/cli.py --post-recovery-check
   ```

5. **Document Incident**:
   ```bash
   python tools/unified-cli/cli.py document-incident --recovery-steps
   ```

---

## Team Collaboration Workflows

### 👥 Team Onboarding

**Purpose**: Efficiently onboard new team members.

**Steps**:

1. **Initial Setup**:

   ```bash
   python tools/training-system/cli.py start-onboarding --role developer
   ```

2. **Environment Validation**:

   ```bash
   python tools/unified-cli/cli.py validate-setup --new-user
   ```

3. **Guided Tutorial**:

   ```bash
   python tools/training-system/cli.py tutorial test-management --interactive
   ```

4. **Practice Exercises**:

   ```bash
   python tools/training-system/cli.py practice health-check
   ```

5. **Knowledge Assessment**:
   ```bash
   python tools/training-system/cli.py assessment all
   ```

---

### 📊 Team Quality Review

**Purpose**: Regular team review of project quality and metrics.

**Frequency**: Bi-weekly team meetings.

**Steps**:

1. **Generate Team Metrics**:

   ```bash
   python tools/quality-monitor/cli.py team-metrics --period 2weeks
   ```

2. **Identify Trends**:

   ```bash
   python tools/quality-monitor/cli.py trend-analysis --detailed
   ```

3. **Review Problem Areas**:

   ```bash
   python tools/health-checker/cli.py problem-areas --team-view
   ```

4. **Plan Improvements**:

   ```bash
   python tools/quality-monitor/cli.py improvement-plan --team-input
   ```

5. **Set Quality Goals**:
   ```bash
   python tools/quality-monitor/cli.py set-goals --team-consensus
   ```

---

## CI/CD Integration Workflows

### 🔄 Continuous Integration

**Purpose**: Automated quality checks in CI/CD pipeline.

**Pipeline Configuration**:

```yaml
# .github/workflows/quality.yml
name: Quality Checks
on: [push, pull_request]

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Setup Environment
        run: python tools/unified-cli/cli.py setup-ci

      - name: Health Check
        run: python tools/health-checker/cli.py --ci-mode

      - name: Test Suite
        run: python tools/test-runner/cli.py --ci --timeout 300

      - name: Code Quality
        run: python tools/code-quality/cli.py --ci --strict

      - name: Generate Reports
        run: python tools/unified-cli/cli.py ci-report --export-artifacts
```

---

### 🚀 Continuous Deployment

**Purpose**: Automated deployment with quality gates.

**Deployment Steps**:

1. **Pre-deployment Validation**:

   ```bash
   python tools/unified-cli/cli.py pre-deploy-check --environment production
   ```

2. **Configuration Validation**:

   ```bash
   python tools/config-manager/cli.py validate-deployment --environment production
   ```

3. **Performance Baseline**:

   ```bash
   python tools/quality-monitor/cli.py capture-baseline --pre-deployment
   ```

4. **Deploy with Monitoring**:

   ```bash
   python tools/unified-cli/cli.py deploy --monitor --rollback-on-failure
   ```

5. **Post-deployment Validation**:
   ```bash
   python tools/health-checker/cli.py --post-deployment --strict
   ```

---

## Workflow Customization

### 🎛️ Personal Workflow Configuration

**Purpose**: Customize workflows for individual preferences.

**Configuration File**: `~/.wan22/workflow-config.yaml`

```yaml
workflows:
  daily_startup:
    enabled: true
    quick_mode: true
    auto_fix: true

  development:
    real_time_quality: true
    auto_test_on_save: true
    documentation_updates: true

  pre_commit:
    strict_mode: false
    auto_fix_safe: true
    require_tests: true
```

**Apply Configuration**:

```bash
python tools/unified-cli/cli.py configure-workflows --personal
```

---

### 🏢 Team Workflow Standards

**Purpose**: Establish consistent team workflows.

**Team Configuration**: `config/team-workflows.yaml`

```yaml
team_standards:
  code_review:
    required_checks: [quality, tests, security, documentation]
    auto_merge_threshold: 95

  maintenance:
    weekly_schedule: "Monday 09:00"
    monthly_deep_clean: "First Friday 14:00"

  quality_gates:
    minimum_coverage: 80
    maximum_complexity: 10
    required_documentation: true
```

**Apply Team Standards**:

```bash
python tools/unified-cli/cli.py apply-team-standards --enforce
```

---

## Workflow Monitoring

### 📈 Workflow Analytics

**Purpose**: Monitor and optimize workflow effectiveness.

**Metrics Tracked**:

- Workflow execution times
- Success/failure rates
- Developer productivity impact
- Quality improvement trends

**Generate Analytics**:

```bash
python tools/quality-monitor/cli.py workflow-analytics --period month
```

---

### 🔧 Workflow Optimization

**Purpose**: Continuously improve workflow efficiency.

**Optimization Steps**:

1. **Analyze Bottlenecks**:

   ```bash
   python tools/quality-monitor/cli.py analyze-bottlenecks --workflows
   ```

2. **Suggest Improvements**:

   ```bash
   python tools/unified-cli/cli.py optimize-workflows --suggestions
   ```

3. **A/B Test Changes**:

   ```bash
   python tools/unified-cli/cli.py test-workflow-changes --experimental
   ```

4. **Apply Optimizations**:
   ```bash
   python tools/unified-cli/cli.py apply-optimizations --validated
   ```

---

## Getting Help

### 📚 Workflow Documentation

- [Tool-specific workflows](../tools/README.md)
- [Best practices](../best-practices/README.md)
- [Troubleshooting guide](../troubleshooting/README.md)

### 🎓 Training Resources

- [Interactive tutorials](../video-tutorials/README.md)
- [Hands-on exercises](../onboarding/hands-on-exercises.md)
- [Team training sessions](../onboarding/team-onboarding-guide.md)

### 💬 Support Channels

- [FAQ](../troubleshooting/faq.md)
- [Community resources](../troubleshooting/community-resources.md)
- [Direct support](../troubleshooting/getting-support.md)

---

**Remember**: Workflows should serve you, not the other way around. Customize them to fit your team's needs and continuously improve based on experience and feedback.
