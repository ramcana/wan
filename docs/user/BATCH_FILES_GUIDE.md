---
category: user
last_updated: '2025-09-15T22:50:00.279609'
original_path: local_installation\BATCH_FILES_GUIDE.md
tags:
- configuration
- troubleshooting
- installation
title: WAN2.2 Batch Files Usage Guide
---

# WAN2.2 Batch Files Usage Guide

This guide explains which batch files to use for different purposes in the WAN2.2 local installation system.

## 📋 Quick Reference

| Batch File            | Purpose                   | When to Use                               |
| --------------------- | ------------------------- | ----------------------------------------- |
| `install.bat`         | **Main Installation**     | First time setup - installs everything    |
| `launch_wan22.bat`    | **Start Application**     | Daily use - launches WAN2.2 application   |
| `launch_web_ui.bat`   | **Start Web Interface**   | Daily use - launches web-based UI         |
| `run_first_setup.bat` | **Initial Configuration** | After installation - configure settings   |
| `manage.bat`          | **System Management**     | Maintenance - update, repair, uninstall   |
| `prepare_release.bat` | **Package Creation**      | Development - create distribution package |
| `run_tests.bat`       | **Testing**               | Development - run validation tests        |

---

## 🚀 For End Users (Most Common)

### 1. **`install.bat`** - Main Installation

**Use this FIRST** - This is your starting point!

```bash
# Basic installation
install.bat

# Silent installation (no prompts)
install.bat --silent

# Installation without downloading models (faster)
install.bat --skip-models

# Test installation without making changes
install.bat --dry-run
```

**What it does:**

- Detects your hardware automatically
- Installs Python and dependencies
- Downloads WAN2.2 models
- Creates optimized configuration
- Sets up desktop shortcuts
- Validates everything works

### 2. **`launch_wan22.bat`** - Start WAN2.2 Application

**Use this DAILY** - After installation, use this to run WAN2.2

```bash
# Start WAN2.2 application
launch_wan22.bat
```

**What it does:**

- Activates the virtual environment
- Starts the WAN2.2 application
- Opens the main interface

### 3. **`launch_web_ui.bat`** - Start Web Interface

**Alternative UI** - Use this for web-based interface

```bash
# Start web interface
launch_web_ui.bat
```

**What it does:**

- Starts the web server
- Opens browser to WAN2.2 web interface
- Provides web-based controls

### 4. **`run_first_setup.bat`** - Initial Configuration

**Use AFTER installation** - Configure your preferences

```bash
# Run first-time setup wizard
run_first_setup.bat
```

**What it does:**

- Guides through initial configuration
- Sets up user preferences
- Configures model settings
- Tests basic functionality

---

## 🔧 For System Management

### 5. **`manage.bat`** - System Management

**Use for maintenance** - Update, repair, or uninstall

```bash
# Open management menu
manage.bat

# Available options:
# - Update WAN2.2
# - Repair installation
# - Change configuration
# - Uninstall
# - View logs
# - Run diagnostics
```

**What it does:**

- Provides management interface
- Handles updates and repairs
- Manages configuration changes
- Provides diagnostic tools

---

## 🛠️ For Developers

### 6. **`prepare_release.bat`** - Package Creation

**Development only** - Create distribution packages

```bash
# Create release package
prepare_release.bat
```

**What it does:**

- Creates WAN22-Installation-Package
- Packages all necessary files
- Prepares for distribution

### 7. **`run_tests.bat`** - Testing

**Development only** - Run validation tests

```bash
# Run all tests
run_tests.bat

# Run specific test
run_tests.bat --test comprehensive
```

**What it does:**

- Runs comprehensive validation
- Tests all installation components
- Validates hardware compatibility

---

## 📍 Typical Usage Flow

### First Time Setup

```
1. install.bat                 # Install everything
2. run_first_setup.bat        # Configure preferences
3. launch_wan22.bat           # Start using WAN2.2
```

### Daily Usage

```
launch_wan22.bat              # Just start the application
# OR
launch_web_ui.bat            # Use web interface
```

### Maintenance

```
manage.bat                    # For updates, repairs, etc.
```

---

## 🎯 Which File Should I Use?

### **I'm a new user and want to install WAN2.2:**

→ Use `install.bat`

### **I've already installed and want to use WAN2.2:**

→ Use `launch_wan22.bat` or `launch_web_ui.bat`

### **I just finished installing and want to configure settings:**

→ Use `run_first_setup.bat`

### **Something's not working and I need to fix it:**

→ Use `manage.bat`

### **I'm a developer creating a release:**

→ Use `prepare_release.bat`

### **I'm testing the installation system:**

→ Use `run_tests.bat`

---

## 📂 File Locations

### Main Directory (`local_installation/`)

- `install.bat` - Main installer
- `launch_wan22.bat` - Application launcher
- `launch_web_ui.bat` - Web UI launcher
- `run_first_setup.bat` - Setup wizard
- `manage.bat` - Management tools
- `prepare_release.bat` - Release preparation
- `run_tests.bat` - Test runner

### Installation Package (`WAN22-Installation-Package/`)

- `install.bat` - Main installer (copy)
- `launch_wan22.bat` - Application launcher (copy)
- `launch_web_ui.bat` - Web UI launcher (copy)
- `run_first_setup.bat` - Setup wizard (copy)
- `manage.bat` - Management tools (copy)

### Scripts Directory (`scripts/`)

- `install_python.bat` - Python installer (internal use)

---

## 🚨 Important Notes

### **Always Start With `install.bat`**

This is the main entry point. Don't use other files until you've run the installation.

### **Use Full Paths or Navigate to Directory**

```bash
# Good - navigate to directory first
cd local_installation
install.bat

# Good - use full path
E:\wan\local_installation\install.bat

# Bad - might not work
install.bat
```

### **Check for Administrator Rights**

Some operations may require administrator privileges. Run Command Prompt as Administrator if needed.

### **Read the Output**

The batch files provide helpful messages. Read them to understand what's happening.

---

## 🆘 Troubleshooting

### **"Command not found" error:**

- Make sure you're in the right directory
- Use `.\install.bat` in PowerShell
- Try running Command Prompt as Administrator

### **Installation fails:**

- Check `logs/installation.log` for details
- Try `install.bat --verbose` for more information
- Use `manage.bat` to run diagnostics

### **Application won't start:**

- Try `manage.bat` to repair installation
- Check if installation completed successfully
- Verify all dependencies are installed

---

_This guide covers all batch files in the WAN2.2 local installation system. For more detailed information, see the README.md and GETTING_STARTED.md files._
