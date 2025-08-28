# WAN2.2 Batch Files - Decision Flowchart

## 🎯 Quick Decision Guide

```
START HERE
    ↓
Have you installed WAN2.2 before?
    ↓
   NO ────────────────────────────→ Use: install.bat
    ↓                                      ↓
   YES                              Installation Complete?
    ↓                                      ↓
What do you want to do?                   YES ──→ Use: run_first_setup.bat
    ↓                                      ↓
┌─────────────────────────────────────────────────────────────┐
│  A) Use WAN2.2 normally                                     │
│  B) Configure settings                                      │
│  C) Fix problems                                           │
│  D) Update/Manage system                                   │
│  E) Create release package (developers)                    │
│  F) Run tests (developers)                                 │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─── A) Daily Usage ───┐
│                      │
│  Desktop App?        │
│  YES → launch_wan22.bat
│                      │
│  Web Interface?      │
│  YES → launch_web_ui.bat
│                      │
└──────────────────────┘

┌─── B) Configuration ───┐
│                         │
│  First time setup?      │
│  YES → run_first_setup.bat
│                         │
│  Change settings?       │
│  YES → manage.bat       │
│                         │
└─────────────────────────┘

┌─── C) Fix Problems ───┐
│                       │
│  Any issues?          │
│  YES → manage.bat     │
│                       │
└───────────────────────┘

┌─── D) System Management ───┐
│                             │
│  Update/Repair/Uninstall?   │
│  YES → manage.bat           │
│                             │
└─────────────────────────────┘

┌─── E) Development ───┐
│                      │
│  Create package?     │
│  YES → prepare_release.bat
│                      │
└──────────────────────┘

┌─── F) Testing ───┐
│                  │
│  Run tests?      │
│  YES → run_tests.bat
│                  │
└──────────────────┘
```

## 🚀 Most Common Usage Patterns

### **New User (First Time)**

```
1. install.bat           ← Start here!
2. run_first_setup.bat   ← Configure after install
3. launch_wan22.bat      ← Start using WAN2.2
```

### **Regular User (Daily)**

```
launch_wan22.bat         ← Just start the app
```

### **Power User (Maintenance)**

```
manage.bat               ← For updates, repairs, config changes
```

### **Developer**

```
prepare_release.bat      ← Create distribution package
run_tests.bat           ← Validate installation system
```

## 🎨 Color-Coded Priority

🟢 **GREEN (Essential)** - Everyone needs these

- `install.bat` - Main installation
- `launch_wan22.bat` - Start application

🟡 **YELLOW (Important)** - Most users will need these

- `run_first_setup.bat` - Initial configuration
- `launch_web_ui.bat` - Alternative interface
- `manage.bat` - System management

🔵 **BLUE (Optional)** - Developers and advanced users

- `prepare_release.bat` - Package creation
- `run_tests.bat` - Testing and validation

## 📱 Mobile-Friendly Quick Reference

**Just installed?** → `install.bat`  
**Want to use WAN2.2?** → `launch_wan22.bat`  
**Need to configure?** → `run_first_setup.bat`  
**Having problems?** → `manage.bat`  
**Prefer web interface?** → `launch_web_ui.bat`

## ⚡ One-Liner Explanations

- **install.bat** = "Install everything"
- **launch_wan22.bat** = "Start WAN2.2"
- **launch_web_ui.bat** = "Start web version"
- **run_first_setup.bat** = "Configure settings"
- **manage.bat** = "Fix/update/manage"
- **prepare_release.bat** = "Create package (dev)"
- **run_tests.bat** = "Run tests (dev)"

---

_Need more details? See BATCH_FILES_GUIDE.md for comprehensive information._
