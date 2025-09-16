---
title: Documentation Style Guide
category: developer-guide
tags: [documentation, style, guidelines, writing]
last_updated: 2024-01-01
status: published
---

# Documentation Style Guide

This guide ensures consistency across all WAN22 project documentation.

## General Principles

### Writing Style

- **Clear and Concise**: Use simple, direct language
- **Active Voice**: Prefer active over passive voice
- **Present Tense**: Use present tense for instructions
- **Audience-Focused**: Write for your specific audience (users, developers, admins)

### Tone

- **Professional but Friendly**: Maintain a helpful, approachable tone
- **Confident**: Use definitive language ("Click Save" not "You might want to click Save")
- **Inclusive**: Use inclusive language and avoid assumptions

## Document Structure

### Required Frontmatter

All documentation files must include YAML frontmatter:

```yaml
---
title: Page Title
category: user-guide|developer-guide|deployment|api|reference
tags: [tag1, tag2, tag3]
last_updated: YYYY-MM-DD
author: Author Name (optional)
status: draft|review|approved|published
---
```

### Heading Hierarchy

Use proper heading hierarchy:

```markdown
# Page Title (H1) - Only one per page

## Main Sections (H2)

### Subsections (H3)

#### Sub-subsections (H4) - Use sparingly
```

### Required Sections

Most pages should include:

1. **Overview**: Brief description of the topic
2. **Prerequisites**: What users need before starting
3. **Main Content**: The core information
4. **Examples**: Practical examples where applicable
5. **Troubleshooting**: Common issues and solutions
6. **Related Documentation**: Links to related pages

## Formatting Guidelines

### Code Blocks

Always specify the language for syntax highlighting:

````markdown
```python
def example_function():
    return "Hello World"
```
````

````

```markdown
```bash
# Shell commands
python main.py --help
````

````

```markdown
```yaml
# Configuration files
key: value
nested:
  key: value
````

````

### Inline Code

Use backticks for inline code, commands, filenames, and configuration values:

- Commands: `python main.py`
- Files: `config.yaml`
- Values: `debug=true`
- Code: `function_name()`

### Lists

**Ordered Lists** for sequential steps:

1. First step
2. Second step
3. Third step

**Unordered Lists** for non-sequential items:

- Item one
- Item two
- Item three

### Tables

Use tables for structured data:

| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Value 1  | Value 2  | Value 3  |
| Value 4  | Value 5  | Value 6  |

### Links

**Internal Links** (relative paths):
```markdown
[Configuration Guide](../user-guide/configuration.md)
[API Reference](../api/index.md)
````

**External Links**:

```markdown
[Python Documentation](https://docs.python.org/)
```

**Anchor Links** within the same page:

```markdown
[Jump to Section](#section-name)
```

### Images

Include alt text for accessibility:

```markdown
![Alt text describing the image](../images/screenshot.png)
```

Store images in `docs/images/` directory.

### Admonitions

Use consistent formatting for callouts:

**Note**:

```markdown
> **Note**: Additional information that's helpful but not critical.
```

**Warning**:

```markdown
> **⚠️ Warning**: Important information that could prevent errors.
```

**Tip**:

```markdown
> **💡 Tip**: Helpful suggestions or best practices.
```

**Important**:

```markdown
> **❗ Important**: Critical information that must not be missed.
```

## Content Guidelines

### Instructions

Write clear, actionable instructions:

**Good**:

```markdown
1. Open the configuration file: `config/config.yaml`
2. Set the `debug` parameter to `true`
3. Save the file
4. Restart the application
```

**Avoid**:

```markdown
You should probably open the config file and maybe change the debug setting.
```

### Code Examples

Provide complete, working examples:

**Good**:

```python
# Complete example
from config import Config

config = Config()
config.load('config.yaml')
print(f"Debug mode: {config.debug}")
```

**Avoid**:

```python
# Incomplete example
config.debug = True
```

### Error Messages

Include exact error messages and solutions:

````markdown
**Error**: `ModuleNotFoundError: No module named 'torch'`

**Solution**: Install PyTorch:

```bash
pip install torch torchvision
```
````

````

### File Paths

Use forward slashes for cross-platform compatibility:

- ✅ `config/settings.yaml`
- ❌ `config\settings.yaml`

Use relative paths from project root:

- ✅ `backend/app.py`
- ❌ `/home/user/project/backend/app.py`

## Language and Grammar

### Capitalization

- **Sentence case** for headings: "Getting started with WAN22"
- **Title case** for proper nouns: "Python", "Docker", "WAN22"
- **Lowercase** for commands and filenames: `python`, `config.yaml`

### Punctuation

- Use serial commas: "red, white, and blue"
- No periods in headings
- Use periods in complete sentences
- Use colons to introduce lists or code blocks

### Common Terms

Use consistent terminology:

- **WAN22** (not wan22 or Wan22)
- **API endpoint** (not API end point)
- **Configuration file** (not config file in formal docs)
- **Command line** (not command-line as noun)
- **Set up** (verb) vs **Setup** (noun)

## Accessibility

### Alt Text

Provide descriptive alt text for images:

```markdown
![Screenshot showing the configuration dialog with debug mode enabled](../images/config-debug.png)
````

### Link Text

Use descriptive link text:

**Good**: [Download the installation guide](installation.md)
**Avoid**: [Click here](installation.md) for the installation guide

### Color

Don't rely solely on color to convey information. Use text labels or symbols.

## Review Process

### Self-Review Checklist

Before submitting documentation:

- [ ] Frontmatter is complete and correct
- [ ] Headings follow proper hierarchy
- [ ] Code blocks specify language
- [ ] Links work and use relative paths
- [ ] Examples are complete and tested
- [ ] Spelling and grammar are correct
- [ ] Content follows style guidelines

### Peer Review

All documentation should be reviewed by:

1. **Technical accuracy**: Someone familiar with the topic
2. **Clarity**: Someone from the target audience
3. **Style compliance**: Documentation maintainer

## Tools and Automation

### Linting

Use markdownlint for consistency:

```bash
# Install markdownlint
npm install -g markdownlint-cli

# Check documentation
markdownlint docs/**/*.md
```

### Link Checking

Regularly check for broken links:

```bash
# Use the documentation validator
python tools/doc-generator/validator.py --check-links
```

### Spell Checking

Use a spell checker with technical dictionary:

```bash
# Example with aspell
aspell check --mode=markdown document.md
```

## Templates

Use provided templates for consistency:

- [Page Template](templates/page-template.md)
- [API Template](templates/api-template.md)
- [Troubleshooting Template](templates/troubleshooting-template.md)

## Maintenance

### Regular Updates

- Review and update documentation quarterly
- Update screenshots when UI changes
- Verify links and examples still work
- Update version-specific information

### Deprecation

When deprecating documentation:

1. Add deprecation notice at the top
2. Provide migration path
3. Set removal date
4. Update related documentation

Example deprecation notice:

```markdown
> **⚠️ Deprecated**: This guide is deprecated as of version 2.0.
> See the [new configuration guide](../configuration-v2.md) for current instructions.
> This page will be removed on 2024-06-01.
```

---

**Last Updated**: 2024-01-01  
**Next Review**: 2024-04-01  
**Maintainer**: Documentation Team
