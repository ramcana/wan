"""
Documentation Server

Static site generator and server for WAN22 documentation with search functionality.
Uses MkDocs for static site generation and provides development server capabilities.
"""

import os
import sys
import json
import yaml
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
import tempfile
from dataclasses import dataclass


@dataclass
class ServerConfig:
    """Configuration for documentation server"""
    site_name: str = "WAN22 Documentation"
    site_url: str = "http://localhost:8000"
    docs_dir: str = "docs"
    site_dir: str = "site"
    theme: str = "material"
    enable_search: bool = True
    enable_navigation: bool = True
    port: int = 8000
    dev_addr: str = "127.0.0.1:8000"


class DocumentationServer:
    """
    Documentation server using MkDocs for static site generation
    """
    
    def __init__(self, config: ServerConfig, project_root: Path):
        self.config = config
        self.project_root = Path(project_root)
        self.docs_dir = self.project_root / config.docs_dir
        self.site_dir = self.project_root / config.site_dir
        self.mkdocs_config_path = self.project_root / 'mkdocs.yml'
        
    def generate_mkdocs_config(self) -> Dict[str, Any]:
        """
        Generate MkDocs configuration
        """
        # Load existing navigation from metadata if available
        nav_structure = self._generate_navigation_structure()
        
        config = {
            'site_name': self.config.site_name,
            'site_url': self.config.site_url,
            'docs_dir': self.config.docs_dir,
            'site_dir': self.config.site_dir,
            
            # Theme configuration
            'theme': {
                'name': self.config.theme,
                'features': [
                    'navigation.tabs',
                    'navigation.sections',
                    'navigation.expand',
                    'navigation.top',
                    'search.highlight',
                    'search.share',
                    'content.code.copy',
                    'content.tabs.link'
                ],
                'palette': [
                    {
                        'scheme': 'default',
                        'primary': 'blue',
                        'accent': 'blue',
                        'toggle': {
                            'icon': 'material/brightness-7',
                            'name': 'Switch to dark mode'
                        }
                    },
                    {
                        'scheme': 'slate',
                        'primary': 'blue',
                        'accent': 'blue',
                        'toggle': {
                            'icon': 'material/brightness-4',
                            'name': 'Switch to light mode'
                        }
                    }
                ],
                'font': {
                    'text': 'Roboto',
                    'code': 'Roboto Mono'
                }
            },
            
            # Navigation structure
            'nav': nav_structure,
            
            # Plugins
            'plugins': [
                'search',
                'awesome-pages',
                {
                    'git-revision-date-localized': {
                        'type': 'date'
                    }
                }
            ],
            
            # Markdown extensions
            'markdown_extensions': [
                'admonition',
                'codehilite',
                'pymdownx.details',
                'pymdownx.superfences',
                'pymdownx.tabbed',
                'pymdownx.highlight',
                'pymdownx.inlinehilite',
                'pymdownx.snippets',
                'toc',
                'tables',
                'attr_list',
                'md_in_html'
            ],
            
            # Extra configuration
            'extra': {
                'social': [
                    {
                        'icon': 'fontawesome/brands/github',
                        'link': 'https://github.com/your-org/wan22'
                    }
                ],
                'version': {
                    'provider': 'mike'
                }
            },
            
            # Development server configuration
            'dev_addr': self.config.dev_addr,
            
            # Copyright
            'copyright': 'Copyright &copy; 2024 WAN22 Project'
        }
        
        return config
    
    def _generate_navigation_structure(self) -> List[Dict[str, Any]]:
        """
        Generate navigation structure from documentation files
        """
        nav = [
            {'Home': 'index.md'},
            {
                'User Guide': [
                    {'Overview': 'user-guide/index.md'},
                    {'Installation': 'user-guide/installation.md'},
                    {'Quick Start': 'user-guide/quick-start.md'},
                    {'Configuration': 'user-guide/configuration.md'},
                    {'Troubleshooting': 'user-guide/troubleshooting.md'}
                ]
            },
            {
                'Developer Guide': [
                    {'Overview': 'developer-guide/index.md'},
                    {'Architecture': 'developer-guide/architecture.md'},
                    {'Contributing': 'developer-guide/contributing.md'},
                    {'Testing': 'developer-guide/testing.md'}
                ]
            },
            {
                'API Reference': [
                    {'Overview': 'api/index.md'},
                    {'Backend API': 'api/backend-api.md'},
                    {'Frontend Components': 'api/frontend-components.md'}
                ]
            },
            {
                'Deployment': [
                    {'Overview': 'deployment/index.md'},
                    {'Production Setup': 'deployment/production-setup.md'},
                    {'Monitoring': 'deployment/monitoring.md'},
                    {'Performance': 'deployment/performance.md'}
                ]
            },
            {
                'Reference': [
                    {'Overview': 'reference/index.md'},
                    {'Configuration': 'reference/configuration/index.md'},
                    {'Error Codes': 'reference/troubleshooting/error-codes.md'},
                    {'Performance Benchmarks': 'reference/performance/benchmarks.md'}
                ]
            }
        ]
        
        return nav
    
    def save_mkdocs_config(self):
        """
        Save MkDocs configuration to file
        """
        config = self.generate_mkdocs_config()
        
        with open(self.mkdocs_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    def install_dependencies(self):
        """
        Install required MkDocs dependencies
        """
        dependencies = [
            'mkdocs',
            'mkdocs-material',
            'mkdocs-awesome-pages-plugin',
            'mkdocs-git-revision-date-localized-plugin',
            'pymdown-extensions'
        ]
        
        print("Installing MkDocs dependencies...")
        for dep in dependencies:
            try:
                subprocess.run([
                    sys.executable, '-m', 'pip', 'install', dep
                ], check=True, capture_output=True)
                print(f"✓ Installed {dep}")
            except subprocess.CalledProcessError as e:
                print(f"✗ Failed to install {dep}: {e}")
                return False
        
        return True
    
    def build_site(self) -> bool:
        """
        Build static documentation site
        """
        try:
            # Ensure MkDocs config exists
            if not self.mkdocs_config_path.exists():
                self.save_mkdocs_config()
            
            # Run MkDocs build
            result = subprocess.run([
                'mkdocs', 'build', '--config-file', str(self.mkdocs_config_path)
            ], cwd=self.project_root, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✓ Documentation site built successfully in {self.site_dir}")
                return True
            else:
                print(f"✗ Build failed: {result.stderr}")
                return False
                
        except FileNotFoundError:
            print("✗ MkDocs not found. Please install MkDocs first.")
            return False
        except Exception as e:
            print(f"✗ Build error: {e}")
            return False
    
    def serve_dev(self) -> bool:
        """
        Start development server with live reload
        """
        try:
            # Ensure MkDocs config exists
            if not self.mkdocs_config_path.exists():
                self.save_mkdocs_config()
            
            print(f"Starting development server at {self.config.dev_addr}")
            print("Press Ctrl+C to stop the server")
            
            # Run MkDocs serve
            subprocess.run([
                'mkdocs', 'serve', 
                '--config-file', str(self.mkdocs_config_path),
                '--dev-addr', self.config.dev_addr
            ], cwd=self.project_root)
            
            return True
            
        except FileNotFoundError:
            print("✗ MkDocs not found. Please install MkDocs first.")
            return False
        except KeyboardInterrupt:
            print("\n✓ Development server stopped")
            return True
        except Exception as e:
            print(f"✗ Server error: {e}")
            return False
    
    def generate_search_index(self) -> bool:
        """
        Generate search index for documentation
        """
        try:
            search_index = []
            
            # Scan all markdown files
            for md_file in self.docs_dir.rglob('*.md'):
                if self._should_skip_file(md_file):
                    continue
                
                try:
                    with open(md_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Extract metadata and content
                    title, body = self._extract_content_for_search(content)
                    relative_path = str(md_file.relative_to(self.docs_dir))
                    
                    search_index.append({
                        'title': title,
                        'path': relative_path,
                        'content': body[:500],  # Limit content length
                        'url': relative_path.replace('.md', '.html')
                    })
                    
                except Exception as e:
                    print(f"Warning: Could not index {md_file}: {e}")
            
            # Save search index
            search_index_path = self.site_dir / 'search' / 'search_index.json'
            search_index_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(search_index_path, 'w', encoding='utf-8') as f:
                json.dump(search_index, f, indent=2, ensure_ascii=False)
            
            print(f"✓ Search index generated with {len(search_index)} pages")
            return True
            
        except Exception as e:
            print(f"✗ Search index generation failed: {e}")
            return False
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped for indexing"""
        skip_patterns = [
            'templates',
            '.metadata',
            'node_modules'
        ]
        
        return any(pattern in str(file_path) for pattern in skip_patterns)
    
    def _extract_content_for_search(self, content: str) -> tuple[str, str]:
        """Extract title and searchable content from markdown"""
        lines = content.split('\n')
        title = "Untitled"
        body_lines = []
        
        # Skip frontmatter
        in_frontmatter = False
        if lines and lines[0].strip() == '---':
            in_frontmatter = True
            lines = lines[1:]
        
        for line in lines:
            if in_frontmatter:
                if line.strip() == '---':
                    in_frontmatter = False
                continue
            
            # Extract title from first heading
            if line.startswith('# ') and title == "Untitled":
                title = line[2:].strip()
            else:
                # Clean up line for search
                clean_line = line.strip()
                if clean_line and not clean_line.startswith('#'):
                    body_lines.append(clean_line)
        
        body = ' '.join(body_lines)
        return title, body
    
    def create_custom_theme(self):
        """
        Create custom theme files for WAN22 branding
        """
        theme_dir = self.project_root / 'theme'
        theme_dir.mkdir(exist_ok=True)
        
        # Custom CSS
        css_content = """
/* WAN22 Custom Styles */
:root {
    --md-primary-fg-color: #1976d2;
    --md-primary-fg-color--light: #42a5f5;
    --md-primary-fg-color--dark: #1565c0;
}

.md-header {
    background: linear-gradient(135deg, #1976d2 0%, #42a5f5 100%);
}

.md-nav__title {
    font-weight: 600;
}

.md-typeset h1 {
    color: var(--md-primary-fg-color);
}

/* Code block styling */
.md-typeset pre > code {
    background-color: #f8f9fa;
}

/* Search highlighting */
.md-search-result__teaser mark {
    background-color: #ffeb3b;
    color: #000;
}

/* Custom admonitions */
.md-typeset .admonition.tip {
    border-color: #00c853;
}

.md-typeset .admonition.tip > .admonition-title {
    background-color: rgba(0, 200, 83, 0.1);
}
"""
        
        css_file = theme_dir / 'extra.css'
        with open(css_file, 'w', encoding='utf-8') as f:
            f.write(css_content)
        
        # Custom JavaScript for enhanced search
        js_content = """
// WAN22 Documentation Enhancements

document.addEventListener('DOMContentLoaded', function() {
    // Enhanced search functionality
    const searchInput = document.querySelector('.md-search__input');
    if (searchInput) {
        searchInput.addEventListener('input', debounce(enhancedSearch, 300));
    }
    
    // Add copy buttons to code blocks
    addCopyButtons();
    
    // Add anchor links to headings
    addAnchorLinks();
});

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function enhancedSearch(event) {
    const query = event.target.value.toLowerCase();
    if (query.length < 2) return;
    
    // Custom search logic can be added here
    console.log('Searching for:', query);
}

function addCopyButtons() {
    const codeBlocks = document.querySelectorAll('pre > code');
    codeBlocks.forEach(block => {
        const button = document.createElement('button');
        button.className = 'copy-button';
        button.textContent = 'Copy';
        button.onclick = () => {
            navigator.clipboard.writeText(block.textContent);
            button.textContent = 'Copied!';
            setTimeout(() => button.textContent = 'Copy', 2000);
        };
        block.parentElement.appendChild(button);
    });
}

function addAnchorLinks() {
    const headings = document.querySelectorAll('h2, h3, h4, h5, h6');
    headings.forEach(heading => {
        if (heading.id) {
            const anchor = document.createElement('a');
            anchor.href = '#' + heading.id;
            anchor.className = 'anchor-link';
            anchor.textContent = '#';
            heading.appendChild(anchor);
        }
    });
}
"""
        
        js_file = theme_dir / 'extra.js'
        with open(js_file, 'w', encoding='utf-8') as f:
            f.write(js_content)
        
        print("✓ Custom theme files created")
    
    def setup_complete_server(self) -> bool:
        """
        Complete server setup including dependencies, config, and build
        """
        print("Setting up documentation server...")
        
        # Install dependencies
        if not self.install_dependencies():
            return False
        
        # Create custom theme
        self.create_custom_theme()
        
        # Generate and save MkDocs config
        self.save_mkdocs_config()
        print("✓ MkDocs configuration saved")
        
        # Build the site
        if not self.build_site():
            return False
        
        # Generate search index
        if not self.generate_search_index():
            return False
        
        print("✅ Documentation server setup complete!")
        print(f"   - Configuration: {self.mkdocs_config_path}")
        print(f"   - Built site: {self.site_dir}")
        print(f"   - To serve: mkdocs serve")
        print(f"   - To build: mkdocs build")
        
        return True


def main():
    """CLI interface for documentation server"""
    import argparse
    
    parser = argparse.ArgumentParser(description='WAN22 Documentation Server')
    parser.add_argument('command', choices=['setup', 'build', 'serve', 'install'], 
                       help='Command to execute')
    parser.add_argument('--docs-dir', default='docs', help='Documentation directory')
    parser.add_argument('--site-dir', default='site', help='Output site directory')
    parser.add_argument('--port', type=int, default=8000, help='Development server port')
    parser.add_argument('--host', default='127.0.0.1', help='Development server host')
    
    args = parser.parse_args()
    
    # Create server configuration
    config = ServerConfig(
        docs_dir=args.docs_dir,
        site_dir=args.site_dir,
        port=args.port,
        dev_addr=f"{args.host}:{args.port}"
    )
    
    # Initialize server
    project_root = Path.cwd()
    server = DocumentationServer(config, project_root)
    
    # Execute command
    if args.command == 'setup':
        success = server.setup_complete_server()
    elif args.command == 'install':
        success = server.install_dependencies()
    elif args.command == 'build':
        server.save_mkdocs_config()
        success = server.build_site()
    elif args.command == 'serve':
        server.save_mkdocs_config()
        success = server.serve_dev()
    else:
        print(f"Unknown command: {args.command}")
        success = False
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
