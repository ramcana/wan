#!/usr/bin/env python3
"""
Configure branch protection rules with health monitoring integration.

This script sets up branch protection rules that require health checks
to pass before allowing merges to protected branches.
"""

import json
import os
import sys
from typing import Dict, List, Optional
import requests


class BranchProtectionManager:
    """Manages GitHub branch protection rules with health monitoring integration."""
    
    def __init__(self, repo_owner: str, repo_name: str, github_token: str):
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.github_token = github_token
        self.base_url = "https://api.github.com"
        self.headers = {
            "Authorization": f"token {github_token}",
            "Accept": "application/vnd.github.v3+json",
            "Content-Type": "application/json"
        }
    
    def get_current_protection(self, branch: str) -> Optional[Dict]:
        """Get current branch protection settings."""
        url = f"{self.base_url}/repos/{self.repo_owner}/{self.repo_name}/branches/{branch}/protection"
        
        response = requests.get(url, headers=self.headers)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            return None
        else:
            raise Exception(f"Failed to get branch protection: {response.status_code} - {response.text}")
    
    def configure_health_monitoring_protection(self, branch: str) -> Dict:
        """Configure branch protection with health monitoring requirements."""
        
        protection_config = {
            "required_status_checks": {
                "strict": True,
                "contexts": [
                    # Health monitoring checks
                    "health-gate-check",
                    "test-gate",
                    "deployment-gate",
                    
                    # Existing CI checks
                    "test (3.10)",  # Main Python version test
                    "performance-tests",
                    "health-check",
                    
                    # Configuration and documentation checks
                    "validate-config",
                    "build-docs",
                    "validate-docs"
                ]
            },
            "enforce_admins": True,
            "required_pull_request_reviews": {
                "required_approving_review_count": 1,
                "dismiss_stale_reviews": True,
                "require_code_owner_reviews": True,
                "require_last_push_approval": False
            },
            "restrictions": None,  # No user/team restrictions
            "allow_force_pushes": False,
            "allow_deletions": False,
            "block_creations": False,
            "required_conversation_resolution": True
        }
        
        return protection_config
    
    def apply_protection(self, branch: str, config: Dict) -> bool:
        """Apply branch protection configuration."""
        url = f"{self.base_url}/repos/{self.repo_owner}/{self.repo_name}/branches/{branch}/protection"
        
        response = requests.put(url, headers=self.headers, json=config)
        
        if response.status_code in [200, 201]:
            print(f"✅ Successfully configured branch protection for '{branch}'")
            return True
        else:
            print(f"❌ Failed to configure branch protection for '{branch}': {response.status_code}")
            print(f"Response: {response.text}")
            return False
    
    def setup_health_monitoring_protection(self, branches: List[str] = None) -> bool:
        """Set up health monitoring branch protection for specified branches."""
        if branches is None:
            branches = ["main", "develop"]
        
        success = True
        
        for branch in branches:
            print(f"\n🔧 Configuring branch protection for '{branch}'...")
            
            try:
                # Get current protection settings
                current = self.get_current_protection(branch)
                if current:
                    print(f"📋 Current protection exists for '{branch}'")
                else:
                    print(f"📋 No existing protection for '{branch}'")
                
                # Configure health monitoring protection
                config = self.configure_health_monitoring_protection(branch)
                
                # Apply configuration
                if self.apply_protection(branch, config):
                    print(f"✅ Branch protection configured for '{branch}'")
                else:
                    print(f"❌ Failed to configure protection for '{branch}'")
                    success = False
                    
            except Exception as e:
                print(f"❌ Error configuring protection for '{branch}': {e}")
                success = False
        
        return success
    
    def create_health_check_ruleset(self) -> bool:
        """Create a repository ruleset for health monitoring."""
        
        ruleset_config = {
            "name": "Health Monitoring Requirements",
            "target": "branch",
            "enforcement": "active",
            "conditions": {
                "ref_name": {
                    "include": ["refs/heads/main", "refs/heads/develop"],
                    "exclude": []
                }
            },
            "rules": [
                {
                    "type": "required_status_checks",
                    "parameters": {
                        "strict_required_status_checks_policy": True,
                        "required_status_checks": [
                            {
                                "context": "health-gate-check",
                                "integration_id": None
                            },
                            {
                                "context": "test-gate", 
                                "integration_id": None
                            },
                            {
                                "context": "deployment-gate",
                                "integration_id": None
                            }
                        ]
                    }
                },
                {
                    "type": "pull_request",
                    "parameters": {
                        "required_approving_review_count": 1,
                        "dismiss_stale_reviews_on_push": True,
                        "require_code_owner_review": True,
                        "require_last_push_approval": False
                    }
                }
            ]
        }
        
        url = f"{self.base_url}/repos/{self.repo_owner}/{self.repo_name}/rulesets"
        
        response = requests.post(url, headers=self.headers, json=ruleset_config)
        
        if response.status_code in [200, 201]:
            print("✅ Successfully created health monitoring ruleset")
            return True
        else:
            print(f"❌ Failed to create ruleset: {response.status_code}")
            print(f"Response: {response.text}")
            return False


def main():
    """Main function to configure branch protection."""
    
    # Get configuration from environment or command line
    repo_owner = os.getenv("GITHUB_REPOSITORY_OWNER")
    repo_name = os.getenv("GITHUB_REPOSITORY", "").split("/")[-1] if os.getenv("GITHUB_REPOSITORY") else None
    github_token = os.getenv("GITHUB_TOKEN")
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Configure branch protection with health monitoring")
    parser.add_argument("--owner", default=repo_owner, help="Repository owner")
    parser.add_argument("--repo", default=repo_name, help="Repository name")
    parser.add_argument("--token", default=github_token, help="GitHub token")
    parser.add_argument("--branches", nargs="+", default=["main", "develop"], help="Branches to protect")
    parser.add_argument("--dry-run", action="store_true", help="Show configuration without applying")
    
    args = parser.parse_args()
    
    if not all([args.owner, args.repo, args.token]):
        print("❌ Missing required parameters: owner, repo, and token must be provided")
        print("   Either set GITHUB_REPOSITORY_OWNER, GITHUB_REPOSITORY, and GITHUB_TOKEN environment variables")
        print("   or provide --owner, --repo, and --token arguments")
        sys.exit(1)
    
    print(f"🔧 Configuring branch protection for {args.owner}/{args.repo}")
    print(f"📋 Branches: {', '.join(args.branches)}")
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No changes will be made")
        
        # Show what would be configured
        manager = BranchProtectionManager(args.owner, args.repo, args.token)
        config = manager.configure_health_monitoring_protection("main")
        
        print("\n📋 Branch protection configuration:")
        print(json.dumps(config, indent=2))
        return
    
    # Apply branch protection
    manager = BranchProtectionManager(args.owner, args.repo, args.token)
    
    success = manager.setup_health_monitoring_protection(args.branches)
    
    if success:
        print("\n✅ Branch protection configuration completed successfully!")
        print("\n📋 Health monitoring is now integrated with branch protection:")
        print("   - Pull requests must pass health checks before merging")
        print("   - Deployment gates will block unhealthy deployments")
        print("   - Health scores and coverage are tracked automatically")
    else:
        print("\n❌ Branch protection configuration failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
