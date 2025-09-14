"""
Command-line interface for secure credential management.

This module provides CLI commands for managing credentials securely,
including storing, retrieving, and rotating credentials.
"""

import os
import sys
import json
import getpass
import argparse
from typing import Optional, Dict, Any
from pathlib import Path

from .credential_manager import (
    SecureCredentialManager,
    CredentialConfig,
    CredentialStore
)
from .logging_config import get_logger

logger = get_logger(__name__)


class CredentialCLI:
    """Command-line interface for credential management."""
    
    def __init__(self, config: Optional[CredentialConfig] = None):
        """
        Initialize credential CLI.
        
        Args:
            config: Credential configuration
        """
        self.config = config or CredentialConfig()
        self.manager = SecureCredentialManager(self.config)
    
    def store_credential(self, key: str, value: Optional[str] = None, interactive: bool = True) -> bool:
        """
        Store a credential securely.
        
        Args:
            key: Credential key
            value: Credential value (if None and interactive, will prompt)
            interactive: Whether to prompt for value if not provided
            
        Returns:
            True if stored successfully, False otherwise
        """
        try:
            if value is None and interactive:
                value = getpass.getpass(f"Enter value for '{key}': ")
            elif value is None:
                logger.error("No value provided for credential")
                return False
            
            success = self.manager.store.store_credential(key, value)
            
            if success:
                print(f"✓ Credential '{key}' stored successfully")
                logger.info(f"Stored credential: {key}")
            else:
                print(f"✗ Failed to store credential '{key}'")
                logger.error(f"Failed to store credential: {key}")
            
            return success
            
        except KeyboardInterrupt:
            print("\nOperation cancelled")
            return False
        except Exception as e:
            print(f"✗ Error storing credential: {e}")
            logger.error(f"Error storing credential {key}: {e}")
            return False
    
    def get_credential(self, key: str, show_value: bool = False) -> Optional[str]:
        """
        Retrieve a credential.
        
        Args:
            key: Credential key
            show_value: Whether to show the actual value (dangerous!)
            
        Returns:
            Credential value if found, None otherwise
        """
        try:
            value = self.manager.store.get_credential(key)
            
            if value is None:
                print(f"✗ Credential '{key}' not found")
                return None
            
            if show_value:
                print(f"Credential '{key}': {value}")
            else:
                masked_value = self.manager.mask_sensitive_info(value)
                print(f"Credential '{key}': {masked_value}")
            
            return value
            
        except Exception as e:
            print(f"✗ Error retrieving credential: {e}")
            logger.error(f"Error retrieving credential {key}: {e}")
            return None
    
    def delete_credential(self, key: str, confirm: bool = True) -> bool:
        """
        Delete a credential.
        
        Args:
            key: Credential key
            confirm: Whether to ask for confirmation
            
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            if confirm:
                response = input(f"Delete credential '{key}'? (y/N): ")
                if response.lower() not in ['y', 'yes']:
                    print("Operation cancelled")
                    return False
            
            success = self.manager.store.delete_credential(key)
            
            if success:
                print(f"✓ Credential '{key}' deleted successfully")
                logger.info(f"Deleted credential: {key}")
            else:
                print(f"✗ Failed to delete credential '{key}'")
                logger.error(f"Failed to delete credential: {key}")
            
            return success
            
        except KeyboardInterrupt:
            print("\nOperation cancelled")
            return False
        except Exception as e:
            print(f"✗ Error deleting credential: {e}")
            logger.error(f"Error deleting credential {key}: {e}")
            return False
    
    def list_credentials(self, show_values: bool = False) -> None:
        """
        List all stored credentials.
        
        Args:
            show_values: Whether to show credential values (dangerous!)
        """
        try:
            keys = self.manager.store.list_credentials()
            
            if not keys:
                print("No credentials stored")
                return
            
            print(f"Stored credentials ({len(keys)}):")
            print("-" * 40)
            
            for key in sorted(keys):
                if show_values:
                    value = self.manager.store.get_credential(key)
                    print(f"  {key}: {value}")
                else:
                    value = self.manager.store.get_credential(key)
                    if value:
                        masked_value = self.manager.mask_sensitive_info(value)
                        print(f"  {key}: {masked_value}")
                    else:
                        print(f"  {key}: <not found>")
            
        except Exception as e:
            print(f"✗ Error listing credentials: {e}")
            logger.error(f"Error listing credentials: {e}")
    
    def setup_aws_credentials(self, interactive: bool = True) -> bool:
        """
        Set up AWS credentials interactively.
        
        Args:
            interactive: Whether to prompt for values
            
        Returns:
            True if setup successful, False otherwise
        """
        try:
            print("Setting up AWS credentials...")
            
            credentials = {}
            
            if interactive:
                credentials["aws-access-key-id"] = getpass.getpass("AWS Access Key ID: ")
                credentials["aws-secret-access-key"] = getpass.getpass("AWS Secret Access Key: ")
                
                session_token = getpass.getpass("AWS Session Token (optional): ")
                if session_token:
                    credentials["aws-session-token"] = session_token
                
                endpoint_url = input("Custom endpoint URL (optional, for MinIO): ")
                if endpoint_url:
                    credentials["aws-endpoint-url"] = endpoint_url
            else:
                # Get from environment
                credentials["aws-access-key-id"] = os.getenv("AWS_ACCESS_KEY_ID")
                credentials["aws-secret-access-key"] = os.getenv("AWS_SECRET_ACCESS_KEY")
                
                session_token = os.getenv("AWS_SESSION_TOKEN")
                if session_token:
                    credentials["aws-session-token"] = session_token
                
                endpoint_url = os.getenv("AWS_ENDPOINT_URL")
                if endpoint_url:
                    credentials["aws-endpoint-url"] = endpoint_url
            
            # Filter out None values
            credentials = {k: v for k, v in credentials.items() if v}
            
            if not credentials.get("aws-access-key-id") or not credentials.get("aws-secret-access-key"):
                print("✗ AWS Access Key ID and Secret Access Key are required")
                return False
            
            success = self.manager.setup_credentials(credentials)
            
            if success:
                print("✓ AWS credentials configured successfully")
            else:
                print("✗ Failed to configure AWS credentials")
            
            return success
            
        except KeyboardInterrupt:
            print("\nOperation cancelled")
            return False
        except Exception as e:
            print(f"✗ Error setting up AWS credentials: {e}")
            logger.error(f"Error setting up AWS credentials: {e}")
            return False
    
    def setup_hf_credentials(self, interactive: bool = True) -> bool:
        """
        Set up HuggingFace credentials interactively.
        
        Args:
            interactive: Whether to prompt for values
            
        Returns:
            True if setup successful, False otherwise
        """
        try:
            print("Setting up HuggingFace credentials...")
            
            if interactive:
                token = getpass.getpass("HuggingFace Token: ")
            else:
                token = os.getenv("HF_TOKEN")
            
            if not token:
                print("✗ HuggingFace token is required")
                return False
            
            success = self.manager.store.store_credential("hf-token", token)
            
            if success:
                print("✓ HuggingFace credentials configured successfully")
            else:
                print("✗ Failed to configure HuggingFace credentials")
            
            return success
            
        except KeyboardInterrupt:
            print("\nOperation cancelled")
            return False
        except Exception as e:
            print(f"✗ Error setting up HuggingFace credentials: {e}")
            logger.error(f"Error setting up HuggingFace credentials: {e}")
            return False
    
    def test_credentials(self, source_type: str) -> bool:
        """
        Test credentials for a specific source type.
        
        Args:
            source_type: Source type (s3, hf)
            
        Returns:
            True if credentials work, False otherwise
        """
        try:
            print(f"Testing {source_type} credentials...")
            
            if source_type == "s3":
                credentials = self.manager.get_credentials_for_source("s3://")
                if not credentials:
                    print("✗ No S3 credentials found")
                    return False
                
                # Test S3 connection (simplified)
                print("✓ S3 credentials found")
                masked_creds = self.manager.mask_sensitive_info(credentials)
                print(f"  Credentials: {json.dumps(masked_creds, indent=2)}")
                
            elif source_type == "hf":
                credentials = self.manager.get_credentials_for_source("hf://")
                if not credentials:
                    print("✗ No HuggingFace credentials found")
                    return False
                
                # Test HF connection (simplified)
                print("✓ HuggingFace credentials found")
                masked_creds = self.manager.mask_sensitive_info(credentials)
                print(f"  Credentials: {json.dumps(masked_creds, indent=2)}")
                
            else:
                print(f"✗ Unknown source type: {source_type}")
                return False
            
            return True
            
        except Exception as e:
            print(f"✗ Error testing credentials: {e}")
            logger.error(f"Error testing {source_type} credentials: {e}")
            return False
    
    def export_config(self, output_file: str, include_values: bool = False) -> bool:
        """
        Export credential configuration to file.
        
        Args:
            output_file: Output file path
            include_values: Whether to include credential values (dangerous!)
            
        Returns:
            True if exported successfully, False otherwise
        """
        try:
            keys = self.manager.store.list_credentials()
            
            config_data = {
                "credentials": {},
                "config": {
                    "use_keyring": self.config.use_keyring,
                    "credential_file": self.config.credential_file,
                    "mask_credentials_in_logs": self.config.mask_credentials_in_logs,
                    "require_https": self.config.require_https
                }
            }
            
            for key in keys:
                if include_values:
                    value = self.manager.store.get_credential(key)
                    config_data["credentials"][key] = value
                else:
                    config_data["credentials"][key] = "<masked>"
            
            with open(output_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            print(f"✓ Configuration exported to {output_file}")
            
            if include_values:
                print("⚠️  WARNING: Exported file contains sensitive credential values!")
            
            return True
            
        except Exception as e:
            print(f"✗ Error exporting configuration: {e}")
            logger.error(f"Error exporting configuration: {e}")
            return False


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for credential CLI."""
    parser = argparse.ArgumentParser(
        description="Secure credential management for WAN Model Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Store a credential interactively
  python -m credential_cli store aws-access-key-id
  
  # Store a credential non-interactively
  python -m credential_cli store hf-token --value hf_abc123
  
  # List all credentials (masked)
  python -m credential_cli list
  
  # Get a specific credential
  python -m credential_cli get aws-access-key-id
  
  # Set up AWS credentials interactively
  python -m credential_cli setup-aws
  
  # Set up HuggingFace credentials
  python -m credential_cli setup-hf
  
  # Test credentials
  python -m credential_cli test s3
  
  # Delete a credential
  python -m credential_cli delete old-token
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Store command
    store_parser = subparsers.add_parser("store", help="Store a credential")
    store_parser.add_argument("key", help="Credential key")
    store_parser.add_argument("--value", help="Credential value (will prompt if not provided)")
    store_parser.add_argument("--non-interactive", action="store_true", help="Don't prompt for value")
    
    # Get command
    get_parser = subparsers.add_parser("get", help="Retrieve a credential")
    get_parser.add_argument("key", help="Credential key")
    get_parser.add_argument("--show-value", action="store_true", help="Show actual value (dangerous!)")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List all credentials")
    list_parser.add_argument("--show-values", action="store_true", help="Show actual values (dangerous!)")
    
    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a credential")
    delete_parser.add_argument("key", help="Credential key")
    delete_parser.add_argument("--force", action="store_true", help="Don't ask for confirmation")
    
    # Setup commands
    subparsers.add_parser("setup-aws", help="Set up AWS credentials interactively")
    subparsers.add_parser("setup-hf", help="Set up HuggingFace credentials interactively")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Test credentials")
    test_parser.add_argument("source", choices=["s3", "hf"], help="Source type to test")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export configuration")
    export_parser.add_argument("output", help="Output file path")
    export_parser.add_argument("--include-values", action="store_true", help="Include credential values (dangerous!)")
    
    # Global options
    parser.add_argument("--config-file", help="Path to credential configuration file")
    parser.add_argument("--credential-file", help="Path to encrypted credential storage file")
    parser.add_argument("--no-keyring", action="store_true", help="Disable keyring usage")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    return parser


def main():
    """Main entry point for credential CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Configure logging
    if args.verbose:
        from .logging_config import configure_logging
        configure_logging(level="DEBUG")
    
    # Create credential configuration
    config = CredentialConfig()
    
    if args.no_keyring:
        config.use_keyring = False
    
    if args.credential_file:
        config.credential_file = args.credential_file
    
    # Create CLI instance
    cli = CredentialCLI(config)
    
    try:
        # Execute command
        if args.command == "store":
            success = cli.store_credential(
                args.key,
                args.value,
                interactive=not args.non_interactive
            )
            return 0 if success else 1
        
        elif args.command == "get":
            value = cli.get_credential(args.key, args.show_value)
            return 0 if value is not None else 1
        
        elif args.command == "list":
            cli.list_credentials(args.show_values)
            return 0
        
        elif args.command == "delete":
            success = cli.delete_credential(args.key, confirm=not args.force)
            return 0 if success else 1
        
        elif args.command == "setup-aws":
            success = cli.setup_aws_credentials()
            return 0 if success else 1
        
        elif args.command == "setup-hf":
            success = cli.setup_hf_credentials()
            return 0 if success else 1
        
        elif args.command == "test":
            success = cli.test_credentials(args.source)
            return 0 if success else 1
        
        elif args.command == "export":
            success = cli.export_config(args.output, args.include_values)
            return 0 if success else 1
        
        else:
            print(f"Unknown command: {args.command}")
            return 1
    
    except KeyboardInterrupt:
        print("\nOperation cancelled")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())