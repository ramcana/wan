"""Tests for S3/MinIO storage backend."""

import os
import pytest
import tempfile
import threading
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass

from .s3_store import S3Store, S3Config
from .base_store import DownloadResult


@dataclass
class MockFileSpec:
    """Mock FileSpec for testing."""
    path: str
    size: int
    sha256: str
    optional: bool = False


class TestS3Store:
    """Test suite for S3Store."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = S3Config(
            endpoint_url="http://localhost:9000",
            access_key_id="test_key",
            secret_access_key="test_secret",
            region_name="us-east-1",
            max_concurrent_downloads=2,
            chunk_size=1024,
            max_retries=2,
            retry_backoff=0.1
        )

    def test_can_handle_s3_urls(self):
        """Test that S3Store can handle S3 URLs."""
        store = S3Store(self.config)
        
        assert store.can_handle("s3://bucket/path")
        assert store.can_handle("s3://my-bucket/models/t2v")
        assert not store.can_handle("hf://repo")
        assert not store.can_handle("local://path")
        assert not store.can_handle("https://example.com")

    def test_parse_s3_url(self):
        """Test S3 URL parsing."""
        store = S3Store(self.config)
        
        # Test basic URL
        bucket, prefix = store._parse_s3_url("s3://my-bucket/models/t2v")
        assert bucket == "my-bucket"
        assert prefix == "models/t2v"
        
        # Test URL with no prefix
        bucket, prefix = store._parse_s3_url("s3://my-bucket")
        assert bucket == "my-bucket"
        assert prefix == ""
        
        # Test URL with trailing slash
        bucket, prefix = store._parse_s3_url("s3://my-bucket/models/")
        assert bucket == "my-bucket"
        assert prefix == "models/"
        
        # Test invalid URL
        with pytest.raises(ValueError, match="Invalid S3 URL"):
            store._parse_s3_url("invalid://url")

    @patch('boto3.client')
    def test_client_initialization(self, mock_boto3_client):
        """Test S3 client initialization."""
        mock_client = Mock()
        mock_boto3_client.return_value = mock_client
        
        store = S3Store(self.config)
        client = store._get_client()
        
        # Verify client was created with correct parameters
        mock_boto3_client.assert_called_once_with(
            service_name="s3",
            region_name="us-east-1",
            endpoint_url="http://localhost:9000",
            aws_access_key_id="test_key",
            aws_secret_access_key="test_secret"
        )
        
        assert client == mock_client
        
        # Test that subsequent calls return the same client
        client2 = store._get_client()
        assert client2 == mock_client
        assert mock_boto3_client.call_count == 1

    @patch('boto3.client')
    def test_client_initialization_with_env_vars(self, mock_boto3_client):
        """Test S3 client initialization with environment variables."""
        mock_client = Mock()
        mock_boto3_client.return_value = mock_client
        
        with patch.dict(os.environ, {
            'AWS_ENDPOINT_URL': 'http://env-endpoint:9000',
            'AWS_ACCESS_KEY_ID': 'env_key',
            'AWS_SECRET_ACCESS_KEY': 'env_secret',
            'AWS_DEFAULT_REGION': 'eu-west-1'
        }):
            store = S3Store()  # No config provided
            client = store._get_client()
            
            # Verify environment variables were used
            mock_boto3_client.assert_called_once_with(
                service_name="s3",
                region_name="eu-west-1",
                endpoint_url="http://env-endpoint:9000",
                aws_access_key_id="env_key",
                aws_secret_access_key="env_secret"
            )

    @patch('boto3.client')
    def test_list_objects(self, mock_boto3_client):
        """Test listing S3 objects."""
        mock_client = Mock()
        mock_boto3_client.return_value = mock_client
        
        # Mock paginator response
        mock_paginator = Mock()
        mock_client.get_paginator.return_value = mock_paginator
        
        mock_page_iterator = [
            {
                'Contents': [
                    {
                        'Key': 'models/t2v/model.safetensors',
                        'Size': 1000,
                        'ETag': '"abc123"',
                        'LastModified': '2024-01-01T00:00:00Z'
                    },
                    {
                        'Key': 'models/t2v/config.json',
                        'Size': 500,
                        'ETag': '"def456"',
                        'LastModified': '2024-01-01T00:00:00Z'
                    }
                ]
            }
        ]
        mock_paginator.paginate.return_value = mock_page_iterator
        
        store = S3Store(self.config)
        objects = store._list_objects("test-bucket", "models/t2v")
        
        assert len(objects) == 2
        assert objects[0]['Key'] == 'models/t2v/model.safetensors'
        assert objects[0]['Size'] == 1000
        assert objects[0]['ETag'] == 'abc123'
        assert objects[1]['Key'] == 'models/t2v/config.json'
        assert objects[1]['Size'] == 500

    @patch('boto3.client')
    def test_list_objects_with_patterns(self, mock_boto3_client):
        """Test listing S3 objects with allow patterns."""
        mock_client = Mock()
        mock_boto3_client.return_value = mock_client
        
        # Mock paginator response
        mock_paginator = Mock()
        mock_client.get_paginator.return_value = mock_paginator
        
        mock_page_iterator = [
            {
                'Contents': [
                    {
                        'Key': 'models/t2v/model.safetensors',
                        'Size': 1000,
                        'ETag': '"abc123"',
                        'LastModified': '2024-01-01T00:00:00Z'
                    },
                    {
                        'Key': 'models/t2v/config.json',
                        'Size': 500,
                        'ETag': '"def456"',
                        'LastModified': '2024-01-01T00:00:00Z'
                    },
                    {
                        'Key': 'models/t2v/readme.txt',
                        'Size': 100,
                        'ETag': '"ghi789"',
                        'LastModified': '2024-01-01T00:00:00Z'
                    }
                ]
            }
        ]
        mock_paginator.paginate.return_value = mock_page_iterator
        
        store = S3Store(self.config)
        
        # Test with patterns that should match only .safetensors and .json files
        objects = store._list_objects("test-bucket", "models/t2v", ["*.safetensors", "*.json"])
        
        assert len(objects) == 2
        assert objects[0]['Key'] == 'models/t2v/model.safetensors'
        assert objects[1]['Key'] == 'models/t2v/config.json'

    @patch('boto3.client')
    def test_download_file_with_resume_new_file(self, mock_boto3_client):
        """Test downloading a new file."""
        mock_client = Mock()
        mock_boto3_client.return_value = mock_client
        
        # Mock head_object response
        mock_client.head_object.return_value = {
            'ContentLength': 1000,
            'ETag': '"abc123"'
        }
        
        # Mock get_object response
        mock_body = Mock()
        mock_body.read.side_effect = [b'x' * 500, b'x' * 500, b'']
        mock_response = {'Body': mock_body}
        mock_client.get_object.return_value = mock_response
        
        store = S3Store(self.config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            local_path = Path(temp_dir) / "test_file.bin"
            
            bytes_downloaded = store._download_file_with_resume(
                "test-bucket", "test-key", local_path, 1000
            )
            
            assert bytes_downloaded == 1000
            assert local_path.exists()
            assert local_path.stat().st_size == 1000
            
            # Verify S3 calls
            mock_client.head_object.assert_called_once_with(Bucket="test-bucket", Key="test-key")
            # The implementation adds Range header even for new files starting from byte 0
            mock_client.get_object.assert_called_once_with(Bucket="test-bucket", Key="test-key", Range="bytes=0-")

    @patch('boto3.client')
    def test_download_file_with_resume_partial_file(self, mock_boto3_client):
        """Test resuming download of a partial file."""
        mock_client = Mock()
        mock_boto3_client.return_value = mock_client
        
        # Mock head_object response
        mock_client.head_object.return_value = {
            'ContentLength': 1000,
            'ETag': '"abc123"'
        }
        
        # Mock get_object response for remaining bytes
        mock_body = Mock()
        mock_body.read.side_effect = [b'y' * 300, b'']
        mock_response = {'Body': mock_body}
        mock_client.get_object.return_value = mock_response
        
        store = S3Store(self.config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            local_path = Path(temp_dir) / "test_file.bin"
            
            # Create partial file
            with open(local_path, "wb") as f:
                f.write(b'x' * 700)
            
            bytes_downloaded = store._download_file_with_resume(
                "test-bucket", "test-key", local_path, 1000
            )
            
            assert bytes_downloaded == 300
            assert local_path.exists()
            assert local_path.stat().st_size == 1000
            
            # Verify range request was made
            mock_client.get_object.assert_called_once_with(
                Bucket="test-bucket", Key="test-key", Range="bytes=700-"
            )

    @patch('boto3.client')
    def test_download_file_already_complete(self, mock_boto3_client):
        """Test handling of already complete file."""
        mock_client = Mock()
        mock_boto3_client.return_value = mock_client
        
        # Mock head_object response
        mock_client.head_object.return_value = {
            'ContentLength': 1000,
            'ETag': '"abc123"'
        }
        
        store = S3Store(self.config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            local_path = Path(temp_dir) / "test_file.bin"
            
            # Create complete file
            with open(local_path, "wb") as f:
                f.write(b'x' * 1000)
            
            bytes_downloaded = store._download_file_with_resume(
                "test-bucket", "test-key", local_path, 1000
            )
            
            assert bytes_downloaded == 0
            assert local_path.stat().st_size == 1000
            
            # Verify no download was attempted
            mock_client.get_object.assert_not_called()

    @patch('boto3.client')
    def test_download_with_retry(self, mock_boto3_client):
        """Test download retry logic."""
        mock_client = Mock()
        mock_boto3_client.return_value = mock_client
        
        # Mock head_object response
        mock_client.head_object.return_value = {
            'ContentLength': 1000,
            'ETag': '"abc123"'
        }
        
        # Mock get_object to fail twice, then succeed
        mock_body_success = Mock()
        mock_body_success.read.side_effect = [b'x' * 1000, b'']
        mock_response_success = {'Body': mock_body_success}
        
        mock_client.get_object.side_effect = [
            Exception("Network error"),
            mock_response_success
        ]
        
        store = S3Store(self.config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            local_path = Path(temp_dir) / "test_file.bin"
            
            # Should succeed after retries
            bytes_downloaded = store._download_file_with_resume(
                "test-bucket", "test-key", local_path, 1000
            )
            
            assert bytes_downloaded == 1000
            assert mock_client.get_object.call_count == 2

    @patch('boto3.client')
    def test_download_max_retries_exceeded(self, mock_boto3_client):
        """Test download failure after max retries."""
        mock_client = Mock()
        mock_boto3_client.return_value = mock_client
        
        # Mock head_object response
        mock_client.head_object.return_value = {
            'ContentLength': 1000,
            'ETag': '"abc123"'
        }
        
        # Mock get_object to always fail
        mock_client.get_object.side_effect = Exception("Persistent error")
        
        store = S3Store(self.config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            local_path = Path(temp_dir) / "test_file.bin"
            
            # Should fail after max retries
            with pytest.raises(Exception, match="Persistent error"):
                store._download_file_with_resume(
                    "test-bucket", "test-key", local_path, 1000
                )
            
            assert mock_client.get_object.call_count == 2  # max_retries

    @patch('boto3.client')
    def test_download_success(self, mock_boto3_client):
        """Test successful download of multiple files."""
        mock_client = Mock()
        mock_boto3_client.return_value = mock_client
        
        # Mock paginator for listing objects
        mock_paginator = Mock()
        mock_client.get_paginator.return_value = mock_paginator
        
        mock_page_iterator = [
            {
                'Contents': [
                    {
                        'Key': 'models/t2v/model.safetensors',
                        'Size': 1000,
                        'ETag': '"abc123"',
                        'LastModified': '2024-01-01T00:00:00Z'
                    },
                    {
                        'Key': 'models/t2v/config.json',
                        'Size': 500,
                        'ETag': '"def456"',
                        'LastModified': '2024-01-01T00:00:00Z'
                    }
                ]
            }
        ]
        mock_paginator.paginate.return_value = mock_page_iterator
        
        # Mock head_object responses
        mock_client.head_object.side_effect = [
            {'ContentLength': 1000, 'ETag': '"abc123"'},
            {'ContentLength': 500, 'ETag': '"def456"'}
        ]
        
        # Mock get_object responses
        mock_body1 = Mock()
        mock_body1.read.side_effect = [b'x' * 1000, b'']
        mock_response1 = {'Body': mock_body1}
        
        mock_body2 = Mock()
        mock_body2.read.side_effect = [b'y' * 500, b'']
        mock_response2 = {'Body': mock_body2}
        
        mock_client.get_object.side_effect = [mock_response1, mock_response2]
        
        store = S3Store(self.config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            local_dir = Path(temp_dir)
            
            result = store.download(
                "s3://test-bucket/models/t2v",
                local_dir,
                allow_patterns=["*.safetensors", "*.json"]
            )
            
            assert result.success
            assert result.bytes_downloaded == 1500
            assert result.files_downloaded == 2
            assert result.error_message is None
            assert result.duration_seconds is not None
            
            # Verify files were created
            assert (local_dir / "model.safetensors").exists()
            assert (local_dir / "config.json").exists()
            assert (local_dir / "model.safetensors").stat().st_size == 1000
            assert (local_dir / "config.json").stat().st_size == 500

    @patch('boto3.client')
    def test_download_with_file_specs_validation(self, mock_boto3_client):
        """Test download with file specs validation."""
        mock_client = Mock()
        mock_boto3_client.return_value = mock_client
        
        # Mock paginator for listing objects
        mock_paginator = Mock()
        mock_client.get_paginator.return_value = mock_paginator
        
        mock_page_iterator = [
            {
                'Contents': [
                    {
                        'Key': 'models/t2v/model.safetensors',
                        'Size': 1000,
                        'ETag': '"abc123"',
                        'LastModified': '2024-01-01T00:00:00Z'
                    }
                ]
            }
        ]
        mock_paginator.paginate.return_value = mock_page_iterator
        
        # Mock head_object response
        mock_client.head_object.return_value = {
            'ContentLength': 1000,
            'ETag': '"abc123"'
        }
        
        # Mock get_object response
        mock_body = Mock()
        mock_body.read.side_effect = [b'x' * 1000, b'']
        mock_response = {'Body': mock_body}
        mock_client.get_object.return_value = mock_response
        
        store = S3Store(self.config)
        
        # Create file spec with expected size
        file_specs = [
            MockFileSpec(path="model.safetensors", size=1000, sha256="abc123")
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            local_dir = Path(temp_dir)
            
            result = store.download(
                "s3://test-bucket/models/t2v",
                local_dir,
                file_specs=file_specs
            )
            
            assert result.success
            assert result.files_downloaded == 1

    @patch('boto3.client')
    def test_download_no_objects_found(self, mock_boto3_client):
        """Test download when no objects are found."""
        mock_client = Mock()
        mock_boto3_client.return_value = mock_client
        
        # Mock paginator with empty response
        mock_paginator = Mock()
        mock_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = []
        
        store = S3Store(self.config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            local_dir = Path(temp_dir)
            
            result = store.download("s3://test-bucket/models/t2v", local_dir)
            
            assert result.success
            assert result.bytes_downloaded == 0
            assert result.files_downloaded == 0

    @patch('boto3.client')
    def test_download_partial_failure(self, mock_boto3_client):
        """Test download with some files failing."""
        mock_client = Mock()
        mock_boto3_client.return_value = mock_client
        
        # Mock paginator for listing objects
        mock_paginator = Mock()
        mock_client.get_paginator.return_value = mock_paginator
        
        mock_page_iterator = [
            {
                'Contents': [
                    {
                        'Key': 'models/t2v/model.safetensors',
                        'Size': 1000,
                        'ETag': '"abc123"',
                        'LastModified': '2024-01-01T00:00:00Z'
                    },
                    {
                        'Key': 'models/t2v/config.json',
                        'Size': 500,
                        'ETag': '"def456"',
                        'LastModified': '2024-01-01T00:00:00Z'
                    }
                ]
            }
        ]
        mock_paginator.paginate.return_value = mock_page_iterator
        
        # Mock head_object responses - first succeeds, second fails
        mock_client.head_object.side_effect = [
            {'ContentLength': 1000, 'ETag': '"abc123"'},
            Exception("Access denied")
        ]
        
        # Mock get_object response for successful file
        mock_body = Mock()
        mock_body.read.side_effect = [b'x' * 1000, b'']
        mock_response = {'Body': mock_body}
        mock_client.get_object.return_value = mock_response
        
        store = S3Store(self.config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            local_dir = Path(temp_dir)
            
            result = store.download("s3://test-bucket/models/t2v", local_dir)
            
            assert not result.success
            assert result.bytes_downloaded == 1000  # One file succeeded
            assert result.files_downloaded == 1
            assert "Some downloads failed" in result.error_message

    @patch('boto3.client')
    def test_verify_availability_success(self, mock_boto3_client):
        """Test successful availability verification."""
        mock_client = Mock()
        mock_boto3_client.return_value = mock_client
        
        # Mock successful list_objects_v2 response
        mock_client.list_objects_v2.return_value = {'Contents': []}
        
        store = S3Store(self.config)
        
        assert store.verify_availability("s3://test-bucket/models/t2v")
        
        mock_client.list_objects_v2.assert_called_once_with(
            Bucket="test-bucket",
            Prefix="models/t2v",
            MaxKeys=1
        )

    @patch('boto3.client')
    def test_verify_availability_no_credentials(self, mock_boto3_client):
        """Test availability verification with no credentials."""
        mock_client = Mock()
        mock_boto3_client.return_value = mock_client
        
        from botocore.exceptions import NoCredentialsError
        mock_client.list_objects_v2.side_effect = NoCredentialsError()
        
        store = S3Store(self.config)
        
        assert not store.verify_availability("s3://test-bucket/models/t2v")

    @patch('boto3.client')
    def test_verify_availability_client_error(self, mock_boto3_client):
        """Test availability verification with client errors."""
        mock_client = Mock()
        mock_boto3_client.return_value = mock_client
        
        from botocore.exceptions import ClientError
        error_response = {'Error': {'Code': 'NoSuchBucket'}}
        mock_client.list_objects_v2.side_effect = ClientError(error_response, 'ListObjectsV2')
        
        store = S3Store(self.config)
        
        assert not store.verify_availability("s3://test-bucket/models/t2v")

    @patch('boto3.client')
    def test_estimate_download_size(self, mock_boto3_client):
        """Test download size estimation."""
        mock_client = Mock()
        mock_boto3_client.return_value = mock_client
        
        # Mock paginator for listing objects
        mock_paginator = Mock()
        mock_client.get_paginator.return_value = mock_paginator
        
        mock_page_iterator = [
            {
                'Contents': [
                    {
                        'Key': 'models/t2v/model.safetensors',
                        'Size': 1000,
                        'ETag': '"abc123"',
                        'LastModified': '2024-01-01T00:00:00Z'
                    },
                    {
                        'Key': 'models/t2v/config.json',
                        'Size': 500,
                        'ETag': '"def456"',
                        'LastModified': '2024-01-01T00:00:00Z'
                    }
                ]
            }
        ]
        mock_paginator.paginate.return_value = mock_page_iterator
        
        store = S3Store(self.config)
        
        size = store.estimate_download_size("s3://test-bucket/models/t2v")
        
        assert size == 1500

    @patch('boto3.client')
    def test_estimate_download_size_with_patterns(self, mock_boto3_client):
        """Test download size estimation with allow patterns."""
        mock_client = Mock()
        mock_boto3_client.return_value = mock_client
        
        # Mock paginator for listing objects
        mock_paginator = Mock()
        mock_client.get_paginator.return_value = mock_paginator
        
        mock_page_iterator = [
            {
                'Contents': [
                    {
                        'Key': 'models/t2v/model.safetensors',
                        'Size': 1000,
                        'ETag': '"abc123"',
                        'LastModified': '2024-01-01T00:00:00Z'
                    },
                    {
                        'Key': 'models/t2v/config.json',
                        'Size': 500,
                        'ETag': '"def456"',
                        'LastModified': '2024-01-01T00:00:00Z'
                    },
                    {
                        'Key': 'models/t2v/readme.txt',
                        'Size': 100,
                        'ETag': '"ghi789"',
                        'LastModified': '2024-01-01T00:00:00Z'
                    }
                ]
            }
        ]
        mock_paginator.paginate.return_value = mock_page_iterator
        
        store = S3Store(self.config)
        
        # Only .safetensors files should be counted
        size = store.estimate_download_size(
            "s3://test-bucket/models/t2v",
            allow_patterns=["*.safetensors"]
        )
        
        assert size == 1000

    def test_config_from_environment(self):
        """Test configuration from environment variables."""
        with patch.dict(os.environ, {
            'AWS_ENDPOINT_URL': 'http://env-endpoint:9000',
            'AWS_ACCESS_KEY_ID': 'env_key',
            'AWS_SECRET_ACCESS_KEY': 'env_secret',
            'AWS_DEFAULT_REGION': 'eu-west-1'
        }):
            store = S3Store()  # No config provided
            
            assert store.config.endpoint_url == 'http://env-endpoint:9000'
            assert store.config.access_key_id == 'env_key'
            assert store.config.secret_access_key == 'env_secret'
            assert store.config.region_name == 'eu-west-1'

    @patch('boto3.client')
    def test_thread_safety(self, mock_boto3_client):
        """Test that client initialization is thread-safe."""
        mock_client = Mock()
        mock_boto3_client.return_value = mock_client
        
        store = S3Store(self.config)
        clients = []
        
        def get_client():
            clients.append(store._get_client())
        
        # Create multiple threads trying to get client simultaneously
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=get_client)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All threads should get the same client instance
        assert len(set(id(client) for client in clients if client)) <= 1
        # Client should only be created once
        assert mock_boto3_client.call_count == 1

    def test_missing_boto3_dependency(self):
        """Test handling of missing boto3 dependency."""
        # This test is more complex since we need to mock the import at the module level
        # For now, we'll skip this test since boto3 is installed
        pytest.skip("boto3 is installed, cannot test missing dependency scenario")


class TestS3Config:
    """Test suite for S3Config."""

    def test_default_config(self):
        """Test default configuration values."""
        config = S3Config()
        
        assert config.endpoint_url is None
        assert config.access_key_id is None
        assert config.secret_access_key is None
        assert config.region_name == "us-east-1"
        assert config.max_concurrent_downloads == 4
        assert config.chunk_size == 8 * 1024 * 1024
        assert config.max_retries == 3
        assert config.retry_backoff == 1.0

    def test_custom_config(self):
        """Test custom configuration values."""
        config = S3Config(
            endpoint_url="http://localhost:9000",
            access_key_id="test_key",
            secret_access_key="test_secret",
            region_name="eu-west-1",
            max_concurrent_downloads=8,
            chunk_size=1024,
            max_retries=5,
            retry_backoff=2.0
        )
        
        assert config.endpoint_url == "http://localhost:9000"
        assert config.access_key_id == "test_key"
        assert config.secret_access_key == "test_secret"
        assert config.region_name == "eu-west-1"
        assert config.max_concurrent_downloads == 8
        assert config.chunk_size == 1024
        assert config.max_retries == 5
        assert config.retry_backoff == 2.0