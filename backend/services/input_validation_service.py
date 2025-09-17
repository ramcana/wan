import re
import html
from typing import Any, Dict, List
from pathlib import Path


class InputValidationService:
    def __init__(self):
        # Allowed HTML tags for rich text inputs (if any)
        self.allowed_tags = []
        self.allowed_attributes = {}

        # File type validation
        self.allowed_image_types = {
            "image/jpeg": [b"\xff\xd8\xff"],
            "image/png": [b"\x89\x50\x4e\x47"],
            "image/webp": [b"\x52\x49\x46\x46"],
        }

        # Content filters
        self.prohibited_content_patterns = [
            r"\b(?:nude|naked|nsfw|porn|sexual|erotic)\b",
            r"\b(?:violence|blood|gore|death|kill|murder)\b",
            r"\b(?:hate|racist|discrimination|nazi|terrorist)\b",
            r"\b(?:child|minor|kid|teen).*(?:nude|sexual|porn)\b",
            r"(?:javascript|script|iframe|object|embed|form)",
        ]

    def sanitize_text(self, text: str, max_length: int = 1000) -> str:
        """Sanitize and clean text input"""
        if not text:
            return ""

        # Limit length
        text = text[:max_length]

        # HTML escape
        text = html.escape(text)

        # Remove potentially dangerous characters
        text = re.sub(r'[<>"\']', "", text)

        # Clean whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def validate_content_policy(self, text: str) -> tuple[bool, List[str]]:
        """Check if text violates content policy"""
        violations = []

        for pattern in self.prohibited_content_patterns:
            if re.search(pattern, text.lower(), re.IGNORECASE):
                violations.append(f"Content matches prohibited pattern: {pattern}")

        return len(violations) == 0, violations

    def validate_file_upload(
        self, file_content: bytes, filename: str, content_type: str
    ) -> tuple[bool, List[str]]:
        """Validate uploaded file"""
        errors = []

        # Check file size (10MB max)
        if len(file_content) > 10 * 1024 * 1024:
            errors.append("File size exceeds 10MB limit")

        # Validate file extension
        file_ext = Path(filename).suffix.lower()
        allowed_extensions = [".jpg", ".jpeg", ".png", ".webp"]
        if file_ext not in allowed_extensions:
            errors.append(f"File extension {file_ext} not allowed")

        # Validate MIME type
        if content_type not in self.allowed_image_types:
            errors.append(f"Content type {content_type} not allowed")

        # Validate file signature (magic bytes)
        if not self._validate_file_signature(file_content, content_type):
            errors.append("File signature doesn't match content type")

        # Scan for embedded scripts or malicious content
        if self._contains_malicious_content(file_content):
            errors.append("File contains potentially malicious content")

        return len(errors) == 0, errors

    def _validate_file_signature(self, content: bytes, content_type: str) -> bool:
        """Check if file signature matches declared content type"""
        if content_type not in self.allowed_image_types:
            return False

        signatures = self.allowed_image_types[content_type]
        for signature in signatures:
            if content.startswith(signature):
                return True

        return False

    def _contains_malicious_content(self, content: bytes) -> bool:
        """Basic check for malicious content in files"""
        try:
            # Convert to string for pattern matching
            content_str = content.decode("utf-8", errors="ignore").lower()

            malicious_patterns = [
                r"<script",
                r"javascript:",
                r"vbscript:",
                r"onload=",
                r"onerror=",
                r"eval\(",
                r"document\.cookie",
            ]

            for pattern in malicious_patterns:
                if re.search(pattern, content_str):
                    return True

        except Exception:
            # If we can't decode, consider it suspicious
            return True

        return False

    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe storage"""
        if not filename:
            return "unknown_file"

        # Remove path components
        filename = Path(filename).name

        # Remove dangerous characters
        filename = re.sub(r"[^a-zA-Z0-9._-]", "_", filename)

        # Limit length
        name_part = Path(filename).stem[:50]
        ext_part = Path(filename).suffix[:10]

        return f"{name_part}{ext_part}"

    def validate_json_payload(
        self, payload: Dict[str, Any], max_depth: int = 5, max_size: int = 1024 * 1024
    ) -> tuple[bool, List[str]]:
        """Validate JSON payload for depth and size"""
        errors = []

        # Check payload size (serialized)
        try:
            import json

            payload_str = json.dumps(payload)
            if len(payload_str) > max_size:
                errors.append(
                    f"Payload size {len(payload_str)} exceeds limit {max_size}"
                )
        except Exception as e:
            errors.append(f"Payload serialization error: {str(e)}")

        # Check depth
        def check_depth(obj, current_depth=0):
            if current_depth > max_depth:
                return False

            if isinstance(obj, dict):
                return all(check_depth(v, current_depth + 1) for v in obj.values())
            elif isinstance(obj, list):
                return all(check_depth(item, current_depth + 1) for item in obj)

            return True

        if not check_depth(payload):
            errors.append(f"Payload depth exceeds limit {max_depth}")

        return len(errors) == 0, errors
