#!/usr/bin/env python3
"""
Simple test to verify error messaging system functionality
"""

from error_messaging_system import (
    ErrorMessageGenerator,
    ErrorContext,
    ErrorSeverity,
    ErrorCategory,
    create_architecture_error,
    create_pipeline_error
)

def test_basic_functionality():
    """Test basic error message generation"""
    print("Testing basic error message generation...")
    
    # Test error message generator
    generator = ErrorMessageGenerator()
    
    context = ErrorContext(
        model_path="/path/to/wan/model",
        model_name="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        pipeline_class="WanPipeline",
        attempted_operation="pipeline_loading"
    )
    
    error_msg = generator.generate_error_message("missing_pipeline_class", context)
    
    print(f"✓ Generated error message: {error_msg.title}")
    print(f"✓ Error category: {error_msg.category}")
    print(f"✓ Error severity: {error_msg.severity}")
    print(f"✓ Recovery actions: {len(error_msg.recovery_actions)}")
    
    # Test convenience functions
    print("\nTesting convenience functions...")
    
    arch_error = create_architecture_error("/path/to/model", "corrupted_model_index")
    print("✓ Architecture error created")
    
    pipeline_error = create_pipeline_error("/path/to/model", "WanPipeline", "missing_pipeline_class")
    print("✓ Pipeline error created")
    
    print("\n✅ All basic tests passed!")

def test_error_requirements():
    """Test specific requirements coverage"""
    print("\nTesting requirements coverage...")
    
    generator = ErrorMessageGenerator()
    
    # Test Requirement 1.4: Clear instructions for obtaining pipeline code
    context = ErrorContext(
        model_path="/path/to/wan/model",
        pipeline_class="WanPipeline"
    )
    
    error_msg = generator.generate_error_message("missing_pipeline_class", context)
    
    action_titles = [action.title for action in error_msg.recovery_actions]
    assert "Install WanPipeline Package" in action_titles, "Missing installation instruction"
    print("✓ Requirement 1.4: Clear installation instructions provided")
    
    # Test Requirement 2.4: Specific VAE error messages
    vae_context = ErrorContext(
        model_path="/path/to/wan/model",
        model_name="Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    )
    
    vae_error = generator.generate_error_message("vae_shape_mismatch", vae_context)
    assert "3D architecture" in vae_error.detailed_description, "Missing VAE architecture details"
    print("✓ Requirement 2.4: Specific VAE error messages provided")
    
    # Test Requirement 6.4: Local installation alternatives
    remote_context = ErrorContext(
        model_name="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        pipeline_class="WanPipeline"
    )
    
    remote_error = generator.generate_error_message("remote_code_blocked", remote_context)
    remote_actions = [action.title for action in remote_error.recovery_actions]
    assert "Install Pipeline Locally" in remote_actions, "Missing local installation option"
    print("✓ Requirement 6.4: Local installation alternatives provided")
    
    # Test Requirement 7.4: Clear installation guidance for encoding
    video_context = ErrorContext(
        user_inputs={"output_path": "/path/to/video.mp4"},
        attempted_operation="video_encoding"
    )
    
    video_error = generator.generate_error_message("encoding_failed", video_context)
    video_actions = [action.title for action in video_error.recovery_actions]
    ffmpeg_actions = [title for title in video_actions if "FFmpeg" in title]
    assert len(ffmpeg_actions) > 0, "Missing FFmpeg installation guidance"
    print("✓ Requirement 7.4: Clear encoding dependency guidance provided")
    
    print("\n✅ All requirements tests passed!")

def test_progressive_disclosure():
    """Test progressive error disclosure"""
    print("\nTesting progressive error disclosure...")
    
    from error_messaging_system import ErrorMessageFormatter, ErrorMessage, RecoveryAction
    
    formatter = ErrorMessageFormatter()
    
    sample_error = ErrorMessage(
        title="Test Error",
        summary="Basic error summary",
        severity=ErrorSeverity.ERROR,
        category=ErrorCategory.PIPELINE_LOADING,
        detailed_description="Detailed explanation of the error",
        technical_details="Technical details for developers",
        debug_info={"debug": "information"},
        recovery_actions=[
            RecoveryAction(
                title="Basic Fix",
                description="Simple fix description",
                priority=1
            )
        ]
    )
    
    # Test basic level
    basic_output = formatter.format_for_console(sample_error, "basic")
    assert "Test Error" in basic_output, "Missing basic title"
    assert "Detailed explanation" not in basic_output, "Should not show detailed info in basic mode"
    print("✓ Basic disclosure level working")
    
    # Test detailed level
    detailed_output = formatter.format_for_console(sample_error, "detailed")
    assert "Detailed explanation of the error" in detailed_output, "Missing detailed info"
    assert "Technical details for developers" not in detailed_output, "Should not show technical details in detailed mode"
    print("✓ Detailed disclosure level working")
    
    # Test diagnostic level
    diagnostic_output = formatter.format_for_console(sample_error, "diagnostic")
    assert "Technical details for developers" in diagnostic_output, "Missing technical details"
    assert "Debug Information" in diagnostic_output, "Missing debug information"
    print("✓ Diagnostic disclosure level working")
    
    print("\n✅ Progressive disclosure tests passed!")

if __name__ == "__main__":
    test_basic_functionality()
    test_error_requirements()
    test_progressive_disclosure()
    print("\n🎉 All error messaging system tests completed successfully!")