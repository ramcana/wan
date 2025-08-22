"""
Test suite for the comprehensive help text and guidance system
"""

import unittest
from unittest.mock import Mock, patch
import sys
import os

# Add the current directory to the path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from help_text_system import (
        HelpTextSystem, 
        get_help_system,
        get_model_help_text,
        get_image_help_text,
        get_tooltip_text,
        get_context_sensitive_help
    )
except ImportError as e:
    print(f"Warning: Could not import help_text_system: {e}")
    HelpTextSystem = None


class TestHelpTextSystem(unittest.TestCase):
    """Test the comprehensive help text system"""
    
    def setUp(self):
        """Set up test fixtures"""
        if HelpTextSystem is None:
            self.skipTest("help_text_system not available")
        
        self.help_system = HelpTextSystem()
    
    def test_model_help_text_content(self):
        """Test that model help text contains required information"""
        models = ["t2v-A14B", "i2v-A14B", "ti2v-5B"]
        
        for model in models:
            help_text = self.help_system.get_model_help_text(model)
            
            # Should have substantial content
            self.assertGreater(len(help_text), 100, f"Help text for {model} too short")
            
            # Should contain key information
            self.assertIn("Input", help_text, f"Missing input info for {model}")
            self.assertTrue("Resolution" in help_text or "resolution" in help_text, f"Missing resolution info for {model}")
            self.assertIn("VRAM", help_text, f"Missing VRAM info for {model}")
            self.assertIn("Tips", help_text, f"Missing tips for {model}")
            
            print(f"✅ {model} help text: {len(help_text)} characters")
    
    def test_mobile_help_text(self):
        """Test mobile-optimized help text"""
        models = ["t2v-A14B", "i2v-A14B", "ti2v-5B"]
        
        for model in models:
            desktop_help = self.help_system.get_model_help_text(model, mobile=False)
            mobile_help = self.help_system.get_model_help_text(model, mobile=True)
            
            # Mobile help should be shorter or same length
            self.assertLessEqual(len(mobile_help), len(desktop_help), 
                               f"Mobile help for {model} should be shorter")
            
            # Should still contain essential info
            self.assertGreater(len(mobile_help), 50, f"Mobile help for {model} too short")
            
            print(f"✅ {model} mobile help: {len(mobile_help)} characters")
    
    def test_image_help_text(self):
        """Test image upload help text"""
        # T2V should have no image help
        t2v_help = self.help_system.get_image_help_text("t2v-A14B")
        self.assertEqual(t2v_help, "", "T2V should have no image help")
        
        # I2V and TI2V should have image help
        for model in ["i2v-A14B", "ti2v-5B"]:
            help_text = self.help_system.get_image_help_text(model)
            
            self.assertGreater(len(help_text), 50, f"Image help for {model} too short")
            self.assertIn("image", help_text.lower(), f"Missing image info for {model}")
            
            print(f"✅ {model} image help: {len(help_text)} characters")
    
    def test_tooltip_text(self):
        """Test tooltip text for UI elements"""
        elements = [
            "model_type", "prompt", "resolution", "steps", 
            "duration", "fps", "image_upload_area", "clear_image"
        ]
        
        for element in elements:
            tooltip = self.help_system.get_tooltip_text(element)
            
            self.assertGreater(len(tooltip), 10, f"Tooltip for {element} too short")
            self.assertLess(len(tooltip), 200, f"Tooltip for {element} too long")
            
            print(f"✅ {element} tooltip: {len(tooltip)} characters")
    
    def test_image_upload_tooltips(self):
        """Test specific image upload tooltips"""
        for image_type in ["start_image", "end_image"]:
            tooltip = self.help_system.get_image_upload_tooltip(image_type)
            
            self.assertGreater(len(tooltip), 20, f"Tooltip for {image_type} too short")
            self.assertIn("image", tooltip.lower(), f"Missing image info in {image_type} tooltip")
            
            print(f"✅ {image_type} tooltip: {len(tooltip)} characters")
    
    def test_requirements_list(self):
        """Test requirements lists for different contexts"""
        contexts = ["t2v-A14B", "i2v-A14B", "ti2v-5B", "start_image", "end_image"]
        
        for context in contexts:
            requirements = self.help_system.get_requirements_list(context)
            
            self.assertIsInstance(requirements, list, f"Requirements for {context} should be a list")
            
            if context != "t2v-A14B":  # T2V might have fewer requirements
                self.assertGreater(len(requirements), 0, f"No requirements found for {context}")
            
            print(f"✅ {context} requirements: {len(requirements)} items")
    
    def test_examples_list(self):
        """Test examples lists for model types"""
        models = ["t2v-A14B", "i2v-A14B", "ti2v-5B"]
        
        for model in models:
            examples = self.help_system.get_examples_list(model)
            
            self.assertIsInstance(examples, list, f"Examples for {model} should be a list")
            self.assertGreater(len(examples), 0, f"No examples found for {model}")
            
            for example in examples:
                self.assertGreater(len(example), 10, f"Example too short for {model}")
            
            print(f"✅ {model} examples: {len(examples)} items")
    
    def test_error_help(self):
        """Test error help messages and suggestions"""
        error_types = ["invalid_format", "too_small", "aspect_mismatch", "file_too_large"]
        
        for error_type in error_types:
            error_help = self.help_system.get_error_help(error_type)
            
            self.assertIn("message", error_help, f"Missing message for {error_type}")
            self.assertIn("suggestions", error_help, f"Missing suggestions for {error_type}")
            
            self.assertGreater(len(error_help["message"]), 5, f"Message too short for {error_type}")
            self.assertGreater(len(error_help["suggestions"]), 0, f"No suggestions for {error_type}")
            
            print(f"✅ {error_type} error help: {len(error_help['suggestions'])} suggestions")
    
    def test_context_sensitive_help(self):
        """Test context-sensitive help that adapts to current state"""
        models = ["t2v-A14B", "i2v-A14B", "ti2v-5B"]
        
        for model in models:
            # Test different states
            states = [
                (False, False),  # No images
                (True, False),   # Start image only
                (True, True)     # Both images
            ]
            
            for has_start, has_end in states:
                help_text = self.help_system.get_context_sensitive_help(
                    model, has_start, has_end
                )
                
                self.assertGreater(len(help_text), 50, 
                                 f"Context help too short for {model} state ({has_start}, {has_end})")
                
                # Should contain model-specific info
                self.assertIn(model.split('-')[0].upper(), help_text, 
                            f"Missing model type in help for {model}")
                
                print(f"✅ {model} context help ({has_start}, {has_end}): {len(help_text)} characters")
    
    def test_html_formatting(self):
        """Test HTML formatting functionality"""
        test_content = "**Bold text** and *italic text* with\nnew lines"
        test_title = "Test Title"
        
        html = self.help_system.format_help_html(test_content, test_title)
        
        self.assertIn("<div class=", html, "Missing HTML wrapper")
        self.assertIn(test_title, html, "Missing title in HTML")
        self.assertIn("<strong>", html, "Bold formatting not applied")
        self.assertIn("<br>", html, "Line breaks not converted")
        
        print(f"✅ HTML formatting: {len(html)} characters")
    
    def test_responsive_css(self):
        """Test responsive CSS generation"""
        css = self.help_system.get_responsive_help_css()
        
        self.assertGreater(len(css), 500, "CSS too short")
        self.assertIn("@media", css, "Missing responsive media queries")
        self.assertIn(".help-content", css, "Missing help content styles")
        self.assertIn(".help-tooltip", css, "Missing tooltip styles")
        
        print(f"✅ Responsive CSS: {len(css)} characters")
    
    def test_tooltip_html_creation(self):
        """Test tooltip HTML creation"""
        text = "Hover me"
        tooltip = "This is a tooltip"
        
        html = self.help_system.create_tooltip_html(text, tooltip)
        
        self.assertIn(text, html, "Missing main text")
        self.assertIn(tooltip, html, "Missing tooltip text")
        self.assertIn("help-tooltip", html, "Missing tooltip class")
        
        print(f"✅ Tooltip HTML: {len(html)} characters")


class TestHelpSystemIntegration(unittest.TestCase):
    """Test help system integration functions"""
    
    def test_global_functions(self):
        """Test global convenience functions"""
        if HelpTextSystem is None:
            self.skipTest("help_text_system not available")
        
        # Test model help
        help_text = get_model_help_text("t2v-A14B")
        self.assertGreater(len(help_text), 50, "Global model help function failed")
        
        # Test image help
        image_help = get_image_help_text("i2v-A14B")
        self.assertGreater(len(image_help), 20, "Global image help function failed")
        
        # Test tooltip
        tooltip = get_tooltip_text("model_type")
        self.assertGreater(len(tooltip), 10, "Global tooltip function failed")
        
        # Test context-sensitive help
        context_help = get_context_sensitive_help("ti2v-5B", True, False)
        self.assertGreater(len(context_help), 50, "Global context help function failed")
        
        print("✅ All global functions working")
    
    def test_help_system_singleton(self):
        """Test that help system uses singleton pattern"""
        if HelpTextSystem is None:
            self.skipTest("help_text_system not available")
        
        system1 = get_help_system()
        system2 = get_help_system()
        
        self.assertIs(system1, system2, "Help system should be singleton")
        
        print("✅ Singleton pattern working")


class TestUIIntegration(unittest.TestCase):
    """Test UI integration with help system"""
    
    def test_ui_help_methods(self):
        """Test that UI can use help system methods"""
        try:
            from ui import Wan22UI
            
            # Mock the config loading to avoid file dependencies
            with patch.object(Wan22UI, '_load_config', return_value={}):
                with patch.object(Wan22UI, '_create_interface', return_value=Mock()):
                    with patch.object(Wan22UI, '_perform_startup_checks'):
                        with patch('ui.get_model_manager', return_value=Mock()):
                            with patch('ui.VRAMOptimizer', return_value=Mock()):
                                with patch('ui.get_performance_profiler', return_value=Mock()):
                                    with patch('ui.start_performance_monitoring'):
                                        ui_instance = Wan22UI()
                                        
                                        # Test help methods
                                        model_help = ui_instance._get_model_help_text("t2v-A14B")
                                        self.assertIsInstance(model_help, str, "Model help should be string")
                                        
                                        image_help = ui_instance._get_image_help_text("i2v-A14B")
                                        self.assertIsInstance(image_help, str, "Image help should be string")
                                        
                                        tooltip = ui_instance._get_tooltip_text("model_type")
                                        self.assertIsInstance(tooltip, str, "Tooltip should be string")
                                        
                                        context_help = ui_instance._get_context_sensitive_help("ti2v-5B")
                                        self.assertIsInstance(context_help, str, "Context help should be string")
                                        
                                        print("✅ UI integration methods working")
        
        except ImportError as e:
            self.skipTest(f"UI not available for testing: {e}")


def run_help_system_tests():
    """Run all help system tests"""
    print("🧪 Running Help Text System Tests...")
    print("=" * 50)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestHelpTextSystem))
    suite.addTests(loader.loadTestsFromTestCase(TestHelpSystemIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestUIIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print(f"📊 Test Summary:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    print(f"   Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\n❌ Failures:")
        for test, traceback in result.failures:
            error_msg = traceback.split('AssertionError: ')[-1].split('\n')[0]
            print(f"   - {test}: {error_msg}")
    
    if result.errors:
        print(f"\n🚨 Errors:")
        for test, traceback in result.errors:
            error_msg = traceback.split('\n')[-2]
            print(f"   - {test}: {error_msg}")
    
    if not result.failures and not result.errors:
        print("\n🎉 All tests passed!")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_help_system_tests()
    sys.exit(0 if success else 1)