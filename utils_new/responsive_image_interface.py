"""
Responsive Image Upload Interface for Wan2.2 UI
Implements responsive design for image upload components with mobile-first approach
"""

import gradio as gr
from typing import Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class ResponsiveImageInterface:
    """Handles responsive design for image upload interface"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.breakpoints = {
            'mobile': 768,
            'tablet': 1024,
            'desktop': 1200
        }
    
    def get_responsive_css(self) -> str:
        """Generate comprehensive responsive CSS for image upload interface"""
        return """
        /* Base responsive image upload styles */
        .image-upload-container {
            width: 100%;
            margin: 10px 0;
        }
        
        .image-inputs-row {
            display: flex;
            gap: 20px;
            width: 100%;
            align-items: flex-start;
        }
        
        .image-column {
            flex: 1;
            min-width: 0; /* Prevent flex item overflow */
        }
        
        /* Desktop layout - side by side */
        @media (min-width: 769px) {
            .image-inputs-row {
                flex-direction: row;
                gap: 30px;
            }
            
            .image-column {
                flex: 1;
                max-width: 50%;
            }
            
            .image-upload-area {
                min-height: 200px;
            }
            
            .image-preview-container {
                max-width: 300px;
            }
            
            .image-preview-thumbnail {
                max-width: 150px;
                max-height: 150px;
            }
            
            .image-requirements-help {
                font-size: 0.85em;
                padding: 12px;
                margin: 10px 0;
            }
            
            .help-content {
                font-size: 0.9em;
                padding: 15px;
            }
            
            .validation-message {
                font-size: 0.9em;
                padding: 10px;
            }
        }
        
        /* Tablet layout - side by side with reduced spacing */
        @media (min-width: 769px) and (max-width: 1024px) {
            .image-inputs-row {
                gap: 20px;
            }
            
            .image-column {
                flex: 1;
            }
            
            .image-upload-area {
                min-height: 180px;
            }
            
            .image-preview-container {
                max-width: 250px;
            }
            
            .image-preview-thumbnail {
                max-width: 120px;
                max-height: 120px;
            }
            
            .image-requirements-help {
                font-size: 0.8em;
                padding: 10px;
            }
            
            .help-content {
                font-size: 0.85em;
                padding: 12px;
            }
        }
        
        /* Mobile layout - stacked vertically */
        @media (max-width: 768px) {
            .image-inputs-row {
                flex-direction: column !important;
                gap: 20px;
            }
            
            .image-column {
                width: 100% !important;
                max-width: none;
                margin-bottom: 20px;
            }
            
            .image-upload-area {
                min-height: 150px;
                padding: 15px;
            }
            
            .image-preview-container {
                max-width: 100%;
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 10px;
            }
            
            .image-preview-thumbnail {
                max-width: 100px;
                max-height: 100px;
            }
            
            .image-requirements-help {
                font-size: 0.75em;
                padding: 8px;
                margin: 8px 0;
                line-height: 1.3;
            }
            
            .help-content {
                font-size: 0.8em;
                padding: 10px;
                line-height: 1.4;
            }
            
            .help-title {
                font-size: 0.9em;
                margin-bottom: 6px;
            }
            
            .validation-message {
                font-size: 0.8em;
                padding: 8px;
                margin: 5px 0;
            }
            
            /* Mobile-specific image preview adjustments */
            .image-preview-info {
                text-align: center;
                font-size: 0.75em;
            }
            
            .image-preview-actions {
                display: flex;
                justify-content: center;
                gap: 10px;
                margin-top: 10px;
            }
            
            .image-preview-actions button {
                font-size: 0.8em;
                padding: 6px 12px;
            }
        }
        
        /* Extra small mobile devices */
        @media (max-width: 480px) {
            .image-upload-area {
                min-height: 120px;
                padding: 10px;
            }
            
            .image-preview-thumbnail {
                max-width: 80px;
                max-height: 80px;
            }
            
            .image-requirements-help {
                font-size: 0.7em;
                padding: 6px;
            }
            
            .help-content {
                font-size: 0.75em;
                padding: 8px;
            }
            
            .validation-message {
                font-size: 0.75em;
                padding: 6px;
            }
        }
        
        /* Responsive image upload drag and drop areas */
        .image-upload-dropzone {
            border: 2px dashed #007bff;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            background: #f8f9fa;
            transition: all 0.3s ease;
            cursor: pointer;
            position: relative;
            overflow: hidden;
        }
        
        .image-upload-dropzone:hover {
            background: #e3f2fd;
            border-color: #0056b3;
            transform: translateY(-1px);
        }
        
        .image-upload-dropzone.dragover {
            background: #cce5ff;
            border-color: #004085;
            box-shadow: 0 4px 12px rgba(0,123,255,0.3);
        }
        
        @media (max-width: 768px) {
            .image-upload-dropzone {
                padding: 15px;
            }
            
            .image-upload-dropzone:hover {
                transform: none; /* Disable hover transform on mobile */
            }
        }
        
        /* Responsive validation messages */
        .validation-success-mobile {
            background: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
            padding: 8px;
            border-radius: 4px;
            margin: 5px 0;
            font-size: 0.8em;
            text-align: center;
        }
        
        .validation-error-mobile {
            background: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
            padding: 8px;
            border-radius: 4px;
            margin: 5px 0;
            font-size: 0.8em;
            text-align: center;
        }
        
        @media (min-width: 769px) {
            .validation-success-mobile {
                text-align: left;
                font-size: 0.9em;
                padding: 10px;
            }
            
            .validation-error-mobile {
                text-align: left;
                font-size: 0.9em;
                padding: 10px;
            }
        }
        
        /* Responsive tooltips */
        .responsive-tooltip {
            position: relative;
            display: inline-block;
            cursor: help;
        }
        
        .responsive-tooltip .tooltip-content {
            visibility: hidden;
            background-color: #333;
            color: #fff;
            text-align: left;
            border-radius: 6px;
            padding: 10px;
            position: absolute;
            z-index: 1000;
            opacity: 0;
            transition: opacity 0.3s;
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
            font-size: 0.8em;
            line-height: 1.3;
            max-width: 250px;
            word-wrap: break-word;
        }
        
        .responsive-tooltip:hover .tooltip-content {
            visibility: visible;
            opacity: 1;
        }
        
        /* Desktop tooltip positioning */
        @media (min-width: 769px) {
            .responsive-tooltip .tooltip-content {
                bottom: 125%;
                left: 50%;
                margin-left: -125px;
                width: 250px;
            }
            
            .responsive-tooltip .tooltip-content::after {
                content: "";
                position: absolute;
                top: 100%;
                left: 50%;
                margin-left: -5px;
                border-width: 5px;
                border-style: solid;
                border-color: #333 transparent transparent transparent;
            }
        }
        
        /* Mobile tooltip positioning */
        @media (max-width: 768px) {
            .responsive-tooltip .tooltip-content {
                bottom: 125%;
                left: 50%;
                margin-left: -100px;
                width: 200px;
                font-size: 0.7em;
                padding: 8px;
            }
            
            .responsive-tooltip .tooltip-content::after {
                content: "";
                position: absolute;
                top: 100%;
                left: 50%;
                margin-left: -5px;
                border-width: 4px;
                border-style: solid;
                border-color: #333 transparent transparent transparent;
            }
        }
        
        /* Responsive image preview animations */
        .image-preview-fade-in {
            animation: fadeInScale 0.3s ease-out;
        }
        
        @keyframes fadeInScale {
            from {
                opacity: 0;
                transform: scale(0.9);
            }
            to {
                opacity: 1;
                transform: scale(1);
            }
        }
        
        .image-preview-slide-up {
            animation: slideUpFade 0.4s ease-out;
        }
        
        @keyframes slideUpFade {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        /* Responsive loading states */
        .loading-spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 2px solid #f3f3f3;
            border-top: 2px solid #007bff;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        @media (max-width: 768px) {
            .loading-spinner {
                width: 16px;
                height: 16px;
                border-width: 1.5px;
            }
        }
        
        /* Responsive button styles */
        .responsive-button {
            padding: 8px 16px;
            font-size: 0.9em;
            border-radius: 4px;
            border: 1px solid #007bff;
            background: #007bff;
            color: white;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        
        .responsive-button:hover {
            background: #0056b3;
            border-color: #0056b3;
            transform: translateY(-1px);
        }
        
        .responsive-button.secondary {
            background: transparent;
            color: #007bff;
        }
        
        .responsive-button.secondary:hover {
            background: #007bff;
            color: white;
        }
        
        @media (max-width: 768px) {
            .responsive-button {
                padding: 6px 12px;
                font-size: 0.8em;
            }
            
            .responsive-button:hover {
                transform: none; /* Disable hover transform on mobile */
            }
        }
        
        /* Responsive grid layouts */
        .responsive-grid {
            display: grid;
            gap: 20px;
            width: 100%;
        }
        
        @media (min-width: 769px) {
            .responsive-grid.two-column {
                grid-template-columns: 1fr 1fr;
            }
        }
        
        @media (max-width: 768px) {
            .responsive-grid {
                grid-template-columns: 1fr;
                gap: 15px;
            }
        }
        
        /* Accessibility improvements */
        @media (prefers-reduced-motion: reduce) {
            .image-preview-fade-in,
            .image-preview-slide-up,
            .loading-spinner,
            .responsive-button,
            .image-upload-dropzone {
                animation: none;
                transition: none;
            }
        }
        
        /* High contrast mode support */
        @media (prefers-contrast: high) {
            .image-upload-dropzone {
                border-width: 3px;
                border-color: #000;
            }
            
            .validation-success-mobile {
                border-width: 2px;
                border-color: #000;
            }
            
            .validation-error-mobile {
                border-width: 2px;
                border-color: #000;
            }
        }
        
        /* Touch interaction styles */
        .touch-active {
            background: #e3f2fd !important;
            transform: scale(0.98);
            transition: all 0.1s ease;
        }
        """
    
    def create_responsive_image_row(self, visible: bool = False) -> Tuple[gr.Row, Dict[str, Any]]:
        """Create responsive image upload row with proper mobile/desktop layout"""
        
        components = {}
        
        with gr.Row(visible=visible, elem_classes=["image-inputs-row", "responsive-grid", "two-column"]) as image_row:
            # Start image column
            with gr.Column(elem_classes=["image-column"]) as start_column:
                # Start image upload
                start_image = gr.Image(
                    label="📸 Start Frame Image",
                    type="pil",
                    interactive=True,
                    elem_id="start_image_upload",
                    elem_classes=["image-upload-area"]
                )
                
                # Start image preview container
                start_preview = gr.HTML(
                    value="",
                    visible=False,
                    elem_classes=["image-preview-container", "image-preview-fade-in"]
                )
                
                # Start image requirements (responsive)
                start_requirements = gr.Markdown(
                    value=self._get_responsive_requirements_text("start"),
                    visible=False,
                    elem_classes=["image-requirements-help"]
                )
                
                # Start image validation
                start_validation = gr.HTML(
                    value="",
                    visible=False,
                    elem_classes=["validation-message", "validation-success-mobile"]
                )
            
            # End image column
            with gr.Column(elem_classes=["image-column"]) as end_column:
                # End image upload
                end_image = gr.Image(
                    label="🎯 End Frame Image (Optional)",
                    type="pil",
                    interactive=True,
                    elem_id="end_image_upload",
                    elem_classes=["image-upload-area"]
                )
                
                # End image preview container
                end_preview = gr.HTML(
                    value="",
                    visible=False,
                    elem_classes=["image-preview-container", "image-preview-fade-in"]
                )
                
                # End image requirements (responsive)
                end_requirements = gr.Markdown(
                    value=self._get_responsive_requirements_text("end"),
                    visible=False,
                    elem_classes=["image-requirements-help"]
                )
                
                # End image validation
                end_validation = gr.HTML(
                    value="",
                    visible=False,
                    elem_classes=["validation-message", "validation-success-mobile"]
                )
        
        components = {
            'image_row': image_row,
            'start_column': start_column,
            'end_column': end_column,
            'start_image': start_image,
            'end_image': end_image,
            'start_preview': start_preview,
            'end_preview': end_preview,
            'start_requirements': start_requirements,
            'end_requirements': end_requirements,
            'start_validation': start_validation,
            'end_validation': end_validation
        }
        
        return image_row, components
    
    def _get_responsive_requirements_text(self, image_type: str) -> str:
        """Get responsive requirements text that adapts to screen size"""
        
        if image_type == "start":
            desktop_text = """
**📸 Start Image Requirements:**

• **Format:** PNG, JPG, JPEG, WebP
• **Min Size:** 256×256 pixels  
• **Recommended:** 1280×720 or higher
• **Purpose:** Defines the first frame of your video
• **Quality:** Higher resolution images produce better results
• **Content:** Clear, well-lit subjects work best

*Tip: The start image becomes the opening frame of your generated video*
            """
            
            mobile_text = """
**📸 Start Image:**

• Format: PNG, JPG, JPEG, WebP
• Min: 256×256 pixels
• Purpose: First video frame
• Tip: Higher resolution = better results
            """
        else:  # end image
            desktop_text = """
**🎯 End Image Requirements:**

• **Format:** PNG, JPG, JPEG, WebP
• **Min Size:** 256×256 pixels
• **Aspect:** Should match start image
• **Purpose:** Defines the final frame (optional)
• **Compatibility:** Best results when aspect ratio matches start image
• **Content:** Should relate to start image for smooth transitions

*Note: End image helps guide the video's conclusion and improves coherence*
            """
            
            mobile_text = """
**🎯 End Image:**

• Format: PNG, JPG, JPEG, WebP
• Min: 256×256 pixels
• Purpose: Final frame (optional)
• Tip: Match start image aspect ratio
            """
        
        return f"""
        <div class="responsive-requirements">
            <div class="desktop-requirements" style="display: block;">
                {desktop_text}
            </div>
            <div class="mobile-requirements" style="display: none;">
                {mobile_text}
            </div>
        </div>
        
        <script>
        // Show appropriate requirements text based on screen size
        function updateRequirementsText() {{
            const desktopReqs = document.querySelectorAll('.desktop-requirements');
            const mobileReqs = document.querySelectorAll('.mobile-requirements');
            
            if (window.innerWidth <= 768) {{
                desktopReqs.forEach(el => el.style.display = 'none');
                mobileReqs.forEach(el => el.style.display = 'block');
            }} else {{
                desktopReqs.forEach(el => el.style.display = 'block');
                mobileReqs.forEach(el => el.style.display = 'none');
            }}
        }}
        
        // Update on load and resize
        updateRequirementsText();
        window.addEventListener('resize', updateRequirementsText);
        </script>
        """
    
    def create_responsive_help_text(self, model_type: str) -> str:
        """Create responsive help text that adapts to different screen sizes"""
        
        help_texts = {
            "t2v-A14B": {
                "title": "Text-to-Video Generation",
                "desktop": """
                **T2V Mode:** Generate videos from text descriptions only.
                
                • No images required - pure text-to-video generation
                • Focus on detailed, descriptive prompts
                • Supports all standard resolutions
                • Best for creative, imaginative content
                """,
                "mobile": """
                **T2V Mode:** Text-only video generation.
                
                • No images needed
                • Use detailed prompts
                • All resolutions supported
                """
            },
            "i2v-A14B": {
                "title": "Image-to-Video Generation", 
                "desktop": """
                **I2V Mode:** Generate videos starting from your uploaded image.
                
                • Start image required - becomes first frame
                • End image optional - guides video conclusion
                • Prompt enhances the transformation
                • Best for bringing static images to life
                """,
                "mobile": """
                **I2V Mode:** Video from your images.
                
                • Start image required
                • End image optional
                • Prompt enhances result
                """
            },
            "ti2v-5B": {
                "title": "Text+Image-to-Video Generation",
                "desktop": """
                **TI2V Mode:** Combine text prompts with uploaded images.
                
                • Start image required - provides visual context
                • End image optional - defines target outcome
                • Text prompt guides the transformation
                • Best balance of control and creativity
                """,
                "mobile": """
                **TI2V Mode:** Text + image generation.
                
                • Start image required
                • Text prompt guides video
                • End image optional
                """
            }
        }
        
        model_help = help_texts.get(model_type, help_texts["t2v-A14B"])
        
        return f"""
        <div class="help-content responsive-help">
            <div class="help-title">{model_help['title']}</div>
            <div class="help-text desktop-help" style="display: block;">
                {model_help['desktop']}
            </div>
            <div class="help-text mobile-help" style="display: none;">
                {model_help['mobile']}
            </div>
        </div>
        
        <script>
        // Show appropriate help text based on screen size
        function updateHelpText() {{
            const desktopHelp = document.querySelectorAll('.desktop-help');
            const mobileHelp = document.querySelectorAll('.mobile-help');
            
            if (window.innerWidth <= 768) {{
                desktopHelp.forEach(el => el.style.display = 'none');
                mobileHelp.forEach(el => el.style.display = 'block');
            }} else {{
                desktopHelp.forEach(el => el.style.display = 'block');
                mobileHelp.forEach(el => el.style.display = 'none');
            }}
        }}
        
        // Update on load and resize
        updateHelpText();
        window.addEventListener('resize', updateHelpText);
        </script>
        """
    
    def create_responsive_validation_message(self, 
                                           message: str, 
                                           is_success: bool = True,
                                           details: Optional[Dict[str, Any]] = None) -> str:
        """Create responsive validation message that adapts to screen size"""
        
        message_class = "validation-success-mobile" if is_success else "validation-error-mobile"
        icon = "✅" if is_success else "❌"
        
        details_html = ""
        if details:
            # Desktop details
            desktop_details = []
            for key, value in details.items():
                desktop_details.append(f"<strong>{key.replace('_', ' ').title()}:</strong> {value}")
            
            # Mobile details (simplified)
            mobile_details = []
            for key, value in details.items():
                if key in ['dimensions', 'format', 'file_size']:  # Show only key info on mobile
                    mobile_details.append(f"{key.replace('_', ' ').title()}: {value}")
            
            details_html = f"""
            <div class="validation-details">
                <div class="desktop-details" style="display: block; margin-top: 8px; font-size: 0.85em;">
                    {' | '.join(desktop_details)}
                </div>
                <div class="mobile-details" style="display: none; margin-top: 5px; font-size: 0.75em;">
                    {' | '.join(mobile_details)}
                </div>
            </div>
            """
        
        return f"""
        <div class="{message_class} responsive-validation">
            <div class="validation-message-content">
                {icon} {message}
            </div>
            {details_html}
        </div>
        
        <script>
        // Show appropriate validation details based on screen size
        function updateValidationDetails() {{
            const desktopDetails = document.querySelectorAll('.desktop-details');
            const mobileDetails = document.querySelectorAll('.mobile-details');
            
            if (window.innerWidth <= 768) {{
                desktopDetails.forEach(el => el.style.display = 'none');
                mobileDetails.forEach(el => el.style.display = 'block');
            }} else {{
                desktopDetails.forEach(el => el.style.display = 'block');
                mobileDetails.forEach(el => el.style.display = 'none');
            }}
        }}
        
        updateValidationDetails();
        window.addEventListener('resize', updateValidationDetails);
        </script>
        """
    
    def create_responsive_image_preview(self, 
                                      image_data: Any,
                                      image_type: str,
                                      metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create responsive image preview that adapts to screen size"""
        
        if not image_data:
            return ""
        
        # Generate thumbnail data URL (simplified for demo)
        thumbnail_data = f"data:image/jpeg;base64,{image_data}"  # Placeholder
        
        # Metadata display
        metadata_html = ""
        if metadata:
            # Desktop metadata
            desktop_meta = f"""
            <div class="image-metadata desktop-metadata">
                <div><strong>Dimensions:</strong> {metadata.get('dimensions', 'Unknown')}</div>
                <div><strong>Format:</strong> {metadata.get('format', 'Unknown')}</div>
                <div><strong>Size:</strong> {metadata.get('file_size', 'Unknown')}</div>
                <div><strong>Aspect Ratio:</strong> {metadata.get('aspect_ratio', 'Unknown')}</div>
            </div>
            """
            
            # Mobile metadata (simplified)
            mobile_meta = f"""
            <div class="image-metadata mobile-metadata">
                <div>{metadata.get('dimensions', 'Unknown')} • {metadata.get('format', 'Unknown')}</div>
                <div>{metadata.get('file_size', 'Unknown')}</div>
            </div>
            """
            
            metadata_html = desktop_meta + mobile_meta
        
        preview_html = f"""
        <div class="image-preview-container responsive-preview" data-image-type="{image_type}">
            <div class="image-preview-content">
                <div class="image-thumbnail-container">
                    <img src="{thumbnail_data}" 
                         alt="{image_type.title()} Image Preview"
                         class="image-preview-thumbnail"
                         onclick="showLargePreview('{image_type}', '{thumbnail_data}')" />
                </div>
                
                <div class="image-info-container">
                    {metadata_html}
                    
                    <div class="image-preview-actions">
                        <button class="responsive-button secondary" 
                                onclick="clearImage('{image_type}')">
                            🗑️ Clear
                        </button>
                        <button class="responsive-button" 
                                onclick="showLargePreview('{image_type}', '{thumbnail_data}')">
                            🔍 View
                        </button>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
        // Show appropriate metadata based on screen size
        function updateImageMetadata() {{
            const desktopMeta = document.querySelectorAll('.desktop-metadata');
            const mobileMeta = document.querySelectorAll('.mobile-metadata');
            
            if (window.innerWidth <= 768) {{
                desktopMeta.forEach(el => el.style.display = 'none');
                mobileMeta.forEach(el => el.style.display = 'block');
            }} else {{
                desktopMeta.forEach(el => el.style.display = 'block');
                mobileMeta.forEach(el => el.style.display = 'none');
            }}
        }}
        
        updateImageMetadata();
        window.addEventListener('resize', updateImageMetadata);
        </script>
        """
        
        return preview_html
    
    def get_responsive_javascript(self) -> str:
        """Get JavaScript for responsive functionality"""
        return """
        <script>
        // Responsive image interface functionality
        class ResponsiveImageInterface {
            constructor() {
                this.breakpoints = {
                    mobile: 768,
                    tablet: 1024,
                    desktop: 1200
                };
                
                this.currentBreakpoint = this.getCurrentBreakpoint();
                this.init();
            }
            
            init() {
                this.setupResizeListener();
                this.setupTouchHandlers();
                this.updateLayout();
            }
            
            getCurrentBreakpoint() {
                const width = window.innerWidth;
                if (width <= this.breakpoints.mobile) return 'mobile';
                if (width <= this.breakpoints.tablet) return 'tablet';
                return 'desktop';
            }
            
            setupResizeListener() {
                let resizeTimeout;
                window.addEventListener('resize', () => {
                    clearTimeout(resizeTimeout);
                    resizeTimeout = setTimeout(() => {
                        const newBreakpoint = this.getCurrentBreakpoint();
                        if (newBreakpoint !== this.currentBreakpoint) {
                            this.currentBreakpoint = newBreakpoint;
                            this.updateLayout();
                        }
                    }, 150);
                });
            }
            
            setupTouchHandlers() {
                // Enhanced touch handling for mobile devices
                const uploadAreas = document.querySelectorAll('.image-upload-area');
                uploadAreas.forEach(area => {
                    area.addEventListener('touchstart', this.handleTouchStart.bind(this), { passive: true });
                    area.addEventListener('touchend', this.handleTouchEnd.bind(this), { passive: true });
                });
            }
            
            handleTouchStart(event) {
                event.target.classList.add('touch-active');
            }
            
            handleTouchEnd(event) {
                setTimeout(() => {
                    event.target.classList.remove('touch-active');
                }, 150);
            }
            
            updateLayout() {
                this.updateHelpText();
                this.updateValidationMessages();
                this.updateImagePreviews();
                this.updateTooltips();
            }
            
            updateHelpText() {
                const desktopHelp = document.querySelectorAll('.desktop-help');
                const mobileHelp = document.querySelectorAll('.mobile-help');
                
                if (this.currentBreakpoint === 'mobile') {
                    desktopHelp.forEach(el => el.style.display = 'none');
                    mobileHelp.forEach(el => el.style.display = 'block');
                } else {
                    desktopHelp.forEach(el => el.style.display = 'block');
                    mobileHelp.forEach(el => el.style.display = 'none');
                }
            }
            
            updateValidationMessages() {
                const desktopDetails = document.querySelectorAll('.desktop-details');
                const mobileDetails = document.querySelectorAll('.mobile-details');
                
                if (this.currentBreakpoint === 'mobile') {
                    desktopDetails.forEach(el => el.style.display = 'none');
                    mobileDetails.forEach(el => el.style.display = 'block');
                } else {
                    desktopDetails.forEach(el => el.style.display = 'block');
                    mobileDetails.forEach(el => el.style.display = 'none');
                }
            }
            
            updateImagePreviews() {
                const desktopMeta = document.querySelectorAll('.desktop-metadata');
                const mobileMeta = document.querySelectorAll('.mobile-metadata');
                
                if (this.currentBreakpoint === 'mobile') {
                    desktopMeta.forEach(el => el.style.display = 'none');
                    mobileMeta.forEach(el => el.style.display = 'block');
                } else {
                    desktopMeta.forEach(el => el.style.display = 'block');
                    mobileMeta.forEach(el => el.style.display = 'none');
                }
            }
            
            updateTooltips() {
                const tooltips = document.querySelectorAll('.responsive-tooltip');
                tooltips.forEach(tooltip => {
                    const content = tooltip.querySelector('.tooltip-content');
                    if (content) {
                        if (this.currentBreakpoint === 'mobile') {
                            content.style.width = '200px';
                            content.style.marginLeft = '-100px';
                            content.style.fontSize = '0.7em';
                            content.style.padding = '8px';
                        } else {
                            content.style.width = '250px';
                            content.style.marginLeft = '-125px';
                            content.style.fontSize = '0.8em';
                            content.style.padding = '10px';
                        }
                    }
                });
            }
            
            // Enhanced image preview modal for mobile
            showLargePreview(imageType, thumbnailData) {
                const modal = document.createElement('div');
                modal.className = 'image-preview-modal';
                modal.style.cssText = `
                    position: fixed;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    background: rgba(0,0,0,0.9);
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    z-index: 10000;
                    cursor: pointer;
                    animation: fadeIn 0.3s ease-out;
                `;
                
                const img = document.createElement('img');
                img.src = thumbnailData;
                img.style.cssText = `
                    max-width: 90%;
                    max-height: 90%;
                    border-radius: 8px;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.5);
                    animation: scaleIn 0.3s ease-out;
                `;
                
                // Add close button for mobile
                if (this.currentBreakpoint === 'mobile') {
                    const closeBtn = document.createElement('button');
                    closeBtn.innerHTML = '✕';
                    closeBtn.style.cssText = `
                        position: absolute;
                        top: 20px;
                        right: 20px;
                        background: rgba(255,255,255,0.9);
                        border: none;
                        border-radius: 50%;
                        width: 40px;
                        height: 40px;
                        font-size: 20px;
                        cursor: pointer;
                        z-index: 10001;
                    `;
                    modal.appendChild(closeBtn);
                    
                    closeBtn.addEventListener('click', (e) => {
                        e.stopPropagation();
                        this.closeModal(modal);
                    });
                }
                
                modal.appendChild(img);
                document.body.appendChild(modal);
                
                // Close handlers
                modal.addEventListener('click', () => this.closeModal(modal));
                
                const escapeHandler = (e) => {
                    if (e.key === 'Escape') {
                        this.closeModal(modal);
                        document.removeEventListener('keydown', escapeHandler);
                    }
                };
                document.addEventListener('keydown', escapeHandler);
            }
            
            closeModal(modal) {
                modal.style.animation = 'fadeOut 0.2s ease-out';
                setTimeout(() => {
                    if (modal.parentNode) {
                        document.body.removeChild(modal);
                    }
                }, 200);
            }
        }
        
        // Initialize responsive interface when DOM is ready
        document.addEventListener('DOMContentLoaded', () => {
            window.responsiveImageInterface = new ResponsiveImageInterface();
        });
        
        // Global functions for backward compatibility
        function clearImage(imageType) {
            const clearBtn = document.getElementById('clear_' + imageType + '_image_btn');
            if (clearBtn) {
                clearBtn.click();
            }
        }
        
        function showLargePreview(imageType, thumbnailData) {
            if (window.responsiveImageInterface) {
                window.responsiveImageInterface.showLargePreview(imageType, thumbnailData);
            }
        }
        
        // Add CSS animations
        const style = document.createElement('style');
        style.textContent = `
            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }
            
            @keyframes fadeOut {
                from { opacity: 1; }
                to { opacity: 0; }
            }
            
            @keyframes scaleIn {
                from { 
                    opacity: 0; 
                    transform: scale(0.8); 
                }
                to { 
                    opacity: 1; 
                    transform: scale(1); 
                }
            }
            
            .touch-active {
                background: #e3f2fd !important;
                transform: scale(0.98);
            }
        `;
        document.head.appendChild(style);
        </script>
        """

def get_responsive_image_interface(config: Dict[str, Any]) -> ResponsiveImageInterface:
    """Get singleton instance of responsive image interface"""
    if not hasattr(get_responsive_image_interface, '_instance'):
        get_responsive_image_interface._instance = ResponsiveImageInterface(config)
    return get_responsive_image_interface._instance