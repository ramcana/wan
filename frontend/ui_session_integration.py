"""
UI Session Integration for Wan2.2
Integrates session state management with the existing UI components
"""

import gradio as gr
import logging
from typing import Optional, Dict, Any, Tuple, List
from PIL import Image
from image_session_manager import get_image_session_manager, cleanup_session_manager

logger = logging.getLogger(__name__)

class UISessionIntegration:
    """Integrates session state management with Gradio UI components"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.session_manager = get_image_session_manager(config)
        
        # UI component references
        self.start_image_input = None
        self.end_image_input = None
        self.model_type_dropdown = None
        
        # Session state tracking
        self.ui_state = {
            "current_model_type": "t2v-A14B",
            "images_loaded_from_session": False,
            "last_session_restore": None
        }
        
        logger.info(f"UI session integration initialized with session: {self.session_manager.session_id}")
    
    def setup_ui_components(self, start_image_input: gr.Image, end_image_input: gr.Image, 
                           model_type_dropdown: gr.Dropdown) -> None:
        """
        Setup UI component references and event handlers
        
        Args:
            start_image_input: Gradio Image component for start image
            end_image_input: Gradio Image component for end image  
            model_type_dropdown: Gradio Dropdown for model type selection
        """
        self.start_image_input = start_image_input
        self.end_image_input = end_image_input
        self.model_type_dropdown = model_type_dropdown
        
        # Setup event handlers for session persistence
        self._setup_session_event_handlers()
        
        # Restore any existing session data
        self._restore_session_on_startup()
        
        logger.debug("UI components setup for session integration")
    
    def _setup_session_event_handlers(self):
        """Setup event handlers for automatic session persistence"""
        if self.start_image_input:
            self.start_image_input.change(
                fn=self._on_start_image_change,
                inputs=[self.start_image_input],
                outputs=[]
            )
        
        if self.end_image_input:
            self.end_image_input.change(
                fn=self._on_end_image_change,
                inputs=[self.end_image_input],
                outputs=[]
            )
        
        if self.model_type_dropdown:
            self.model_type_dropdown.change(
                fn=self._on_model_type_change,
                inputs=[self.model_type_dropdown],
                outputs=[]
            )
    
    def _on_start_image_change(self, image: Optional[Image.Image]) -> None:
        """Handle start image upload/change"""
        try:
            if image is not None:
                success, message = self.session_manager.store_image(image, "start")
                if success:
                    logger.debug("Start image stored in session")
                else:
                    logger.warning(f"Failed to store start image: {message}")
            else:
                # Image was cleared
                self.session_manager.clear_image("start")
                logger.debug("Start image cleared from session")
        except Exception as e:
            logger.error(f"Error handling start image change: {e}")
    
    def _on_end_image_change(self, image: Optional[Image.Image]) -> None:
        """Handle end image upload/change"""
        try:
            if image is not None:
                success, message = self.session_manager.store_image(image, "end")
                if success:
                    logger.debug("End image stored in session")
                else:
                    logger.warning(f"Failed to store end image: {message}")
            else:
                # Image was cleared
                self.session_manager.clear_image("end")
                logger.debug("End image cleared from session")
        except Exception as e:
            logger.error(f"Error handling end image change: {e}")
    
    def _on_model_type_change(self, model_type: str) -> None:
        """Handle model type change"""
        try:
            self.ui_state["current_model_type"] = model_type
            logger.debug(f"Model type changed to: {model_type}")
            
            # Images are preserved when switching model types
            # This allows users to switch between I2V and TI2V without losing images
            
        except Exception as e:
            logger.error(f"Error handling model type change: {e}")
    
    def _restore_session_on_startup(self) -> bool:
        """Restore session data when UI starts up"""
        try:
            session_info = self.session_manager.get_session_info()
            
            # Check if we have images to restore
            has_start = session_info.get("has_start_image", False)
            has_end = session_info.get("has_end_image", False)
            
            if has_start or has_end:
                logger.info(f"Restoring session with start_image={has_start}, end_image={has_end}")
                
                # Mark that we have loaded images from session
                self.ui_state["images_loaded_from_session"] = True
                self.ui_state["last_session_restore"] = session_info.get("last_updated")
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to restore session on startup: {e}")
            return False
    
    def get_session_images(self) -> Tuple[Optional[Image.Image], Optional[Image.Image]]:
        """
        Get current session images for UI display
        
        Returns:
            Tuple of (start_image, end_image)
        """
        try:
            start_image, start_metadata = self.session_manager.retrieve_image("start")
            end_image, end_metadata = self.session_manager.retrieve_image("end")
            
            return start_image, end_image
            
        except Exception as e:
            logger.error(f"Failed to get session images: {e}")
            return None, None
    
    def get_session_image_metadata(self) -> Dict[str, Any]:
        """Get metadata for session images"""
        try:
            session_info = self.session_manager.get_session_info()
            return {
                "start_image_metadata": session_info.get("start_image_metadata"),
                "end_image_metadata": session_info.get("end_image_metadata"),
                "session_id": session_info.get("session_id"),
                "last_updated": session_info.get("last_updated")
            }
        except Exception as e:
            logger.error(f"Failed to get session image metadata: {e}")
            return {}
    
    def clear_session_images(self) -> bool:
        """Clear all images from current session"""
        try:
            success = self.session_manager.clear_all_images()
            if success:
                logger.info("All session images cleared")
            return success
        except Exception as e:
            logger.error(f"Failed to clear session images: {e}")
            return False
    
    def restore_from_session_id(self, session_id: str) -> bool:
        """
        Restore UI state from a specific session ID
        
        Args:
            session_id: Session ID to restore from
            
        Returns:
            Success status
        """
        try:
            success = self.session_manager.restore_from_session_id(session_id)
            if success:
                self.ui_state["images_loaded_from_session"] = True
                self.ui_state["last_session_restore"] = session_id
                logger.info(f"Successfully restored from session: {session_id}")
            return success
        except Exception as e:
            logger.error(f"Failed to restore from session {session_id}: {e}")
            return False
    
    def get_ui_state_for_generation(self) -> Dict[str, Any]:
        """
        Get UI state data needed for generation tasks
        
        Returns:
            Dictionary with current UI state including images
        """
        try:
            start_image, end_image = self.get_session_images()
            metadata = self.get_session_image_metadata()
            
            return {
                "start_image": start_image,
                "end_image": end_image,
                "model_type": self.ui_state["current_model_type"],
                "session_id": self.session_manager.session_id,
                "image_metadata": metadata,
                "has_session_images": start_image is not None or end_image is not None
            }
        except Exception as e:
            logger.error(f"Failed to get UI state for generation: {e}")
            return {
                "start_image": None,
                "end_image": None,
                "model_type": self.ui_state["current_model_type"],
                "session_id": None,
                "image_metadata": {},
                "has_session_images": False
            }
    
    def prepare_generation_task_with_session_data(self, base_task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare generation task data with session images
        
        Args:
            base_task_data: Base task data from UI
            
        Returns:
            Enhanced task data with session images
        """
        try:
            ui_state = self.get_ui_state_for_generation()
            
            # Merge session data with base task data
            enhanced_task_data = base_task_data.copy()
            enhanced_task_data.update({
                "start_image": ui_state["start_image"],
                "end_image": ui_state["end_image"],
                "session_id": ui_state["session_id"],
                "image_metadata": ui_state["image_metadata"]
            })
            
            return enhanced_task_data
            
        except Exception as e:
            logger.error(f"Failed to prepare generation task with session data: {e}")
            return base_task_data
    
    def create_session_info_display(self) -> str:
        """Create HTML display of current session information"""
        try:
            session_info = self.session_manager.get_session_info()
            
            html = "<div style='padding: 10px; background: #f5f5f5; border-radius: 5px; margin: 5px 0;'>"
            html += f"<h4>Session Information</h4>"
            html += f"<p><strong>Session ID:</strong> {session_info.get('session_id', 'Unknown')}</p>"
            
            if session_info.get('has_start_image'):
                start_meta = session_info.get('start_image_metadata', {})
                html += f"<p><strong>Start Image:</strong> {start_meta.get('size', 'Unknown size')} "
                html += f"({start_meta.get('format', 'Unknown format')})</p>"
            
            if session_info.get('has_end_image'):
                end_meta = session_info.get('end_image_metadata', {})
                html += f"<p><strong>End Image:</strong> {end_meta.get('size', 'Unknown size')} "
                html += f"({end_meta.get('format', 'Unknown format')})</p>"
            
            if not session_info.get('has_start_image') and not session_info.get('has_end_image'):
                html += "<p><em>No images in current session</em></p>"
            
            html += f"<p><strong>Last Updated:</strong> {session_info.get('last_updated', 'Unknown')}</p>"
            html += "</div>"
            
            return html
            
        except Exception as e:
            logger.error(f"Failed to create session info display: {e}")
            return f"<div style='color: red;'>Error loading session info: {str(e)}</div>"
    
    def cleanup_session(self):
        """Cleanup current session (call on UI shutdown)"""
        try:
            self.session_manager.cleanup_current_session()
            logger.info("Session cleanup completed")
        except Exception as e:
            logger.error(f"Session cleanup failed: {e}")


# Global UI session integration instance
_ui_session_integration = None

def get_ui_session_integration(config: Dict[str, Any]) -> UISessionIntegration:
    """Get or create global UI session integration"""
    global _ui_session_integration
    if _ui_session_integration is None:
        _ui_session_integration = UISessionIntegration(config)
    return _ui_session_integration

def cleanup_ui_session_integration():
    """Cleanup global UI session integration"""
    global _ui_session_integration
    if _ui_session_integration is not None:
        _ui_session_integration.cleanup_session()
        _ui_session_integration = None
    
    # Also cleanup the session manager
    cleanup_session_manager()

def create_session_management_ui() -> Tuple[gr.HTML, gr.Button, gr.Button]:
    """
    Create UI components for session management
    
    Returns:
        Tuple of (session_info_display, clear_session_button, refresh_info_button)
    """
    with gr.Row():
        with gr.Column():
            session_info_display = gr.HTML(
                label="Session Information",
                value="<p>Loading session information...</p>"
            )
        
        with gr.Column(scale=0):
            refresh_info_button = gr.Button(
                "Refresh Info",
                size="sm",
                variant="secondary"
            )
            clear_session_button = gr.Button(
                "Clear Session",
                size="sm", 
                variant="stop"
            )
    
    return session_info_display, clear_session_button, refresh_info_button

def setup_session_management_handlers(
    ui_integration: UISessionIntegration,
    session_info_display: gr.HTML,
    clear_session_button: gr.Button,
    refresh_info_button: gr.Button,
    start_image_input: gr.Image,
    end_image_input: gr.Image
) -> None:
    """Setup event handlers for session management UI components"""
    
    def refresh_session_info():
        """Refresh session information display"""
        return ui_integration.create_session_info_display()
    
    def clear_session():
        """Clear current session and update UI"""
        success = ui_integration.clear_session_images()
        if success:
            return (
                ui_integration.create_session_info_display(),
                None,  # Clear start image
                None   # Clear end image
            )
        else:
            return (
                "<div style='color: red;'>Failed to clear session</div>",
                None,  # No change to start image
                None   # No change to end image
            )
    
    # Setup event handlers
    refresh_info_button.click(
        fn=refresh_session_info,
        outputs=[session_info_display]
    )
    
    clear_session_button.click(
        fn=clear_session,
        outputs=[session_info_display, start_image_input, end_image_input]
    )
    
    # Initial load of session info
    session_info_display.value = ui_integration.create_session_info_display()
