#!/usr/bin/env python3
"""
Reliability System Monitoring Script

This script monitors the health and performance of the reliability system.
"""

import time
import json
import logging
from datetime import datetime
from reliability_integration import get_reliability_integration


def main():
    """Main monitoring loop."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    integration = get_reliability_integration()
    
    if not integration.is_available():
        logger.error("Reliability system not available")
        return
    
    logger.info("Starting reliability system monitoring")
    
    while True:
        try:
            # Get health status
            health_status = integration.get_health_status()
            
            # Log health status
            logger.info(f"Health status: {health_status}")
            
            # Check for alerts
            if health_status.get("error_rate", 0) > 0.1:
                logger.warning("High error rate detected")
            
            # Wait for next check
            time.sleep(60)
            
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
            break
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
            time.sleep(60)


if __name__ == "__main__":
    main()
