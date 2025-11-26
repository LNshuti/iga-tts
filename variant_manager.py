"""
User-variant assignment and state management.

Manages assigning curriculum variants to users and switching between variants.
"""

import json
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class VariantManager:
    """Manages user-variant assignments and state."""

    def __init__(self, db_logger):
        """
        Initialize variant manager.

        Args:
            db_logger: ABTestLogger instance for persistence
        """
        self.db_logger = db_logger

    def get_or_assign_user_variant(self, user_id: str, optimizer) -> Dict[str, str]:
        """
        Get user's current variant or assign a new one.

        Args:
            user_id: Anonymized user session ID
            optimizer: AxOptimizer instance for arm selection

        Returns:
            Variant dictionary with curriculum parameters
        """
        # Ensure user exists in database
        self.db_logger.ensure_user_exists(user_id)

        # Check if user has existing variant
        variant_id = self.db_logger.get_user_variant(user_id)

        if variant_id:
            try:
                variant = json.loads(variant_id)
                logger.debug(f"User {user_id} returning, using variant: {variant}")
                return variant
            except json.JSONDecodeError:
                logger.warning(f"Invalid variant JSON for user {user_id}, reassigning")

        # New user: assign variant via optimizer
        variant = optimizer.get_next_arm()
        variant_id = json.dumps(variant)

        self.db_logger.update_user_variant(user_id, variant_id)
        logger.info(f"Assigned new variant to user {user_id}: {variant}")

        return variant

    def update_user_variant(
        self, user_id: str, new_variant: Dict[str, str]
    ) -> None:
        """
        Update user's variant assignment (adaptive switching).

        Args:
            user_id: Anonymized user session ID
            new_variant: New variant parameters
        """
        variant_id = json.dumps(new_variant)
        self.db_logger.update_user_variant(user_id, variant_id)
        logger.info(f"Updated variant for user {user_id}: {new_variant}")

    def adaptive_variant_switch(
        self,
        user_id: str,
        current_performance: float,
        median_performance: float,
        optimizer,
    ) -> Optional[Dict[str, str]]:
        """
        Suggest switching to better-performing variant if user is struggling.

        Args:
            user_id: Anonymized user session ID
            current_performance: User's current performance (0-1)
            median_performance: Median performance across all users (0-1)
            optimizer: AxOptimizer instance to get best variant

        Returns:
            New variant if switch recommended, else None
        """
        # If performance is significantly below median, suggest switch
        threshold = median_performance * 0.7  # 30% below median

        if current_performance < threshold:
            best_variant = optimizer.get_best_arm()

            if best_variant:
                logger.info(
                    f"Recommending variant switch for user {user_id}: "
                    f"performance {current_performance:.2f} < threshold {threshold:.2f}"
                )
                return best_variant

        return None
