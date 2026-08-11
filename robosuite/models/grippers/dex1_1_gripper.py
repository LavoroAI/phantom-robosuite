"""
Unitree Dex1-1, the parallel jaw that ships on the G1.
"""
import numpy as np

from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion

# Travel of either finger, metres, open-positive; Unitree's limits and the ctrlrange.
JOINT_RANGE = (-0.02, 0.0245)

# Jaw opening = JAW_OFFSET + 2 * travel, one factor of 2 per finger. Measured in sim.
JAW_OFFSET = 0.05788


def opening_to_travel(opening):
    """
    Finger travel that puts the jaws `opening` metres apart, clipped to the stroke.

    Args:
        opening (float or np.array): desired jaw opening in metres

    Returns:
        float or np.array: commanded travel in metres, open-positive
    """
    return np.clip((opening - JAW_OFFSET) / 2.0, *JOINT_RANGE)


class Dex1Gripper(GripperModel):
    """
    Unitree Dex1-1: two prismatic fingers on a shared actuator.

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/dex1_1_gripper.xml"), idn=idn)

    def format_action(self, action):
        """
        Maps continuous action into binary output
        -1 => open, 1 => closed

        Args:
            action (np.array): gripper-specific action

        Raises:
            AssertionError: [Invalid action dimension size]
        """
        assert len(action) == 1
        self.current_action = np.clip(self.current_action + self.speed * np.sign(action), -1.0, 1.0)
        # robosuite's abstract action is closed-positive; the actuator is open-positive.
        return -self.current_action

    @property
    def opposed_actuators(self):
        # One tendon over both fingers, so there is a single command to send.
        return False

    @property
    def speed(self):
        return 0.01

    @property
    def dof(self):
        return 1

    @property
    def init_qpos(self):
        return np.array([JOINT_RANGE[1], JOINT_RANGE[1]])

    @property
    def _important_geoms(self):
        # "left"/"right" label the two jaws, not the robot's sides.
        return {
            "left_finger": ["finger1_collision"],
            "right_finger": ["finger2_collision"],
            "left_fingerpad": ["finger1_collision"],
            "right_fingerpad": ["finger2_collision"],
        }
