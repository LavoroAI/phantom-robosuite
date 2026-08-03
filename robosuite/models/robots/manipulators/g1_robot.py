import numpy as np

from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion


class G1(ManipulatorModel):
    """
    G1 is a humanoid by Unitree. This is the 29-DoF variant with 7-DoF arms, reduced
    to the two arms: the legs and waist are welded rigid in the asset, since Bimanual
    splits right from left at len(joints) // 2 and OSC's Jacobian spans every joint.

    Args:
        idn (int or str): Number or some other unique identification string for this robot instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("robots/g1/robot.xml"), idn=idn)

    @property
    def default_mount(self):
        # MOUNT_MAPPING keys NullMount under None, not under its class name.
        return None

    @property
    def default_gripper(self):
        """
        Since this is bimanual robot, returns dict with `'right'`, `'left'` keywords corresponding to their respective
        values

        Returns:
            dict: Dictionary containing arm-specific gripper names
        """
        return {"right": "Robotiq85Gripper", "left": "Robotiq85Gripper"}

    @property
    def default_controller_config(self):
        """
        Since this is bimanual robot, returns dict with `'right'`, `'left'` keywords corresponding to their respective
        values

        Returns:
            dict: Dictionary containing arm-specific default controller config names
        """
        return {"right": "default_g1", "left": "default_g1"}

    @property
    def init_qpos(self):
        """
        Since this is bimanual robot, returns [right, left] array corresponding to respective values

        Note that this is a pose such that the arms are held out in front of the body

        Returns:
            np.array: default initial qpos for the right, left arms
        """
        # [right, left], each shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw.
        # Negative shoulder pitch holds the arms out in front; 0 hangs them at the sides.
        # The elbows differ deliberately: the two hands work at different distances, and
        # a poor start drops OSC into a branch that walks the elbow into its +2.094 stop
        # and strands the hand. Both are swept per dataset, so expect to re-tune them.
        # Not upstream's -1.57, which is outside this asset's own [-1.047, 2.094].
        return np.array([-1.0, -0.1, 0.0, 0.8, 0.0, 0.0, 0.0, -1.0, 0.1, 0.0, 1.2, 0.0, 0.0, 0.0])

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.30, -0.1, 0.95),
            "empty": (-0.29, 0, 0.95),
            "table": lambda table_length: (-0.15 - table_length / 2, 0, 0.95),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "bimanual"

    @property
    def _eef_name(self):
        """
        Since this is bimanual robot, returns dict with `'right'`, `'left'` keywords corresponding to their respective
        values

        Returns:
            dict: Dictionary containing arm-specific eef names
        """
        return {"right": "right_eef", "left": "left_eef"}
