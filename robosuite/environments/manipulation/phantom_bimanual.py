
from collections import OrderedDict

import numpy as np
from scipy.spatial.transform import Rotation

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.environments.manipulation.two_arm_env import TwoArmEnv
from robosuite.models.arenas import TableArena2, EmptyArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial, find_elements
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat, mat2euler, quat2mat

class PhantomBimanual(TwoArmEnv):
    """
    This class corresponds to the stacking task for a single robot arm.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        placement_initializer (ObjectPositionSampler): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

        camera_segmentations (None or str or list of str or list of list of str): Camera segmentation(s) to use
            for each camera. Valid options are:

                `None`: no segmentation sensor used
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level

            If not None, multiple types of segmentations can be specified. A [list of str / str or None] specifies
            [multiple / a single] segmentation(s) to use for all cameras. A list of list of str specifies per-camera
            segmentation setting(s) to use.

    Raises:
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        robots,
        bimanual_setup,
        env_configuration="default",
        controller_configs=None,
        mount_types="default",
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="zed",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mujoco",
        renderer_config=None,
        object_placements=None,
        direct_gripper_control=False,
        camera_pos=None,
        camera_quat_wxyz=None,
        camera_fov=None,
        camera_sensorsize=None,
        camera_principalpixel=None,
        camera_focalpixel=None,
    ):

        self.bimanual_setup = bimanual_setup
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.4))

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        self.object_placements = object_placements
        self.camera_pos = camera_pos
        self.camera_quat_wxyz = camera_quat_wxyz
        self.camera_fov = camera_fov
        self.camera_sensorsize = camera_sensorsize
        self.camera_principalpixel = camera_principalpixel
        self.camera_focalpixel = camera_focalpixel

        self.robot_base_height = 2.0
        self.robot_base_offset = -0.5

        self.head_camera_suffix = "head_camera"

        # How a single humanoid is placed. See _place_humanoid.
        #   "egocentric" -- body stands under the camera, taking its lateral axis from
        #                   the camera's, so the camera sits between the arms both in
        #                   the world and in the image. 100% IK-feasible on epic demo 0.
        #   "reach"      -- pose solved for hand reach with all 6 DoF free. Also
        #                   feasible, but the fit drifts off-axis: on epic demo 0 it
        #                   puts the camera 0.28 m to the body's right and 43 deg off
        #                   its heading, so both arms enter the frame from one side.
        #   "camera"     -- head camera anchored exactly on the real extrinsics. True
        #                   egocentric view, but the human's head-to-hand distance
        #                   exceeds the G1's reach, so the arms cannot work (left arm
        #                   0% feasible).
        self.humanoid_placement = "egocentric"
        # Camera position in the pelvis frame: (forward, up). The lateral component is
        # 0 by construction. Both numbers are fitted for hand reach against a single
        # episode, and against an earlier body orientation at that, so re-fit them
        # before trusting them on new data.
        self.humanoid_camera_offset = (0.0, 0.70)
        self.humanoid_base_pos = (0.38688, 0.29905, 1.35596)
        self.humanoid_base_yaw = -1.19144

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types=mount_types,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
            direct_gripper_control=direct_gripper_control,
        )

    def reset(self, object_placements=None):
        self.object_placements = object_placements
        return super().reset()

    def _humanoid_head_camera(self):
        """
        Prefixed name of the single robot's own head camera, or None if this setup has
        no such robot (the two-arm cases, or a humanoid asset without the camera).
        """
        if len(self.robots) != 1:
            return None
        return next(
            (c for c in self.robots[0].robot_model.cameras if c.endswith(self.head_camera_suffix)),
            None,
        )

    def _place_humanoid(self):
        """
        Position and orient the single humanoid, per self.humanoid_placement.

        The two-mount rule used for the two-arm setups puts each robot's base at a
        human shoulder with a shoulder's orientation; a humanoid's base is its pelvis,
        so the same numbers lay the whole body on its side. It needs its own rule.

        "egocentric" (default) stands the pelvis under the camera and takes the body's
        lateral axis from the camera's, which is what puts one arm either side of the
        view -- see the derivation inline below. Only how far forward and how far below
        the camera the pelvis sits is free, and both are fitted for hand reach.

        "reach" places the pelvis where both arms can reach the whole trajectory, with
        nothing tying it to the camera. Feasible, but the fit is free to rotate the body
        away from the camera, which is how both arms once ended up on the same side.

        "camera" anchors the robot's own head camera onto the extrinsics, which is exact
        rather than fitted (T_base = T_camera_target @ inv(T_camera_in_base)). Not the
        default because the extrinsics are a human's head, and a human's head-to-hand
        distance is longer than the G1's arm.
        """
        robot_model = self.robots[0].robot_model

        if self.humanoid_placement == "egocentric" and self.camera_pos is not None:
            cam_R = quat2mat(convert_quat(np.array(self.camera_quat_wxyz), to="xyzw"))
            # The body's lateral axis is the camera's, in full 3D. That is the only way
            # to get both of the things "the camera sits between the arms" means, and a
            # yaw alone cannot do it:
            #
            #   the body's sagittal plane projects to a line through the image centre
            #     <=> the plane contains the optical axis  <=> left . cam_forward == 0
            #   that line is vertical, so the arms fall either side of it
            #     <=> the plane contains the camera's up axis  <=> left . cam_up == 0
            #
            # Both at once means left is perpendicular to cam_forward and cam_up, i.e.
            # left = -cam_right. On epic demo 0 an upright body can satisfy one or the
            # other -- yaw -0.439 or +0.153 -- and lands both arms on the same side
            # either way. Taking the axis whole rolls the body by the camera's roll,
            # 16 degrees here, which is invisible: only the arms are rendered.
            left = -cam_R[:, 0]
            left = left / max(np.linalg.norm(left), 1e-9)
            up = np.array((0.0, 0.0, 1.0)) - np.dot((0.0, 0.0, 1.0), left) * left
            up = up / max(np.linalg.norm(up), 1e-9)
            forward = np.cross(left, up)
            R = np.column_stack((forward, left, up))

            fwd_off, up_off = self.humanoid_camera_offset
            base_pos = np.array(self.camera_pos) - R @ np.array((fwd_off, 0.0, up_off))
            robot_model.set_base_xpos(base_pos + robot_model.bottom_offset)
            robot_model.set_base_ori(mat2euler(R))
            return

        if self.humanoid_placement == "reach":
            xpos = np.array(self.humanoid_base_pos, dtype=float)
            xpos[2] += robot_model.bottom_offset[2]
            robot_model.set_base_xpos(xpos)
            robot_model.set_base_ori(np.array((0.0, 0.0, self.humanoid_base_yaw)))
            return

        if self.camera_pos is None:
            # No extrinsics supplied (a bare env, e.g. for tests). Stand it upright at
            # the nominal base height rather than deriving from a camera we don't have.
            robot_model.set_base_xpos(
                np.array((0, 0, self.robot_base_height + self.robot_base_offset)) + robot_model.bottom_offset
            )
            return

        # Camera pose in the robot's own base frame, from a standalone compile before
        # the base is moved.
        import mujoco

        m = robot_model.get_model(mode="mujoco")
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, self._humanoid_head_camera())
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, robot_model.root_body)
        base_R = d.xmat[bid].reshape(3, 3)
        cam_in_base_R = base_R.T @ d.cam_xmat[cid].reshape(3, 3)
        cam_in_base_p = base_R.T @ (d.cam_xpos[cid] - d.xpos[bid])

        target_R = quat2mat(convert_quat(np.array(self.camera_quat_wxyz), to="xyzw"))
        target_p = np.array(self.camera_pos)

        R = target_R @ cam_in_base_R.T
        robot_model.set_base_xpos(target_p - R @ cam_in_base_p + robot_model.bottom_offset)
        robot_model.set_base_ori(mat2euler(R))

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        if self.bimanual_setup == "tabletop":
            count = 0
            for robot, offset, rotation in zip(self.robots, (-0.2, 0.2), (0, 0)):
                xpos = np.array((0, offset, self.robot_base_height))
                robot.robot_model.set_base_xpos(xpos)
                rot = np.array((rotation, 0, np.pi)) if count == 1 else np.array((rotation, 0, 0))
                robot.robot_model.set_base_ori(rot)
                count += 1
        elif self.bimanual_setup == "shoulders1":
            count = 0
            for robot, offset, rotation in zip(self.robots, (-0.2, 0.2), (np.pi*2/3, -np.pi*2/3)):
                xpos = np.array((0, offset, self.robot_base_height))
                robot.robot_model.set_base_xpos(xpos)
                rot = np.array((rotation, 0, np.pi)) if count == 1 else np.array((rotation, 0, 0))
                robot.robot_model.set_base_ori(rot)
                count += 1
        elif self.bimanual_setup == "shoulders2":
            count = 0
            for robot, offset, rotation in zip(self.robots, (-0.2, 0.2), (np.pi/3, -np.pi/3)):
                xpos = np.array((0, offset, self.robot_base_height))
                robot.robot_model.set_base_xpos(xpos)
                rot = np.array((rotation, 0, np.pi)) if count == 1 else np.array((rotation, 0, 0))
                robot.robot_model.set_base_ori(rot)
                count += 1
        elif self.bimanual_setup == "shoulders" and self._humanoid_head_camera() is not None:
            self._place_humanoid()

        elif self.bimanual_setup == "shoulders":
            count = 0
            for robot, offset, rotation in zip(self.robots, (-0.2, 0.2), (np.pi/3, -np.pi/3)):
                if count == 1:
                    xpos = np.array((0, 0.2, self.robot_base_height+self.robot_base_offset+robot.robot_model.bottom_offset[2]))
                else:
                    xpos = np.array((-0.00656507, -0.14111039, 1.58980033+robot.robot_model.bottom_offset[2]))
                robot.robot_model.set_base_xpos(xpos)
                if count == 1:
                    rot = np.array((rotation, 0, np.pi/2)) 
                else: 
                    rot = np.array((0.50415113, -0.05164374, -1.57347674))
                robot.robot_model.set_base_ori(rot)
                count += 1

        mujoco_arena = EmptyArena()

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
        )

        # Modify zed camera
        if self.camera_pos is not None:
            intrinsics = {"sensorsize": np.array2string(self.camera_sensorsize)[1:-1],
                          "resolution": f"{self.camera_widths[0]} {self.camera_heights[0]}",
                          "principalpixel": np.array2string(self.camera_principalpixel)[1:-1],
                          "focalpixel": np.array2string(self.camera_focalpixel)[1:-1]}

            head_camera = self._humanoid_head_camera() if self.humanoid_placement == "camera" else None
            if head_camera is not None:
                # Only the intrinsics: this camera is rigidly fixed to the robot's
                # torso, and _place_humanoid has already put it at self.camera_pos.
                cam = find_elements(root=self.robots[0].robot_model.worldbody, tags="camera",
                                    attribs={"name": head_camera}, return_first=True)
                for attrib, value in intrinsics.items():
                    cam.set(attrib, value)
            else:
                mujoco_arena.set_camera(
                    camera_name="zed",
                    pos=self.camera_pos,
                    quat=self.camera_quat_wxyz,
                    camera_attribs=intrinsics,
                )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

   
    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()


    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the cube.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

    def reward(self, action):
        return 0.0