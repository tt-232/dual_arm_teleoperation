"""
Dual-arm teleoperation node.

Architecture
------------
  Camera → MediaPipe hand tracking → delta pose → /compute_ik → JointTrajectory
                                                              → gripper JointTrajectory

Each arm is controlled independently:
  operator's LEFT hand  → left_panda_arm
  operator's RIGHT hand → right_panda_arm

End-effector motion:
  • 2-D translation in the XZ plane of the world frame
  • 1-D rotation around the Y axis (hand tilt)

Gripper:
  • Pinch gesture → finger_joint1 + finger_joint2 position [0, 0.04 m]
"""

import math
import threading
import time

import cv2
import numpy as np
import rclpy
import tf2_geometry_msgs  # noqa: F401 — registers PoseStamped transform support
import tf2_ros
from geometry_msgs.msg import Point, PoseStamped, Vector3
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.srv import GetPositionIK
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import ColorRGBA
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker, MarkerArray

from dual_arm_teleop.hand_tracker import HandTracker

_FINGER_MAX = 0.04  # metres
_AXIS_LENGTH = 0.12  # metres — length of each goal pose axis arrow
_AXIS_WIDTH = 0.01  # metres — shaft diameter


def _make_axis_marker(
    frame_id: str,
    ns: str,
    marker_id: int,
    pose: PoseStamped,
    axis: int,
    color: ColorRGBA,
    stamp,
) -> Marker:
    """
    Build a single ARROW marker along one axis of a pose.
    axis: 0=X(red), 1=Y(green), 2=Z(blue)
    The arrow starts at pose.position and points along the rotated axis direction.
    """
    # Rotate unit axis vector by the pose quaternion
    q = pose.pose.orientation
    q_arr = np.array([q.x, q.y, q.z, q.w])
    unit = np.zeros(3)
    unit[axis] = 1.0
    tip = _rotate_vec(unit, q_arr) * _AXIS_LENGTH

    m = Marker()
    m.header.frame_id = frame_id
    m.header.stamp = stamp
    m.ns = ns
    m.id = marker_id
    m.type = Marker.ARROW
    m.action = Marker.ADD
    m.scale = Vector3(x=_AXIS_WIDTH, y=_AXIS_WIDTH * 2.0, z=0.0)
    m.color = color
    m.lifetime.sec = 1  # auto-expire after 1 s if not refreshed

    start = pose.pose.position
    m.points = [
        Point(x=start.x, y=start.y, z=start.z),
        Point(x=start.x + tip[0], y=start.y + tip[1], z=start.z + tip[2]),
    ]
    return m


def _rotate_vec(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q = [x, y, z, w]."""
    qx, qy, qz, qw = q
    # sandwich product: q * [0,v] * q_conj
    t = 2.0 * np.array(
        [
            qy * v[2] - qz * v[1],
            qz * v[0] - qx * v[2],
            qx * v[1] - qy * v[0],
        ]
    )
    return v + qw * t + np.cross([qx, qy, qz], t)


def _rotate_pose_y(pose: PoseStamped, delta_rad: float) -> PoseStamped:
    """Apply a rotation around the world X-axis (the red marker arrow)."""
    q = pose.pose.orientation
    # Current quaternion as array [x, y, z, w]
    q_cur = np.array([q.x, q.y, q.z, q.w])
    # Rotation delta around world X: q_delta = [sin(d/2), 0, 0, cos(d/2)]
    half = delta_rad / 2.0
    q_delta = np.array([math.sin(half), 0.0, 0.0, math.cos(half)])
    # Pre-multiply: q_new = q_delta * q_cur  (rotation in world frame)
    q_new = _quat_mul(q_delta, q_cur)
    out = PoseStamped()
    out.header = pose.header
    out.pose.position = pose.pose.position
    out.pose.orientation.x = q_new[0]
    out.pose.orientation.y = q_new[1]
    out.pose.orientation.z = q_new[2]
    out.pose.orientation.w = q_new[3]
    return out


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two [x, y, z, w] quaternions."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


class ArmController:
    """
    Controls one arm + its gripper.

    Arm motion: get current EE pose via TF2 → apply delta → /compute_ik → JointTrajectory
    Gripper:    direct JointTrajectory to gripper controller
    """

    def __init__(self, node: Node, side: str, joint_names: list):
        prefix = f"{side}_panda"
        self._node = node
        self._side = side
        self._group = f"{side}_panda_arm"
        self._ee_link = f"{prefix}_link8"
        self._joint_names = joint_names  # 7 arm joints

        self._arm_pub = node.create_publisher(
            JointTrajectory, f"/{side}_arm_controller/joint_trajectory", 1
        )
        self._gripper_pub = node.create_publisher(
            JointTrajectory, f"/{side}_gripper_controller/joint_trajectory", 1
        )

        self._gripper_joints = [f"{prefix}_finger_joint1"]
        self._last_gripper_pos = None
        self._GRIPPER_DEADBAND = 0.0005

        # IK service
        self._ik_client = node.create_client(GetPositionIK, "/compute_ik")

        # TF2 for FK (current EE pose)
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, node)

        # Goal pose marker publisher (axes visualisation in RViz)
        self._marker_pub = node.create_publisher(
            MarkerArray, f"/teleop/{side}_goal_marker", 10
        )
        # Axis colours: X=red, Y=green, Z=blue  (left arm full, right arm translucent)
        alpha = 1.0 if side == "left" else 0.7
        self._axis_colors = [
            ColorRGBA(r=1.0, g=0.0, b=0.0, a=alpha),  # X red
            ColorRGBA(r=0.0, g=1.0, b=0.0, a=alpha),  # Y green
            ColorRGBA(r=0.0, g=0.0, b=1.0, a=alpha),  # Z blue
        ]

        # Target pose maintained between frames (absolute, world frame)
        self._target_pose: PoseStamped | None = None
        self._last_good_pose: PoseStamped | None = None  # last pose IK succeeded for
        self._ik_pending = False  # guard against overlapping IK calls
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Arm control
    # ------------------------------------------------------------------

    def apply_delta(self, dx: float, dz: float, d_angle: float, current_js: JointState):
        """
        Update the target EE pose by (dx, dz, d_angle) and trigger IK.

        dx, dz     : world-frame Cartesian deltas in metres
        d_angle    : delta rotation around world Y axis in radians
        current_js : latest /joint_states message for IK seed
        """
        with self._lock:
            if self._ik_pending:
                self._node.get_logger().debug(
                    f"[{self._side}] IK pending, skipping frame"
                )
                return

            # Initialise target pose from current FK on first call
            if self._target_pose is None:
                pose = self._get_current_ee_pose()
                if pose is None:
                    self._node.get_logger().warn(
                        f"[{self._side}] TF lookup failed for {self._ee_link}"
                    )
                    return
                p = pose.pose.position
                self._node.get_logger().info(
                    f"[{self._side}] Target pose initialised from TF: "
                    f"x={p.x:.3f} y={p.y:.3f} z={p.z:.3f}"
                )
                self._target_pose = pose

            # Apply translation: screen horizontal → world Y, screen vertical → world Z
            self._target_pose.pose.position.y += (
                dx  # screen right = robot right (negate for mirroring)
            )
            self._target_pose.pose.position.z += dz

            # Apply Y-axis rotation
            if abs(d_angle) > 1e-4:
                self._target_pose = _rotate_pose_y(self._target_pose, d_angle)

            target = PoseStamped()
            target.header.frame_id = "world"
            target.header.stamp = self._node.get_clock().now().to_msg()
            target.pose = self._target_pose.pose

            self._ik_pending = True

        self._publish_goal_marker(target)
        self._call_ik_async(target, current_js)

    def _publish_goal_marker(self, pose: PoseStamped):
        """Publish XYZ axis arrows at the target EE pose for RViz validation."""
        stamp = self._node.get_clock().now().to_msg()
        ns = f"{self._side}_goal"
        markers = MarkerArray()
        for axis, color in enumerate(self._axis_colors):
            markers.markers.append(
                _make_axis_marker("world", ns, axis, pose, axis, color, stamp)
            )
        self._marker_pub.publish(markers)

    def reset_target(self):
        """Force re-initialise target pose from current FK next call."""
        with self._lock:
            self._target_pose = None
            self._last_good_pose = None

    def _get_current_ee_pose(self) -> PoseStamped | None:
        try:
            tf = self._tf_buffer.lookup_transform(
                "world",
                self._ee_link,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.05),
            )
        except Exception:
            return None
        pose = PoseStamped()
        pose.header.frame_id = "world"
        pose.pose.position.x = tf.transform.translation.x
        pose.pose.position.y = tf.transform.translation.y
        pose.pose.position.z = tf.transform.translation.z
        pose.pose.orientation = tf.transform.rotation
        return pose

    def _call_ik_async(self, target: PoseStamped, current_js: JointState):
        if not self._ik_client.service_is_ready():
            self._node.get_logger().warn(f"[{self._side}] /compute_ik not ready")
            with self._lock:
                self._ik_pending = False
            return

        req = GetPositionIK.Request()
        req.ik_request.group_name = self._group
        req.ik_request.pose_stamped = target
        req.ik_request.avoid_collisions = False  # collision check added separately
        req.ik_request.timeout.sec = 0
        req.ik_request.timeout.nanosec = 100_000_000  # 100 ms

        # Seed with only this arm's joints so the solver is unambiguous
        arm_set = set(self._joint_names)
        seed_names, seed_pos = (
            zip(
                *[
                    (n, p)
                    for n, p in zip(current_js.name, current_js.position)
                    if n in arm_set
                ]
            )
            if any(n in arm_set for n in current_js.name)
            else ([], [])
        )
        req.ik_request.robot_state.joint_state.name = list(seed_names)
        req.ik_request.robot_state.joint_state.position = list(seed_pos)

        p = target.pose.position
        self._node.get_logger().info(
            f"[{self._side}] IK target: x={p.x:.3f} y={p.y:.3f} z={p.z:.3f}"
        )
        future = self._ik_client.call_async(req)
        future.add_done_callback(self._on_ik_result)

    def _on_ik_result(self, future):
        with self._lock:
            self._ik_pending = False
        try:
            result: GetPositionIK.Response = future.result()
        except Exception as e:
            self._node.get_logger().warn(f"[{self._side}] IK call failed: {e}")
            return

        if result.error_code.val != MoveItErrorCodes.SUCCESS:
            self._node.get_logger().warn(
                f"[{self._side}] IK failed, error_code={result.error_code.val} "
                f"— reverting to last good pose"
            )
            with self._lock:
                self._target_pose = self._last_good_pose  # revert; None → re-init from TF
            return

        self._node.get_logger().debug(
            f"[{self._side}] IK success, publishing trajectory"
        )

        # Extract the 7 arm joints in order and publish trajectory
        js = result.solution.joint_state
        name_to_pos = dict(zip(js.name, js.position))
        positions = [name_to_pos.get(j, 0.0) for j in self._joint_names]

        # IK succeeded — save this as the last known-good target pose
        with self._lock:
            self._last_good_pose = self._target_pose

        msg = JointTrajectory()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.joint_names = self._joint_names
        pt = JointTrajectoryPoint()
        pt.positions = positions
        pt.velocities = [0.0] * len(positions)
        pt.time_from_start = Duration(seconds=0.1).to_msg()
        msg.points = [pt]
        self._arm_pub.publish(msg)

    # ------------------------------------------------------------------
    # Gripper control
    # ------------------------------------------------------------------

    def send_gripper(self, pinch_ratio: float):
        position = float(np.clip(pinch_ratio * _FINGER_MAX, 0.0, _FINGER_MAX))
        if (
            self._last_gripper_pos is not None
            and abs(position - self._last_gripper_pos) < self._GRIPPER_DEADBAND
        ):
            return
        self._last_gripper_pos = position

        msg = JointTrajectory()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.joint_names = self._gripper_joints
        pt = JointTrajectoryPoint()
        pt.positions = [position] * len(self._gripper_joints)
        pt.velocities = [0.0] * len(self._gripper_joints)
        pt.time_from_start = Duration(seconds=0.05).to_msg()
        msg.points = [pt]
        self._gripper_pub.publish(msg)


class TeleopNode(Node):

    _LEFT_JOINTS = [f"left_panda_joint{i}" for i in range(1, 8)]
    _RIGHT_JOINTS = [f"right_panda_joint{i}" for i in range(1, 8)]

    def __init__(self):
        super().__init__("dual_arm_teleop")

        self.declare_parameter("camera_index", 0)
        self.declare_parameter("linear_scale", 0.003)  # metres per normalised delta
        self.declare_parameter("angular_scale", 0.04)  # radians per normalised delta
        self.declare_parameter("publish_rate_hz", 30.0)
        self.declare_parameter("show_camera", True)

        self._cam_idx = self.get_parameter("camera_index").value
        self._lin_scale = self.get_parameter("linear_scale").value
        self._ang_scale = self.get_parameter("angular_scale").value
        self._rate_hz = self.get_parameter("publish_rate_hz").value
        self._show_cam = self.get_parameter("show_camera").value

        self._left_ctrl = ArmController(self, "left", self._LEFT_JOINTS)
        self._right_ctrl = ArmController(self, "right", self._RIGHT_JOINTS)

        # Latest joint state (thread-safe)
        self._joint_state: JointState | None = None
        self._js_lock = threading.Lock()
        self.create_subscription(JointState, "/joint_states", self._js_cb, 10)

        self._tracker = HandTracker(alpha=0.3)
        self._cap = cv2.VideoCapture(self._cam_idx)
        if not self._cap.isOpened():
            self.get_logger().error(f"Cannot open camera index {self._cam_idx}")
            raise RuntimeError("Camera not available")

        self.get_logger().info(
            "Dual-arm teleop node started (IK mode). Press Q to quit."
        )

        # Previous hand positions per side
        self._prev = {
            "left": {"x": None, "y": None, "angle": None, "t": None},
            "right": {"x": None, "y": None, "angle": None, "t": None},
        }

        # Gesture-based activation: arm only responds to commands when active.
        # Point index finger up (hold 20 frames) → activate.
        # Make a fist (hold 20 frames) → deactivate.
        _GESTURE_FRAMES = 20
        self._active = {"left": False, "right": False}
        self._gesture_count = {"left": 0, "right": 0}
        self._GESTURE_FRAMES = _GESTURE_FRAMES
        # Frames after activation where prev is updated but no deltas are sent,
        # allowing the hand to settle from the pointing pose into control pose.
        self._settling = {"left": 0, "right": 0}
        self._SETTLING_FRAMES = 10

        self._running = True
        self._thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._thread.start()

    def _js_cb(self, msg: JointState):
        with self._js_lock:
            self._joint_state = msg

    def _camera_loop(self):
        period = 1.0 / self._rate_hz
        while self._running and rclpy.ok():
            t0 = time.monotonic()
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)
            hand_states, mp_results = self._tracker.process(frame)

            now = time.monotonic()
            with self._js_lock:
                js = self._joint_state

            if js is not None:
                self._process_hand(
                    hand_states.get("Left"), "left", self._left_ctrl, now, js
                )
                self._process_hand(
                    hand_states.get("Right"), "right", self._right_ctrl, now, js
                )

            if self._show_cam:
                display = self._tracker.draw(frame.copy(), mp_results)
                self._annotate(display, hand_states)
                cv2.imshow("Dual Arm Teleop", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self._running = False
                    break

            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, period - elapsed))

        self._cap.release()
        if self._show_cam:
            cv2.destroyAllWindows()

    def _process_hand(
        self, state, side: str, ctrl: ArmController, now: float, js: JointState
    ):
        prev = self._prev[side]

        if state is None:
            ctrl.reset_target()
            ctrl._last_gripper_pos = None
            prev["x"] = prev["y"] = prev["angle"] = prev["t"] = None
            self._gesture_count[side] = 0
            return

        gesture = state.get("gesture", "other")
        active = self._active[side]

        # --- Gesture-based activation toggle ---
        if not active and gesture == "pointing":
            self._gesture_count[side] += 1
            if self._gesture_count[side] >= self._GESTURE_FRAMES:
                self._active[side] = True
                self._gesture_count[side] = 0
                self._settling[side] = self._SETTLING_FRAMES
                # Reset target so IK re-initialises from actual robot EE via TF
                ctrl.reset_target()
                # Reset prev so first control frame captures baseline without any delta
                prev["x"] = prev["y"] = prev["angle"] = prev["t"] = None
                self.get_logger().info(f"[{side}] Control ACTIVATED")
            return  # don't send commands while waiting for activation

        if active and gesture == "fist":
            self._gesture_count[side] += 1
            if self._gesture_count[side] >= self._GESTURE_FRAMES:
                self._active[side] = False
                self._gesture_count[side] = 0
                ctrl.reset_target()
                ctrl._last_gripper_pos = None
                prev["x"] = prev["y"] = prev["angle"] = prev["t"] = None
                self.get_logger().info(f"[{side}] Control DEACTIVATED")
            return  # don't send commands while fist is held

        # Reset gesture counter when no relevant gesture
        if gesture not in ("pointing", "fist"):
            self._gesture_count[side] = 0

        if not active:
            return  # inactive — ignore all commands

        # --- Normal control (only when active) ---
        x, y, angle, pinch = state["x"], state["y"], state["angle"], state["pinch"]

        if prev["x"] is None:
            prev.update({"x": x, "y": y, "angle": None, "t": now})
        elif self._settling[side] > 0:
            # Settling period: track x/y baseline but keep angle=None so the
            # rotation reference is captured fresh after the hand has fully opened.
            prev.update({"x": x, "y": y, "angle": None, "t": now})
            self._settling[side] -= 1
        else:
            dx = float(np.clip(x - prev["x"], -0.1, 0.1))
            dy = float(np.clip(y - prev["y"], -0.1, 0.1))

            if prev["angle"] is None:
                # First post-settling frame: capture angle baseline, no rotation applied
                da = 0.0
            else:
                da = _angular_diff(angle, prev["angle"])
                da = float(np.clip(da, -0.3, 0.3))

            ctrl.apply_delta(
                dx=dx * self._lin_scale,
                dz=-dy * self._lin_scale,  # screen Y down → world Z up
                d_angle=-da * self._ang_scale,  # negate to match hand rotation direction
                current_js=js,
            )
            prev.update({"x": x, "y": y, "angle": angle, "t": now})

        ctrl.send_gripper(pinch)

    def _annotate(self, frame, hand_states):
        h, w = frame.shape[:2]
        for side, color in [("Left", (0, 255, 0)), ("Right", (0, 100, 255))]:
            ctrl_side = side.lower()
            active = self._active.get(ctrl_side, False)
            count = self._gesture_count.get(ctrl_side, 0)
            state = hand_states.get(side)
            if state:
                cx, cy = int(state["x"] * w), int(state["y"] * h)
                cv2.circle(frame, (cx, cy), 8, color, -1)
                gesture = state.get("gesture", "other")
                label = (
                    f"{side} pinch={state['pinch']:.2f} "
                    f"ang={math.degrees(state['angle']):.0f}° [{gesture}]"
                )
                cv2.putText(
                    frame, label, (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
                )
            # Activation status banner per side
            status_text = f"{side}: {'ACTIVE' if active else 'INACTIVE'}"
            if count > 0:
                status_text += f" ({count}/{self._GESTURE_FRAMES})"
            status_color = (0, 255, 0) if active else (0, 0, 255)
            y_off = 30 if side == "Left" else 60
            cv2.putText(
                frame, status_text, (10, y_off),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2,
            )
        cv2.putText(
            frame,
            "Point(index up)=START  Fist=STOP  Q=quit",
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
        )

    def destroy_node(self):
        self._running = False
        self._thread.join(timeout=2.0)
        self._tracker.close()
        super().destroy_node()


def _angular_diff(a: float, b: float) -> float:
    diff = a - b
    while diff > math.pi:
        diff -= 2 * math.pi
    while diff < -math.pi:
        diff += 2 * math.pi
    return diff


def main(args=None):
    rclpy.init(args=args)
    node = TeleopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
