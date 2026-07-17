"""
Point Cloud Module - Point cloud generation, denoising, and Open3D visualization
"""
import os
import threading
import queue
from typing import Optional, Tuple, Any

import numpy as np
import cv2

# Set environment variables for Open3D
os.environ["XDG_SESSION_TYPE"] = "x11"
os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"


def filter_depth_boundary(disp: np.ndarray, fx: float, baseline: float, zfar: float,
                          threshold: float = 0.1, dilate_iter: int = 0) -> np.ndarray:
    """
    Filter depth discontinuities using gradient-based denoising
    Sets disparity values to 0 where depth gradient is large

    Args:
        disp: Disparity map
        fx: Focal length x
        baseline: Camera baseline
        zfar: Max depth
        threshold: Gradient threshold for boundary detection
        dilate_iter: Dilation iterations for boundary region

    Returns:
        Filtered disparity map
    """
    # 预先计算有效视差mask并计算深度
    valid_disp = disp > 0.1
    depth = np.zeros_like(disp, dtype=np.float32)

    if np.any(valid_disp):
        depth[valid_disp] = (fx * baseline) / disp[valid_disp]

    # 合并有效性检查
    valid_depth = (depth > 0) & (depth < zfar)
    depth[~valid_depth] = 0

    # 使用cv2.magnitude代替np.sqrt(grad_x**2 + grad_y**2)，更快
    grad_x = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)

    max_grad = grad_mag.max()
    if max_grad > 1e-6:
        grad_norm = grad_mag / max_grad
    else:
        grad_norm = grad_mag

    mask = grad_norm > threshold
    if dilate_iter > 0:
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=dilate_iter).astype(bool)

    # 原地修改避免拷贝（如果调用方不需要原disp）
    disp_filtered = disp.copy()
    disp_filtered[mask] = 0.0
    return disp_filtered


class PointCloudProcessor:
    """
    Threaded point cloud processor - generates point clouds from disparity and RGB images
    """
    def __init__(self, K: np.ndarray, baseline: float, zfar: float, subsample: int = 2,
                 flip_y_z: bool = True):
        """
        Initialize point cloud processor

        Args:
            K: Camera intrinsic matrix (3x3)
            baseline: Camera baseline
            zfar: Max depth to include
            subsample: Subsampling factor for performance
            flip_y_z: Whether to flip y and z axes (True for Open3D, False for Rerun)
        """
        self.K = K
        self.baseline = baseline
        self.zfar = zfar
        self.subsample = subsample
        self.flip_y_z = flip_y_z
        self.input_queue = queue.Queue(maxsize=2)
        self.output_queue = queue.Queue(maxsize=2)
        self.stopped = False
        self.thread = threading.Thread(target=self.process_loop, daemon=True)

        # Pre-compute camera parameters
        self._init_camera_params()

        # Pre-allocated grid indices
        self._v = None
        self._u = None
        self._last_shape = None

        self.thread.start()

    def _init_camera_params(self):
        """Pre-compute camera intrinsics"""
        subsample = self.subsample
        if subsample > 1:
            self._fx = self.K[0, 0] / subsample
            self._fy = self.K[1, 1] / subsample
            self._cx = self.K[0, 2] / subsample
            self._cy = self.K[1, 2] / subsample
        else:
            self._fx, self._fy = self.K[0, 0], self.K[1, 1]
            self._cx, self._cy = self.K[0, 2], self.K[1, 2]

    def process_loop(self):
        """Point cloud processing thread loop"""
        while not self.stopped:
            try:
                disp, left_img, frame_id = self.input_queue.get(timeout=0.01)
                if disp is None:
                    continue
                points, colors, pixel_map = self.compute_pointcloud(disp, left_img)
                try:
                    while self.output_queue.full():
                        self.output_queue.get_nowait()
                    self.output_queue.put_nowait((points, colors, pixel_map, frame_id))
                except queue.Full:
                    pass
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[ERROR] PointCloudProcessor error: {e}")

    def compute_pointcloud(self, disp: np.ndarray, left_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute point cloud from disparity and left image

        Args:
            disp: Disparity map
            left_img: Left RGB image

        Returns:
            (points, colors, pixel_map) - Point coords, colors, and source pixel (u,v) per point
        """
        subsample = self.subsample
        if subsample > 1:
            disp = disp[::subsample, ::subsample]
            left_img = left_img[::subsample, ::subsample]

        fx, fy, cx, cy = self._fx, self._fy, self._cx, self._cy

        # Pre-compute grid indices
        disp_shape = disp.shape
        if self._last_shape != disp_shape:
            self._v, self._u = np.indices(disp_shape)
            self._last_shape = disp_shape

        # Mask computation
        mask = disp > 0.1
        valid = mask.copy()

        # Compute depth only in valid regions
        depth = np.zeros_like(disp, dtype=np.float32)
        depth[mask] = (fx * self.baseline) / disp[mask]

        # Combine validity checks
        valid &= (depth > 0) & (depth < self.zfar)

        # Compute 3D coordinates
        z = depth[valid]
        x = (self._u[valid] - cx) * z / fx
        y = (self._v[valid] - cy) * z / fy

        points = np.stack((x, y, z), axis=-1)

        # Extract colors
        colors = left_img[valid]
        if colors.shape[-1] == 3 and colors.dtype == np.uint8:
            colors = colors[..., ::-1]  # BGR -> RGB
        colors = colors.astype(np.float32) / 255.0

        # Coordinate system adjustment (flip for Open3D, not for Rerun)
        if self.flip_y_z:
            points[:, 1] *= -1
            points[:, 2] *= -1

        # Source pixel coordinates for each 3D point (undo subsample scaling)
        u_src = self._u[valid].astype(np.float32) * subsample
        v_src = self._v[valid].astype(np.float32) * subsample
        pixel_map = np.stack((u_src, v_src), axis=-1)  # (N, 2)

        return points, colors, pixel_map

    def submit(self, disp: np.ndarray, left_img: np.ndarray, frame_id: int):
        """Submit point cloud computation task"""
        try:
            while self.input_queue.full():
                self.input_queue.get_nowait()
            self.input_queue.put_nowait((disp, left_img, frame_id))
        except queue.Full:
            pass

    def get_result(self):
        """Get latest point cloud result"""
        latest = None
        try:
            while True:
                latest = self.output_queue.get_nowait()
        except queue.Empty:
            pass
        return latest if latest else (None, None, None, None)

    def stop(self):
        """Stop processing thread"""
        self.stopped = True
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)


try:
    import open3d as o3d

    class O3DVisualizer:
        """
        Open3D real-time visualization with point picking support.
        Press 'c' to select the center-most point and trigger cost-curve lookup.
        """
        def __init__(self, window_name: str = "3D Point Cloud View",
                     width: int = 1280, height: int = 720,
                     background_color: tuple = (0.2, 0.2, 0.2),
                     point_size: float = 4.5):
            self.vis = None
            self.pcd = None
            self.is_initialized = False
            self.geometry_added = False
            self.data_updated = False
            self.window_name = window_name
            self.width = width
            self.height = height
            self.background_color = np.asarray(background_color)
            self.point_size = point_size
            self._pixel_map = None  # (N,2) float32 – source (u,v) for each 3D point
            self.picked_pixel = None  # (u,v) of last selected point (via 'c' key)
            self._win_id = None
            self._pts_cache = None  # cached numpy array of points
            self._last_hl_time = 0
            self._colors_orig = None  # original colors before highlight
            self._prev_hl_idx = None  # previously highlighted point index
            self._disp_map = None  # disparity map (for lookups)
            self._prev_picked_pt = None  # previous picked 3D point (for distance calc)
            self._fx = 0.0
            self._baseline = 0.0
            self.should_close = False

        def init_window(self):
            """Initialize visualization window"""
            self.vis = o3d.visualization.VisualizerWithKeyCallback()
            self.vis.create_window(window_name=self.window_name,
                                   width=self.width, height=self.height)
            self.should_close = False
            self.vis.register_key_callback(ord('C'), self._on_pick)
            self.vis.register_key_callback(ord('c'), self._on_pick)
            self.vis.register_key_callback(ord('Q'), self._on_quit)
            self.vis.register_key_callback(ord('q'), self._on_quit)
            self.vis.register_key_callback(256, self._on_quit)  # Esc
            self.vis.register_animation_callback(self._animate_highlight)
            self.pcd = o3d.geometry.PointCloud()
            self.is_initialized = True
            self._anim_count = 0
            # Capture window ID for mouse-position lookup
            import subprocess as _sp
            try:
                r = _sp.run(['xdotool', 'search', '--name', self.window_name],
                           capture_output=True, text=True, timeout=2)
                self._win_id = r.stdout.strip().split('\n')[0] if r.stdout.strip() else None
            except Exception:
                self._win_id = None

        def set_pointcloud_data(self, points: np.ndarray, colors: np.ndarray,
                                pixel_map: np.ndarray = None,
                                disp_map: np.ndarray = None,
                                fx: float = 0.0, baseline: float = 0.0):
            if not self.is_initialized:
                self.init_window()
            if points is not None and colors is not None and len(points) > 0:
                self.pcd.points = o3d.utility.Vector3dVector(points)
                self.pcd.colors = o3d.utility.Vector3dVector(colors)
                self._pixel_map = pixel_map
                self._pts_cache = points
                self._colors_orig = colors.copy()
                self._prev_hl_idx = None
                self._disp_map = disp_map
                self._fx = fx
                self._baseline = baseline
                self.data_updated = True

        def _raycast(self, mx, my):
            """Find index of point closest to mouse: project to screen, pick nearest depth."""
            if self._pixel_map is None or len(self._pixel_map) == 0:
                return None
            pts = np.asarray(self.pcd.points)
            try:
                vc = self.vis.get_view_control()
                cam = vc.convert_to_pinhole_camera_parameters()
                cam_int = cam.intrinsic
                K = np.asarray(cam_int.intrinsic_matrix)
                ext = np.asarray(cam.extrinsic)
                R = ext[:3, :3]
                t = ext[:3, 3]
                # Project all points to screen
                pts_cam = (R @ pts.T + t.reshape(3, 1)).T  # (N,3) in camera space
                # Only points in front of camera
                front = pts_cam[:, 2] > 1e-3
                if not front.any():
                    return None
                uv_h = (K @ pts_cam[front].T).T  # homogeneous
                uv = uv_h[:, :2] / uv_h[:, 2:3]  # (N,2) pixel coords
                # Distance in screen pixels
                dist2d = np.linalg.norm(uv - np.array([[mx, my]]), axis=1)
                # Among points within 5px of mouse, pick closest depth
                nearby = dist2d < 5.0
                front_indices = np.where(front)[0]
                if nearby.any():
                    nearby_depth = pts_cam[front][nearby, 2]
                    idx = front_indices[nearby][int(np.argmin(nearby_depth))]
                else:
                    # Fallback: closest to ray
                    C = -R.T @ t
                    dir_world = R.T @ (np.linalg.inv(K) @ np.array([float(mx), float(my), 1.0]))
                    dir_world /= np.linalg.norm(dir_world)
                    C_to_pts = pts[front] - C
                    proj = np.dot(C_to_pts, dir_world)
                    perp = np.linalg.norm(C_to_pts - np.outer(proj, dir_world), axis=1)
                    idx = front_indices[int(np.argmin(perp))]
                return idx
            except Exception as e:
                print(f"[O3D ray] error: {e}")
                return None

        def _animate_highlight(self, vis):
            """Animation callback: highlight point nearest to mouse (modifies pcd colors)."""
            import time, subprocess as _sp
            now = time.time()
            if now - self._last_hl_time < 0.1:
                return
            self._last_hl_time = now

            if self._pixel_map is None or self._pts_cache is None or self._colors_orig is None:
                return

            # Get mouse pos
            try:
                r = _sp.run(['xdotool', 'getmouselocation'],
                           capture_output=True, text=True, timeout=0.5)
                parts = {}
                for tok in r.stdout.strip().split():
                    k, v = tok.split(':', 1)
                    if k in ('x', 'y'):
                        parts[k] = int(v)
                mx_g, my_g = parts.get('x', 0), parts.get('y', 0)
            except Exception:
                return

            # Lazy-find window ID if not known yet
            if not self._win_id:
                try:
                    r = _sp.run(['xdotool', 'search', '--name', self.window_name],
                               capture_output=True, text=True, timeout=1)
                    ids = r.stdout.strip().split('\n')
                    self._win_id = ids[0] if ids and ids[0] else None
                except Exception:
                    pass

            mx, my = mx_g, my_g
            if self._win_id:
                try:
                    r2 = _sp.run(['xdotool', 'getwindowgeometry', self._win_id],
                                capture_output=True, text=True, timeout=0.5)
                    wx = wy = 0
                    for line in r2.stdout.strip().split('\n'):
                        if 'Position' in line:
                            pos_str = line.split(':', 1)[1].strip().split()[0]
                            wx, wy = map(int, pos_str.split(','))
                    mx, my = mx_g - wx, my_g - wy
                except Exception:
                    pass

            self._anim_count += 1
            if self._anim_count == 1:
                print(f"[O3D hl] FIRST FIRE win_id={self._win_id} pts={len(self._pts_cache) if self._pts_cache is not None else 0}", flush=True)
            if self._anim_count % 30 == 0:
                print(f"[O3D hl] #{self._anim_count} mouse=({mx},{my}) map_ok={self._pixel_map is not None}", flush=True)

            if mx < 0 or my < 0:
                return

            idx = self._raycast(mx, my)
            if idx is None:
                return

            # Restore previous highlight, apply new one
            clr = self._colors_orig.copy()
            if self._prev_hl_idx is not None and self._prev_hl_idx < len(clr):
                pass  # already restored by copy
            if 0 <= idx < len(clr):
                clr[idx] = [0.0, 1.0, 0.0]  # bright green
            self.pcd.colors = o3d.utility.Vector3dVector(clr)
            vis.update_geometry(self.pcd)
            self._prev_hl_idx = idx

        def _on_quit(self, vis):
            """Key callback: request closing the Open3D point cloud window."""
            self.should_close = True
            return False

        def _on_pick(self, vis):
            """Key callback: select highlighted point for cost-curve display."""
            if self._pixel_map is None:
                return
            import subprocess as _sp
            try:
                r = _sp.run(['xdotool', 'getmouselocation'],
                           capture_output=True, text=True, timeout=1)
                parts = {}
                for tok in r.stdout.strip().split():
                    k, v = tok.split(':', 1)
                    if k in ('x', 'y'):
                        parts[k] = int(v)
                mx_g, my_g = parts.get('x', 0), parts.get('y', 0)
                wx, wy = 0, 0
                if self._win_id:
                    r2 = _sp.run(['xdotool', 'getwindowgeometry', self._win_id],
                                capture_output=True, text=True, timeout=1)
                    for line in r2.stdout.strip().split('\n'):
                        if 'Position' in line:
                            pos_str = line.split(':', 1)[1].strip().split()[0]
                            wx, wy = map(int, pos_str.split(','))
                mx, my = mx_g - wx, my_g - wy
                print(f"[O3D pick] mouse=({mx},{my})", flush=True)
                idx = self._raycast(mx, my)
                if idx is not None:
                    uv = self._pixel_map[idx]
                    self.picked_pixel = (int(uv[0]), int(uv[1]))
                    pt = self._pts_cache[idx]
                    # Disparity at picked point: from depth map + verify with z
                    disp_map_val = -1.0
                    disp_depth_val = -1.0
                    if self._disp_map is not None:
                        u, v = int(uv[0]), int(uv[1])
                        h, w = self._disp_map.shape
                        if 0 <= u < w and 0 <= v < h:
                            disp_map_val = float(self._disp_map[v, u])
                    depth = -pt[2]  # z was negated in compute_pointcloud
                    if depth > 0 and self._fx > 0 and self._baseline > 0:
                        disp_depth_val = self._fx * self._baseline / depth
                    # Distance to previous pick
                    dist_mm = 0.0
                    if self._prev_picked_pt is not None:
                        dist_mm = float(np.linalg.norm(pt - self._prev_picked_pt)) * 1000.0
                    self._prev_picked_pt = pt.copy()
                    print(f"[O3D pick] pixel=({uv[0]:.0f},{uv[1]:.0f}) "
                          f"disp_map={disp_map_val:.2f} disp_z={disp_depth_val:.2f} "
                          f"depth={depth:.3f}m dist_prev={dist_mm:.1f}mm",
                          flush=True)
            except Exception as e:
                print(f"[O3D] Pick failed: {e}")

        def spin_once(self) -> bool:
            if not self.is_initialized:
                return False
            if self.should_close:
                self.close()
                return False
            if self.data_updated:
                if not self.geometry_added:
                    self.vis.add_geometry(self.pcd)
                    opt = self.vis.get_render_option()
                    opt.background_color = self.background_color
                    opt.point_size = self.point_size
                    ctr = self.vis.get_view_control()
                    ctr.set_front([0, 0, -1])
                    ctr.set_up([0, -1, 0])
                    self.geometry_added = True
                else:
                    self.vis.update_geometry(self.pcd)
                self.data_updated = False
            if not self.vis.poll_events():
                self.close()
                return False
            if self.should_close:
                self.close()
                return False
            self.vis.update_renderer()
            return True

        def close(self):
            if self.is_initialized:
                self.vis.destroy_window()
                self.is_initialized = False
                self.geometry_added = False
                self.data_updated = False

    class O3DImageVisualizer:
        """
        Open3D OpenGL image display for the 2D disparity view.

        Why this exists instead of cv2.imshow: on a single GPU shared with
        TensorRT, cv2.imshow re-uploads the frame to the X server every call and
        lets the X compositor (Glamor) re-composite the window on the GPU every
        frame, which contends heavily with CUDA inference (measured ~46 FPS at
        1280x720). Keeping the image as a persistent GL texture and rendering a
        single textured quad avoids that per-frame upload/composite and reaches
        ~54 FPS at full resolution.

        Image data is updated in place: ``np.asarray(o3d.geometry.Image(...))``
        returns a writable view into Open3D's internal buffer, so blitting into
        it + ``update_geometry`` is picked up by the renderer without reallocating.
        """
        def __init__(self, window_name: str = "Disparity", width: int = 1280, height: int = 720):
            self.window_name = window_name
            self.width = width
            self.height = height
            self.vis = None
            self.img = None
            self.buf = None
            self.is_initialized = False

        def init_window(self, key_callbacks=None):
            """Create the window + persistent image texture. Optionally register
            (key_int, callback) pairs for keyboard handling."""
            self.vis = o3d.visualization.VisualizerWithKeyCallback()
            self.vis.create_window(window_name=self.window_name,
                                   width=self.width, height=self.height)
            self.img = o3d.geometry.Image(np.zeros((self.height, self.width, 3), dtype=np.uint8))
            self.vis.add_geometry(self.img)
            buf = np.asarray(self.img)  # writable view into Open3D's buffer
            self.buf = buf if buf.flags.writeable else buf.copy()
            if key_callbacks:
                for key, cb in key_callbacks:
                    self.vis.register_key_callback(key, cb)
            self.is_initialized = True

        def _blit(self, rgb: np.ndarray):
            h, w = rgb.shape[:2]
            if (h, w) != (self.height, self.width):
                rgb = cv2.resize(rgb, (self.width, self.height))
            self.buf[:] = rgb

        def update(self, rgb: np.ndarray) -> bool:
            """Push an RGB frame and pump window events. Returns False if the
            window was closed (so the caller can quit)."""
            if not self.is_initialized:
                return True
            self._blit(rgb)
            self.vis.update_geometry(self.img)
            if not self.vis.poll_events():
                return False
            self.vis.update_renderer()
            return True

        def poll(self) -> bool:
            """Pump window events only (no re-render). Call this between frames
            to keep the window responsive while waiting for the next result.
            Intentionally does NOT call update_renderer(): re-rendering on every
            spin-iteration (~1ms) would hammer the GPU with GL draws and contend
            with CUDA inference. Rendering happens in update() when there is a
            new frame to show."""
            if not self.is_initialized:
                return True
            return self.vis.poll_events()

        def close(self):
            if self.is_initialized:
                try:
                    self.vis.destroy_window()
                except Exception:
                    pass
                self.is_initialized = False

except ImportError:
    print("[WARNING] Open3D not available, 3D point cloud visualization disabled")

    class O3DVisualizer:
        """Fallback O3DVisualizer when Open3D is not available"""
        def __init__(self, *args, **kwargs):
            print("[WARNING] Open3D not installed, 3D visualization disabled")
            self.is_initialized = False

        def init_window(self):
            pass

        def set_pointcloud_data(self, *args):
            pass

        def spin_once(self) -> bool:
            return False

        def close(self):
            pass

    class O3DImageVisualizer:
        """Fallback when Open3D is not available"""
        def __init__(self, *args, **kwargs):
            self.is_initialized = False

        def init_window(self, key_callbacks=None):
            print("[WARNING] Open3D not installed, image display disabled")

        def update(self, rgb):
            return True

        def poll(self):
            return True

        def close(self):
            pass
