import cv2
import pyudev
import os
import glob
import numpy as np
import time
import threading

try:
    import moderngl
    import glfw
except ImportError:
    moderngl = None
    glfw = None


class ZEDUVC:
    """
    ZED 相机 UVC 驱动版 - 使用 OpenCV VideoCapture 直接读取
    不需要 pyzed/zed-sdk/CUDA，仅作为普通 USB 双目相机使用
    """

    # Stereolabs ZED 相机的 VID/PID
    DEFAULT_VID_PID = "2b03:f582"

    # 分辨率预设 (左右合并后的总宽度, 高度)
    RESOLUTIONS = {
        'HD2K': (4416, 1242),   # 2208x2
        'HD1080': (3840, 1080),  # 1920x2
        'HD720': (2560, 720),    # 1280x2
        'VGA': (1344, 376)       # 672x2
    }

    def __init__(self, device_id=None, vid_pid=None, resolution='HD720', fmt=None, fps=None, map_dir=None):
        """
        初始化 ZED UVC 相机

        Args:
            device_id (int): 手动指定 /dev/videoX 编号
            vid_pid (str): 通过 USB VID:PID 自动查找设备，例如 "2b03:f580"
            resolution (str): 分辨率 - 'HD2K', 'HD1080', 'HD720', 'VGA'
            fmt (str): 视频格式 - 'MJPG' 或 'YUYV' (默认优先 MJPG 以获得高帧率)
            fps (int): 目标帧率 (可选)
            map_dir (str): 校正映射表目录（可选）
        """
        self.device_path = None

        # 分辨率与目标帧率映射
        res_fps_map = {
            'HD2K': 15,
            'HD1080': 30,
            'HD720': 60,
            'VGA': 100
        }

        # 设置分辨率
        if resolution not in self.RESOLUTIONS:
            raise ValueError(f"不支持的分辨率: {resolution}，可选: {list(self.RESOLUTIONS.keys())}")

        self.resolution_name = resolution
        width, height = self.RESOLUTIONS[resolution]
        target_fps = fps if fps is not None else res_fps_map.get(resolution, 30)

        # 查找设备
        if vid_pid is None and device_id is None:
            vid_pid = self.DEFAULT_VID_PID
            print(f"[INFO] 使用默认 ZED VID:PID = {vid_pid}")

        if vid_pid:
            self.device_path = self._find_video_device(vid_pid)
            if not self.device_path:
                raise RuntimeError(f"未找到 VID:PID = {vid_pid} 对应的摄像头\n请使用 'lsusb' 确认设备，或手动指定 device_id")
            device_id = int(self.device_path.replace("/dev/video", ""))
            print(f"[INFO] 根据 VID:PID={vid_pid} 自动识别设备: {self.device_path}")
        elif device_id is not None:
            self.device_path = f"/dev/video{device_id}"
            print(f"[INFO] 使用手动指定的设备: {self.device_path}")
        else:
            raise ValueError(
                "必须指定参数：\n"
                "  device_id  → /dev/videoX 的编号\n"
                "  或 vid_pid → USB 设备ID (默认: 2b03:f582)"
            )

        # 校正映射表
        self.map1x = None
        self.map1y = None
        self.map2x = None
        self.map2y = None
        if map_dir is not None:
            self._load_rectify_maps(map_dir)
            print(f"[INFO] 已加载校正映射表")
        else:
            print("[WARNING] 未提供校正映射表目录，无法使用 get_rectifyframe() 方法")

        # 打开视频流
        self.cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 {self.device_path}")

        # 尝试设置格式和分辨率 - 优先 MJPG 以获得高帧率
        formats_to_try = [fmt] if fmt else ['MJPG', 'YUYV']
        success = False

        for try_fmt in formats_to_try:
            print(f"[INFO] 尝试格式: {try_fmt} @ {target_fps}fps")

            # 先释放再重新打开，确保设置生效
            self.cap.release()
            self.cap.open(device_id, cv2.CAP_V4L2)

            # 设置格式、分辨率、帧率（顺序很重要）
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*try_fmt))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, target_fps)

            # 读取几帧让设置生效
            for _ in range(5):
                ret, _ = self.cap.read()
                if not ret:
                    break

            # 验证设置
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

            # 尝试读取一帧测试
            ret, test_frame = self.cap.read()
            if ret and test_frame is not None and w == width and h == height:
                success = True
                print(f"[INFO] ZED UVC 相机初始化成功")
                print(f"[INFO]   分辨率: {w}x{h} ({self.resolution_name})")
                print(f"[INFO]   目标帧率: {target_fps}, 实际: {actual_fps}")
                print(f"[INFO]   格式: {try_fmt}")
                self.current_fps = actual_fps
                self.current_format = try_fmt
                break
            else:
                print(f"[WARN] 格式 {try_fmt} 不工作 (实际: {w}x{h}@{actual_fps}fps)，尝试下一个...")

        if not success:
            raise RuntimeError(f"无法设置分辨率 {resolution} @ {target_fps}fps，请尝试使用 HD720")

    def get_frame(self):
        """
        获取一帧图像并返回左右目和完整帧

        Returns:
            dict: {'left': img_left, 'right': img_right, 'binocular': frame}
        """
        ret, frame = self.cap.read()
        if not ret or frame is None:
            raise RuntimeError("读取相机帧失败")

        h, w = frame.shape[:2]
        mid = w // 2
        frame_left = frame[:, :mid]
        frame_right = frame[:, mid:]

        return {
            'left': frame_left,
            'right': frame_right,
            'binocular': frame
        }

    def get_rectifyframe(self, interpolation=cv2.INTER_LINEAR):
        """
        获取校正后的双目图像帧

        Args:
            interpolation (int): 插值方法

        Returns:
            dict: {'left': rectL, 'right': rectR, 'binocular': rect_frame}
        """
        if self.map1x is None:
            raise RuntimeError("未加载校正映射表，请在初始化时指定 map_dir")

        raw_frames = self.get_frame()
        imgL = raw_frames['left']
        imgR = raw_frames['right']

        rectL = cv2.remap(imgL, self.map1x, self.map1y, interpolation=interpolation)
        rectR = cv2.remap(imgR, self.map2x, self.map2y, interpolation=interpolation)

        rect_frame = np.hstack((rectL, rectR))

        return {
            'left': rectL,
            'right': rectR,
            'binocular': rect_frame
        }

    def _load_rectify_maps(self, map_dir=None):
        """加载双目校正映射表"""
        map_paths = {
            'map1x': f"{map_dir}/map1x.npy",
            'map1y': f"{map_dir}/map1y.npy",
            'map2x': f"{map_dir}/map2x.npy",
            'map2y': f"{map_dir}/map2y.npy"
        }

        for name, path in map_paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"校正映射表文件不存在: {path}")

        try:
            self.map1x = np.load(map_paths['map1x'])
            self.map1y = np.load(map_paths['map1y'])
            self.map2x = np.load(map_paths['map2x'])
            self.map2y = np.load(map_paths['map2y'])
            print(f"[INFO] 成功加载双目校正映射表")
        except Exception as e:
            raise RuntimeError(f"加载校正映射表失败: {str(e)}")

    def take_photo(self, save_dir="./data/chessboard_images/binocular"):
        """
        启动实时预览，按 's' 保存 left_N.jpg 和 right_N.jpg，按 'q' 或 ESC 退出。
        """
        os.makedirs(save_dir, exist_ok=True)

        photo_count = 1
        existing_files = glob.glob(os.path.join(save_dir, "left_*.jpg"))
        if existing_files:
            max_num = 0
            for file_path in existing_files:
                try:
                    filename = os.path.splitext(os.path.basename(file_path))[0]
                    number = int(filename.split('_')[-1])
                    if number > max_num:
                        max_num = number
                except (ValueError, IndexError):
                    continue
            photo_count = max_num + 1

        width, height = self.RESOLUTIONS[self.resolution_name]
        print(f"ZED UVC 相机预览中 ({self.resolution_name})... 按 's' 保存图像，按 'q' 或 ESC 退出。")

        display = ModernGLDisplay(width, height, "ZED UVC Camera - Press s to save, q/ESC to quit")

        try:
            while True:
                frames = self.get_frame()

                display.update(frames['binocular'])
                display.render()

                # 检查按键
                for key in display.get_pressed_keys():
                    if key == glfw.KEY_S:
                        left_path = os.path.join(save_dir, f"left_{photo_count}.jpg")
                        right_path = os.path.join(save_dir, f"right_{photo_count}.jpg")
                        cv2.imwrite(left_path, frames['left'])
                        cv2.imwrite(right_path, frames['right'])
                        print(f"已保存: {left_path} | {right_path}")
                        photo_count += 1
                    elif key in (glfw.KEY_Q, glfw.KEY_ESCAPE):
                        return

                if display.should_close():
                    break

        finally:
            display.close()

    def stop(self):
        """释放摄像头资源"""
        if self.cap.isOpened():
            self.cap.release()
            print(f"[INFO] ZED UVC 摄像头 {self.device_path} 已释放")

    @staticmethod
    def _find_video_device(vid_pid):
        """根据 VID:PID 查找对应的 /dev/videoX"""
        context = pyudev.Context()
        for device in context.list_devices(subsystem='video4linux'):
            parent = device.find_parent('usb', 'usb_device')
            if not parent:
                continue

            vid = parent.attributes.get('idVendor')
            pid = parent.attributes.get('idProduct')
            if not vid or not pid:
                continue

            if f"{vid.decode()}:{pid.decode()}".lower() == vid_pid.lower():
                return device.device_node
        return None

    @staticmethod
    def list_resolutions():
        """列出支持的分辨率"""
        print("ZED UVC 相机支持的分辨率 (左右合并):")
        print("  注意: ZED UVC 模式可能仅支持 HD720/VGA")
        for name, (w, h) in ZEDUVC.RESOLUTIONS.items():
            single_w = w // 2
            print(f"  {name}: {single_w}x{h} (单目) | {w}x{h} (合并)")




class ModernGLDisplay:
    """高性能 ModernGL 显示器用于 ZED 相机"""

    # 顶点着色器
    _vertex_shader = """
    #version 330
    in vec2 in_vert;
    in vec2 in_text;
    out vec2 v_text;
    void main() {
        gl_Position = vec4(in_vert, 0.0, 1.0);
        v_text = in_text;
    }
    """

    # 片段着色器（BGR to RGB 转换）
    _fragment_shader = """
    #version 330
    uniform sampler2D Texture;
    in vec2 v_text;
    out vec4 f_color;
    void main() {
        vec3 color = texture(Texture, v_text).rgb;
        // OpenCV 是 BGR，交换为 RGB
        color = color.bgr;
        f_color = vec4(color, 1.0);
    }
    """

    def __init__(self, width=2560, height=720, title="ZED UVC Camera"):
        self.width = width
        self.height = height
        self._initialized = False

        # 初始化 GLFW
        if not glfw.init():
            raise RuntimeError("无法初始化 GLFW")

        # 创建窗口
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        self.window = glfw.create_window(width, height, title, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("无法创建 GLFW 窗口")

        glfw.make_context_current(self.window)

        # 创建 ModernGL 上下文
        self.ctx = moderngl.create_context()

        # 创建着色器程序
        self.prog = self.ctx.program(
            vertex_shader=self._vertex_shader,
            fragment_shader=self._fragment_shader
        )

        # 全屏四边形顶点
        vertices = np.array([
            -1.0, -1.0, 0.0, 1.0,
             1.0, -1.0, 1.0, 1.0,
            -1.0,  1.0, 0.0, 0.0,
             1.0,  1.0, 1.0, 0.0,
        ], dtype=np.float32)

        self.vbo = self.ctx.buffer(vertices)
        self.vao = self.ctx.vertex_array(
            self.prog,
            [(self.vbo, '2f 2f', 'in_vert', 'in_text')]
        )

        # 创建纹理
        self.texture = self.ctx.texture((width, height), 3)
        self.texture.use()

        self._key_callback = None
        self._saved_keys = []
        glfw.set_key_callback(self.window, self._on_key)
        self._initialized = True

    def _on_key(self, window, key, scancode, action, mods):
        """内部键盘回调"""
        if action == glfw.PRESS:
            self._saved_keys.append(key)

    def get_pressed_keys(self):
        """获取并清除已按下的键列表"""
        keys = self._saved_keys.copy()
        self._saved_keys.clear()
        return keys

    def update(self, image):
        """更新纹理图像 (H, W, 3) BGR 格式"""
        self.texture.write(image.tobytes())

    def render(self):
        """渲染一帧"""
        self.ctx.clear(0.0, 0.0, 0.0)
        self.vao.render(moderngl.TRIANGLE_STRIP)
        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def should_close(self):
        return glfw.window_should_close(self.window)

    def close(self):
        """关闭窗口并释放资源"""
        if not self._initialized:
            return
        self.vao.release()
        self.vbo.release()
        self.texture.release()
        self.prog.release()
        self.ctx.release()
        glfw.destroy_window(self.window)
        glfw.terminate()
        self._initialized = False

    def __del__(self):
        self.close()


    @staticmethod
    def probe_device(device_id=0):
        """
        探测设备实际支持的格式和分辨率

        Args:
            device_id: /dev/videoX 编号
        """
        print(f"=== 探测设备 /dev/video{device_id} ===")

        formats = [('MJPG', 'MJPG'), ('YUYV', 'YUYV')]
        resolutions = [
            ('HD2K', 4416, 1242, 15),
            ('HD1080', 3840, 1080, 30),
            ('HD720', 2560, 720, 60),
            ('VGA', 1344, 376, 100)
        ]

        working_configs = []

        for fmt_name, fmt_code in formats:
            print(f"\n测试格式: {fmt_name}")

            for res_name, w, h, target_fps in resolutions:
                # 每次重新打开设备
                cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)

                if not cap.isOpened():
                    print(f"  ✗ 无法打开设备")
                    continue

                # 设置格式、分辨率和帧率
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fmt_code))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                cap.set(cv2.CAP_PROP_FPS, target_fps)

                # 读取几次来让设置生效
                for _ in range(5):
                    ret, frame = cap.read()

                actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                actual_fps = cap.get(cv2.CAP_PROP_FPS)

                ret, frame = cap.read()
                cap.release()

                if ret and frame is not None and actual_w == w and actual_h == h:
                    print(f"  ✓ {res_name}: {w}x{h} @ {actual_fps:.0f}fps - 工作正常")
                    working_configs.append((fmt_name, res_name, w, h, actual_fps))
                else:
                    print(f"  ✗ {res_name}: {w}x{h} (实际: {actual_w}x{actual_h}@{actual_fps:.0f}fps) - 不支持")

        print("\n=== 可用配置 ===")
        if working_configs:
            for cfg in working_configs:
                print(f"  {cfg[0]} - {cfg[1]} ({cfg[2]}x{cfg[3]} @ {cfg[4]:.0f}fps)")
        else:
            print("  未找到可用配置")


def test_performance(resolution='VGA', device_id=None):
    """测试 ZED UVC 相机性能

    Args:
        resolution: 分辨率 ('HD2K', 'HD1080', 'HD720', 'VGA')
        device_id: 设备ID (可选)
    """
    print(f"{'='*60}")
    print(f"ZED UVC 性能测试 - {resolution}")
    print(f"{'='*60}")

    res_fps_map = {
        'HD2K': 15,
        'HD1080': 30,
        'HD720': 60,
        'VGA': 100
    }
    target_fps = res_fps_map.get(resolution, 30)

    cam_kwargs = {
        'resolution': resolution,
        'fmt': 'MJPG',
        'fps': target_fps
    }
    if device_id is not None:
        cam_kwargs['device_id'] = device_id

    cam = ZEDUVC(**cam_kwargs)

    try:
        # 测试1: 纯读取（不处理）
        print(f"\n{'='*60}")
        print("测试1: 纯读取 (cap.read() 仅返回)")
        print(f"{'='*60}")
        frame_count = 0
        warmup_frames = 10
        warmed_up = False
        stats_start = 0

        while True:
            ret, _ = cam.cap.read()
            if ret:
                frame_count += 1

                if not warmed_up:
                    if frame_count >= warmup_frames:
                        warmed_up = True
                        frame_count = 0
                        stats_start = time.time()
                        print("预热完成，开始统计...")
                    continue

                elapsed = time.time() - stats_start
                if elapsed >= 2.0:
                    fps = frame_count / elapsed
                    print(f"纯读取 FPS: {fps:.1f} (目标: {target_fps})")
                    print(f"达到理论值: {fps/target_fps*100:.1f}%")
                    break

        # 测试2: get_frame (分割左右图)
        print(f"\n{'='*60}")
        print("测试2: get_frame() (含分割)")
        print(f"{'='*60}")
        frame_count = 0
        warmed_up = False
        stats_start = 0

        while True:
            frames = cam.get_frame()
            frame_count += 1

            if not warmed_up:
                if frame_count >= warmup_frames:
                    warmed_up = True
                    frame_count = 0
                    stats_start = time.time()
                    print("预热完成，开始统计...")
                continue

            elapsed = time.time() - stats_start
            if elapsed >= 2.0:
                fps = frame_count / elapsed
                print(f"get_frame FPS: {fps:.1f} (目标: {target_fps})")
                print(f"达到理论值: {fps/target_fps*100:.1f}%")
                break

        # 测试3: 显示 - 使用多线程分离捕获和显示
        print(f"\n{'='*60}")
        print("测试3: 实时显示 (按 'q' 退出)")
        print(f"{'='*60}")
        width, height = cam.RESOLUTIONS[resolution]

        # 共享数据和锁
        latest_frame = None
        latest_frame_lock = threading.Lock()
        current_capture_fps = 0
        current_display_fps = 0
        current_capture_ms = 0
        current_display_ms = 0
        stopped = False

        # 捕获线程函数
        def capture_thread():
            nonlocal latest_frame, stopped, current_capture_fps, current_capture_ms
            capture_count = 0
            capture_time_total = 0.0
            fps_start = time.time()

            while not stopped:
                t0 = time.time()
                frames = cam.get_frame()
                t1 = time.time()

                # 更新最新帧
                with latest_frame_lock:
                    latest_frame = frames.copy()

                capture_count += 1
                capture_time_total += (t1 - t0)

                # 统计捕获 FPS
                elapsed = time.time() - fps_start
                if elapsed >= 0.5:
                    current_capture_fps = capture_count / elapsed
                    current_capture_ms = (capture_time_total / capture_count) * 1000 if capture_count > 0 else 0
                    capture_count = 0
                    capture_time_total = 0.0
                    fps_start = time.time()

        # 使用 ModernGL 显示
        display = ModernGLDisplay(width, height, "ZED UVC")
        print("[INFO] 使用 ModernGL 显示 (多线程)")

        # 启动捕获线程
        cap_thread = threading.Thread(target=capture_thread, daemon=True)
        cap_thread.start()

        try:
            display_count = 0
            display_time_total = 0.0
            fps_start = time.time()

            while True:
                # 获取最新帧
                frame_to_show = None
                with latest_frame_lock:
                    if latest_frame is not None:
                        frame_to_show = latest_frame

                if frame_to_show is not None:
                    # 显示帧
                    display_img = frame_to_show['binocular'].copy()
                    cv2.putText(display_img, f"Capture: {current_capture_fps:.1f} ({current_capture_ms:.1f}ms) | Display: {current_display_fps:.1f} ({current_display_ms:.1f}ms)", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(display_img, f"Res: {resolution} | Target: {target_fps}", (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    t0 = time.time()
                    display.update(display_img)
                    display.render()
                    t1 = time.time()

                    display_count += 1
                    display_time_total += (t1 - t0)

                # 统计显示 FPS
                elapsed = time.time() - fps_start
                if elapsed >= 0.5:
                    current_display_fps = display_count / elapsed
                    current_display_ms = (display_time_total / display_count) * 1000 if display_count > 0 else 0
                    display_count = 0
                    display_time_total = 0.0
                    fps_start = time.time()

                # 检查按键
                for key in display.get_pressed_keys():
                    if key in (glfw.KEY_Q, glfw.KEY_ESCAPE):
                        stopped = True
                        break
                else:
                    if not display.should_close():
                        continue
                break
        finally:
            stopped = True
            cap_thread.join(timeout=1.0)
            display.close()

    finally:
        cam.stop()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='ZED UVC 相机测试')
    parser.add_argument('--probe', type=int, metavar='DEVICE_ID',
                        help='探测指定设备支持的格式和分辨率')
    parser.add_argument('--device', type=int, help='手动指定设备ID')
    parser.add_argument('--resolution', type=str, default='VGA',
                        choices=['HD2K', 'HD1080', 'HD720', 'VGA'],
                        help='分辨率')
    parser.add_argument('--test', action='store_true',
                        help='运行性能测试')
    args = parser.parse_args()

    if args.probe is not None:
        ZEDUVC.probe_device(args.probe)
    elif args.test:
        # 性能测试模式
        test_performance(resolution=args.resolution, device_id=args.device)
    else:
        print("=== ZED UVC 相机测试 ===")
        print()
        print("使用 --test 运行性能测试")
        print("使用 --probe N 探测设备可用配置")
        print()

        # 默认运行性能测试
        test_performance(resolution=args.resolution, device_id=args.device)
