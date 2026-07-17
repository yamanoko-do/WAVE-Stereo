"""
Stereo video source — simulates a binocular camera from a side-by-side video file.
"""
import cv2
import numpy as np
import time


class StereoVideoSource:
    """
    Reads a side-by-side stereo video and provides get_rectifyframe() interface,
    compatible with the infercam pipeline.
    """

    def __init__(self, video_path, loop=True, target_fps=None):
        """
        Args:
            video_path: Path to side-by-side stereo MP4
            loop: Loop playback when video ends
            target_fps: Playback speed (None = video native fps)
        """
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Each eye is half the width
        self.half_w = self.frame_width // 2

        self.loop = loop
        self.target_fps = target_fps if target_fps else self.video_fps
        self.frame_interval = 1.0 / self.target_fps if self.target_fps > 0 else 0
        self._last_time = time.time()
        self._frame_count = 0

        print(f"[INFO] StereoVideoSource: {video_path}")
        print(f"       Resolution: {self.frame_width}×{self.frame_height} "
              f"(each eye: {self.half_w}×{self.frame_height})")
        print(f"       FPS: {self.video_fps:.1f}, Frames: {self.total_frames}")
        print(f"       Playback: {self.target_fps:.1f} FPS, loop={self.loop}")

    def get_rectifyframe(self):
        """
        Returns next frame in camera-compatible format.
        Blocks to maintain target playback FPS.
        Returns None when video ends (if loop=False).
        """
        # Pace playback
        elapsed = time.time() - self._last_time
        wait = self.frame_interval - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_time = time.time()

        ret, frame = self.cap.read()
        if not ret:
            if self.loop:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if not ret:
                    return None
            else:
                return None

        # Split side-by-side → left/right
        left = frame[:, :self.half_w, :]
        right = frame[:, self.half_w:, :]

        self._frame_count += 1
        return {'left': left, 'right': right}

    def stop(self):
        self.cap.release()

    def get_fps(self):
        return self.target_fps

    @property
    def fps(self):
        return self.target_fps
