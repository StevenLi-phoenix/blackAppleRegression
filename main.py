import cv2
import numpy as np
import os
import hashlib
from pathlib import Path
from typing import Generator, Optional
import tqdm
import time
import logging
from functools import partial
from multiprocessing import Pool
import numba

video_path = Path("BadApple.mp4")

MINIMAL_PATTERN_SIZE = 10  # 最小方块尺寸

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
        except Exception:
            self.handleError(record)
logger.addHandler(TqdmLoggingHandler())

def _md5_video_path(path: Path) -> str:
    return hashlib.md5(str(Path(path).resolve()).encode("utf-8")).hexdigest()

def _md5_frame(frame: np.ndarray) -> str:
    return hashlib.md5(frame.tobytes()).hexdigest()

@numba.njit(cache=True)
def find_first_fit(background_mask: np.ndarray, patch_h: int, patch_w: int):
    """
    在 background_mask 中找到首个全 True 的 patch_h x patch_w 区域。
    返回 (found, top, left)。
    """
    h, w = background_mask.shape
    if patch_h > h or patch_w > w:
        return False, -1, -1

    # 积分图加速区域求和，True 记作 1
    integral = np.zeros((h + 1, w + 1), dtype=np.int32)
    for i in range(h):
        row_sum = 0
        for j in range(w):
            if background_mask[i, j]:
                row_sum += 1
            integral[i + 1, j + 1] = integral[i, j + 1] + row_sum

    area = patch_h * patch_w
    limit_h = h - patch_h + 1
    limit_w = w - patch_w + 1
    for top in range(limit_h):
        y1 = top
        y2 = top + patch_h
        for left in range(limit_w):
            x1 = left
            x2 = left + patch_w
            total = integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1]
            if total == area:
                return True, top, left
    return False, -1, -1

def process_frame_with_cache(frame: np.ndarray, video_hash: str, cache_root: Path = Path("cache")) -> np.ndarray:
    """
    基于帧内容和视频路径 md5 的缓存：cache/<video_hash>/<frame_hash>.png
    """
    cache_dir = cache_root / video_hash
    cache_dir.mkdir(parents=True, exist_ok=True)
    frame_hash = _md5_frame(frame)
    cache_file = cache_dir / f"{frame_hash}.png"

    if cache_file.exists():
        cached = cv2.imread(str(cache_file), cv2.IMREAD_UNCHANGED)
        if cached is not None:
            logger.debug(f"Cache hit: {cache_file}")
            return cached
        logger.debug(f"Cache miss (corrupted cache): {cache_file}")

    processed = process_frame(frame)
    cv2.imwrite(str(cache_file), processed)
    return processed

class recursive_video_processor:
    video_informations = {
        "fps": 0,
        "width": 0,
        "height": 0,
        "codec": "",
        "bitrate": 0,
        "duration": 0,
        "size": 0,
    }

    def play_video(self, video_path:Path = video_path, show:bool=False) -> Generator[np.ndarray, None, None]:
        ## read video informations
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            self.video_informations["fps"] = cap.get(cv2.CAP_PROP_FPS)
            self.video_informations["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.video_informations["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.video_informations["codec"] = int(cap.get(cv2.CAP_PROP_FOURCC))
            self.video_informations["bitrate"] = cap.get(cv2.CAP_PROP_BITRATE)
            self.video_informations["duration"] = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 0
            self.video_informations["size"] = os.path.getsize(video_path) if os.path.exists(video_path) else 0
            logger.info(f"Video information: {self.video_informations}")
        else:
            logger.error(f"Error: Could not open video file {video_path}")
            return
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Total frames: {total_frames}")
        
        for _ in tqdm.tqdm(range(total_frames), desc="Playing video"):
            ret, frame = cap.read()
            if not ret:
                break
            if frame is None or frame.size == 0:
                logger.warning("Warning: Empty frame received")
                continue
            if show:
                cv2.imshow("frame", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            yield frame
        cap.release()
        
        if show:
            cv2.destroyAllWindows()

    def process_frame_stream(
        self,
        stream:Generator[np.ndarray, None, None],
        use_multiprocessing: bool = False,
        workers: Optional[int] = None,
        video_path: Path = video_path,
        cache_root: Path = Path("cache"),
    ) -> Generator[np.ndarray, None, None]:
        video_hash = _md5_video_path(video_path)
        if use_multiprocessing:
            worker_count = workers or os.cpu_count() or 1
            worker = partial(process_frame_with_cache, video_hash=video_hash, cache_root=cache_root)
            with Pool(processes=worker_count) as pool:
                for processed in pool.imap(worker, stream, chunksize=1):
                    yield processed
            return

        for frame in stream:
            assert frame is not None, "Frame is None"
            assert frame.size > 0, "Frame is empty"
            yield process_frame_with_cache(frame, video_hash, cache_root)

    def save_stream(self, stream:Generator[np.ndarray, None, None], output_path:Path = Path("output.mp4")) -> None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = None
        frame_count = 0
        
        for frame in stream:
            if frame is None or frame.size == 0:
                logger.warning("Empty frame skipped in save_stream")
                continue
            
            # lazy load the video parameters
            if out is None:
                height, width = frame.shape[:2]
                fps = self.video_informations["fps"] or 30
                out = cv2.VideoWriter(
                    str(output_path),
                    fourcc,
                    fps,
                    (width, height)
                )
                if not out.isOpened():
                    logger.error("Failed to open VideoWriter")
                    return
            
            out.write(frame)
            frame_count += 1
        
        if out is not None:
            out.release()
            logger.info(f"Video saved to {output_path}, frames written: {frame_count}")
        else:
            logger.warning("No frames written, nothing saved")

    def run(self, video_path: Path = video_path, use_multiprocessing: bool = False, workers: Optional[int] = None):
        stream = self.play_video(video_path=video_path)
        processed_stream = self.process_frame_stream(
            stream,
            use_multiprocessing=use_multiprocessing,
            workers=workers,
            video_path=video_path,
        )
        self.save_stream(processed_stream, output_path="out.mp4")

    @staticmethod
    def process_frame(frame:np.ndarray) -> np.ndarray:
        """Process a frame of the video.
        Args:
            frame (np.ndarray): The frame to process.

        Returns:
            np.ndarray: The processed frame.
        """
        return frame

def process_frame(frame: np.ndarray) -> np.ndarray:
    """
    将主体 pattern 在空白区域递归缩放铺满后返回新帧。
    JIT: 用 numba 加速单帧位置搜索；
    Multiprocessing: run() 里可以按需并行处理多帧。
    """

    def binarize(input_frame: np.ndarray) -> np.ndarray:
        if input_frame.ndim == 3:
            gray = cv2.cvtColor(input_frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = input_frame
        return np.where(gray < 128, 0, 255).astype(np.uint8)

    def extract_pattern(binarized: np.ndarray) -> tuple[Optional[np.ndarray], int]:
        # 主颜色：出现次数最多的那个（通常是背景）
        counts = np.bincount(binarized.flatten(), minlength=256)
        main_color_local = int(np.argmax(counts))
        body_mask = (binarized != main_color_local)
        if not np.any(body_mask):
            return None, main_color_local
        ys, xs = np.where(body_mask)
        y1, y2 = ys.min(), ys.max() + 1
        x1, x2 = xs.min(), xs.max() + 1
        return binarized[y1:y2, x1:x2].copy(), main_color_local

    def resize_pattern(pattern: np.ndarray, target_h: int, target_w: int, main_color_local: int) -> np.ndarray:
        resized = cv2.resize(pattern, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        resized = np.where(resized < 128, 0, 255).astype(np.uint8)
        foreground_color = 0 if main_color_local == 255 else 255
        return np.where(resized == main_color_local, main_color_local, foreground_color).astype(np.uint8)

    def recursive_place(binarized: np.ndarray, pattern: np.ndarray, main_color_local: int, max_length: int) -> int:
        """
        从 max_length 开始尝试等比例缩放 pattern，优先放置大尺寸。
        成功放置后写回 binarized，返回放置的最大边长；失败返回 0。
        """
        ph, pw = pattern.shape
        frame_h, frame_w = binarized.shape
        base_max_len = max(ph, pw)
        # 允许扩张到帧的最大边或当前限制
        allowed_max = min(max(frame_h, frame_w), max_length)
        # 从大到小尝试，直到触底 MINIMAL_PATTERN_SIZE
        for target_len in range(allowed_max, MINIMAL_PATTERN_SIZE - 1, -1):
            scale = target_len / base_max_len
            target_h = max(1, int(round(ph * scale)))
            target_w = max(1, int(round(pw * scale)))
            if max(target_h, target_w) < MINIMAL_PATTERN_SIZE:
                continue
            if target_h > frame_h or target_w > frame_w:
                continue

            scaled_pattern = resize_pattern(pattern, target_h, target_w, main_color_local)
            background_mask = (binarized == main_color_local)
            found, top, left = find_first_fit(background_mask, target_h, target_w)
            if found:
                binarized[top:top+target_h, left:left+target_w] = scaled_pattern
                return max(target_h, target_w)
        return 0

    bin_frame = binarize(frame)
    pattern, main_color = extract_pattern(bin_frame)

    if pattern is None:
        logger.debug("No body detected in frame, returning original")
        return frame

    placement_count = 0
    max_len = max(bin_frame.shape)
    last_len = max_len

    while last_len > MINIMAL_PATTERN_SIZE:
        placed_len = recursive_place(bin_frame, pattern, main_color, last_len)
        if placed_len == 0 or placed_len <= MINIMAL_PATTERN_SIZE:
            break
        placement_count += 1
        last_len = placed_len

    if placement_count > 0:
        logger.debug(f"Placed pattern {placement_count} times")

    if frame.ndim == 3:
        return cv2.cvtColor(bin_frame, cv2.COLOR_GRAY2BGR)
    
    return bin_frame

if __name__ == "__main__":
    processor = recursive_video_processor()
    processor.process_frame = staticmethod(process_frame) # override the process_frame method with real implementation
    processor.run(use_multiprocessing=True, workers=max(1, os.cpu_count()//2)) # reason for //2 is adjust for hyperthreading and other workloads
