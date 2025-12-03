import cv2
import numpy as np
import os
from pathlib import Path
from typing import Generator
import tqdm
import time
import logging
import numba
from multiprocessing import Pool

video_path = Path("BadApple.mp4")

FAST_TEST = 100
MIMIMAL_PATTERN_SIZE = 10  # 最小方块尺寸


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

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

    def process_frame_stream(self, stream:Generator[np.ndarray, None, None]) -> Generator[np.ndarray, None, None]:
        for frame in stream:
            assert frame is not None, "Frame is None"
            assert frame.size > 0, "Frame is empty"
            yield self.process_frame(frame)

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
            
        if FAST_TEST:
            if frame_count > FAST_TEST:
                logger.info(f"Fast test completed, frames written: {frame_count}")
                out.release()
                logger.info(f"Video saved to {output_path}, frames written: {frame_count}")

    def run(self):
        stream = self.play_video(video_path=video_path)
        processed_stream = self.process_frame_stream(stream)
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
    # 转灰度 + 二值化到 0/255
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame.copy()

    bin_frame = np.where(gray < 128, 0, 255).astype(np.uint8)

    # 主颜色：出现次数最多的那个（通常是背景）
    counts = np.bincount(bin_frame.flatten(), minlength=256)
    main_color = np.argmax(counts)

    # 主体区域（非 main_color）的 mask
    body_mask = (bin_frame != main_color)
    
    # 没有主体，直接返回原帧
    if not np.any(body_mask):
        logger.debug("No body detected in frame, returning original")
        return frame

    # 主体的最小外接矩形
    ys, xs = np.where(body_mask)
    y1, y2 = ys.min(), ys.max() + 1
    x1, x2 = xs.min(), xs.max() + 1

    # pattern：主体所在的矩形块（单通道二值图）
    pattern = bin_frame[y1:y2, x1:x2].copy()
    ph, pw = pattern.shape
    
    def recursive_place(binarized: np.ndarray, pattern: np.ndarray) -> int:
        """
        把 pattern 放到当前 frame 的空白区域（== main_color），保持原尺寸。
        能放则放一块，返回 pattern 的尺寸；不能放返回 0。
        """
        ph, pw = pattern.shape
        h, w = binarized.shape
        
        if max(ph, pw) < MIMIMAL_PATTERN_SIZE:
            return 0

        background_mask = (binarized == main_color)
        tqdm_agent = tqdm.tqdm(total=(h - ph + 1) * (w - pw + 1), desc="Recursive place")
        for top in range(0, h - ph + 1):
            for left in range(0, w - pw + 1):
                tqdm_agent.update(1)
                region = background_mask[top:top+ph, left:left+pw]
                if np.all(region):
                    binarized[top:top+ph, left:left+pw] = pattern
                    return max(ph, pw)
        
        return 0

    # 用 while 循环不断往空白处塞 pattern，直到再也塞不下
    length = max(ph, pw)
    placement_count = 0
    
    while length > MIMIMAL_PATTERN_SIZE:
        new_len = recursive_place(bin_frame, pattern)
        if new_len == 0:
            break
        placement_count += 1
        length = new_len
        tqdm.tqdm.write(f"Placed pattern at {length} length")
        cv2.imwrite(f"out_{placement_count}.png", bin_frame)
    
    if placement_count > 0:
        logger.debug(f"Placed pattern {placement_count} times")

    # 把单通道结果扩展回 3 通道，保持和输入一样的形状
    if frame.ndim == 3:
        out = cv2.cvtColor(bin_frame, cv2.COLOR_GRAY2BGR)
    else:
        out = bin_frame
        
    cv2.imwrite(f"out_{time.time()}.png", out)

    return out

if __name__ == "__main__":
    processor = recursive_video_processor()
    processor.process_frame = staticmethod(process_frame) # override the process_frame method with real implementation
    processor.run()
