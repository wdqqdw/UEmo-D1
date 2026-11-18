"""Original media budgets: full audio, 2 FPS video, ordered image frames."""
import av
import numpy as np
import soundfile as sf
import soxr
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize


def read_audio(path):
    audio, rate = sf.read(path, dtype='float32', always_2d=True)
    audio = audio.mean(axis=1)
    if rate != 16000:
        audio = soxr.resample(audio, rate, 16000)
    if not len(audio):
        raise ValueError(f'Empty audio: {path}')
    # Original training handling for clips shorter than the audio encoder minimum.
    if len(audio) < 640:
        audio = np.pad(audio, (0, 640-len(audio)))
    return audio


def read_video(path):
    container = av.open(str(path))
    try:
        stream = container.streams.video[0]
        h, w = smart_resize(stream.height, stream.width, factor=28,
                            min_pixels=56*56, max_pixels=224*224)
        frames, target, start = [], 0.0, None
        for frame in container.decode(stream):
            timestamp = (float(frame.pts*frame.time_base) if frame.pts is not None
                         else float(start or 0)+len(frames)*0.5)
            if start is None:
                start = timestamp
            if timestamp-start+1e-6 >= target:
                frames.append(frame.reformat(width=w, height=h, format='rgb24').to_ndarray())
                target += 0.5
    finally:
        container.close()
    if not frames:
        raise ValueError(f'Empty video: {path}')
    if len(frames) == 1:
        frames.append(frames[-1])
    return np.stack(frames).transpose(0, 3, 1, 2)
