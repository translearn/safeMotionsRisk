# The code below is adapted from Open AI Gym v0.18.3
# https://github.com/openai/gym/blob/7feb3615d4ee17d81037f301efb77f8f32b74a1b/gym/wrappers/monitoring/video_recorder.py

# The MIT License
# Copyright (c) 2016 OpenAI (https://openai.com)
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions
# of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import json
import os
import subprocess
import tempfile
import os.path
from shutil import which
import numpy as np
from gymnasium import error
import logging as logger

def touch(path):
    open(path, 'a').close()

class InvalidFrame(Exception):
    """Raised when the user performs an action not contained within the action space."""

class VideoRecorder(object):
    """VideoRecorder renders a nice movie of a rollout, frame by frame. It
    comes with an `enabled` option so you can still use the same code
    on episodes where you don't want to record video.

    Note:
        You are responsible for calling `close` on a created
        VideoRecorder, or else you may leak an encoder process.

    Args:
        env (Env): Environment to take video of.
        path (Optional[str]): Path to the video file; will be randomly chosen if omitted.
        base_path (Optional[str]): Alternatively, path to the video file without extension, which will be added.
        metadata (Optional[dict]): Contents to save to the metadata file.
        enabled (bool): Whether to actually record video, or just no-op (for convenience)
    """

    def __init__(self, env, path=None, metadata=None, enabled=True, base_path=None):
        modes = env.metadata.get('render.modes', [])
        self._async = env.metadata.get('semantics.async')
        self.enabled = enabled

        # Don't bother setting anything else if not enabled
        if not self.enabled:
            return

        if path is not None and base_path is not None:
            raise error.Error("You can pass at most one of `path` or `base_path`.")

        self.last_frame = None
        self.env = env

        required_ext = '.mp4'
        if path is None:
            if base_path is not None:
                # Base path given, append ext
                path = base_path + required_ext
            else:
                # Otherwise, just generate a unique filename
                with tempfile.NamedTemporaryFile(suffix=required_ext, delete=False) as f:
                    path = f.name
        self.path = path

        path_base, actual_ext = os.path.splitext(self.path)

        if actual_ext != required_ext:
            hint = ''
            raise error.Error("Invalid path given: {} -- must have file extension {}.{}".format(self.path, required_ext, hint))
        # Touch the file in any case, so we know it's present. (This
        # corrects for platform platform differences. Using ffmpeg on
        # OS X, the file is precreated, but not on Linux.
        touch(path)

        self.frames_per_sec = env.metadata.get('video.frames_per_second', 30)
        self.output_frames_per_sec = env.metadata.get('video.output_frames_per_second', self.frames_per_sec)
        self.encoder = None # lazily start the process
        self.broken = False

        # Dump metadata
        self.metadata = metadata or {}
        self.metadata['content_type'] = 'video/mp4'
        self.metadata_path = '{}.meta.json'.format(path_base)
        self.write_metadata()

        logger.info('Starting new video recorder writing to %s', self.path)
        self.empty = True

    @property
    def functional(self):
        return self.enabled and not self.broken

    def capture_frame(self):
        """Render the given `env` and add the resulting frame to the video."""
        if not self.functional: return
        logger.debug('Capturing video frame: path=%s', self.path)

        render_mode = 'rgb_array'
        frame = self.env.render(mode=render_mode)

        if frame is None:
            if self._async:
                return
            else:
                # Indicates a bug in the environment: don't want to raise
                # an error here.
                logger.warning('Env returned None on render(). Disabling further rendering for video recorder by '
                               'marking as disabled: path=%s metadata_path=%s', self.path, self.metadata_path)
                self.broken = True
        else:
            self.last_frame = frame
            self._encode_image_frame(frame)

    def close(self):
        """Make sure to manually close, or else you'll leak the encoder process"""
        if not self.enabled:
            return

        if self.encoder:
            logger.debug('Closing video encoder: path=%s', self.path)
            self.encoder.close()
            self.encoder = None
        else:
            # No frames captured. Set metadata, and remove the empty output file.
            os.remove(self.path)

            if self.metadata is None:
                self.metadata = {}
            self.metadata['empty'] = True

        # If broken, get rid of the output file, otherwise we'd leak it.
        if self.broken:
            logger.info('Cleaning up paths for broken video recorder: path=%s metadata_path=%s',
                        self.path, self.metadata_path)

            # Might have crashed before even starting the output file, don't try to remove in that case.
            if os.path.exists(self.path):
                os.remove(self.path)

            if self.metadata is None:
                self.metadata = {}
            self.metadata['broken'] = True

        self.write_metadata()

    def write_metadata(self):
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f)

    def _encode_image_frame(self, frame):
        if not self.encoder:
            self.encoder = ImageEncoder(self.path, frame.shape, self.frames_per_sec, self.output_frames_per_sec)
            self.metadata['encoder_version'] = self.encoder.version_info

        try:
            self.encoder.capture_frame(frame)
        except InvalidFrame as e:
            logger.warning('Tried to pass invalid video frame, marking as broken: %s', e)
            self.broken = True
        else:
            self.empty = False

class ImageEncoder(object):
    def __init__(self, output_path, frame_shape, frames_per_sec, output_frames_per_sec):
        self.proc = None
        self.output_path = output_path
        # Frame shape should be lines-first, so w and h are swapped
        h, w, pixfmt = frame_shape
        if pixfmt != 3 and pixfmt != 4:
            raise InvalidFrame("Your frame has shape {}, but we require (w,h,3) or (w,h,4), i.e., "
                               "RGB values for a w-by-h image, with an optional alpha channel.".format(frame_shape))
        self.wh = (w,h)
        self.includes_alpha = (pixfmt == 4)
        self.frame_shape = frame_shape
        self.frames_per_sec = frames_per_sec
        self.output_frames_per_sec = output_frames_per_sec

        if which('avconv') is not None:
            self.backend = 'avconv'
        elif which('ffmpeg') is not None:
            self.backend = 'ffmpeg'
        else:
            raise error.DependencyNotInstalled("""Found neither the ffmpeg nor avconv executables. 
            On OS X, you can install ffmpeg via `brew install ffmpeg`. On most Ubuntu variants, 
            `sudo apt-get install ffmpeg` should do it. On Ubuntu 14.04, however, you'll need to install avconv 
            with `sudo apt-get install libav-tools`.""")

        self.start()

    @property
    def version_info(self):
        return {
            'backend':self.backend,
            'version':str(subprocess.check_output([self.backend, '-version'],
                                                  stderr=subprocess.STDOUT)),
            'cmdline':self.cmdline
        }

    def start(self):
        self.cmdline = (self.backend,
                     '-nostats',
                     '-loglevel', 'error', # suppress warnings
                     '-y',

                     # input
                     '-f', 'rawvideo',
                     '-s:v', '{}x{}'.format(*self.wh),
                     '-pix_fmt',('rgb32' if self.includes_alpha else 'rgb24'),
                     '-framerate', '%d' % self.frames_per_sec,
                     '-i', '-', # this used to be /dev/stdin, which is not Windows-friendly

                     # output
                     '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
                     '-vcodec', 'libx264',
                     '-pix_fmt', 'yuv420p',
                     '-r', '%d' % self.output_frames_per_sec,
                     self.output_path
                     )

        logger.debug('Starting ffmpeg with "%s"', ' '.join(self.cmdline))
        if hasattr(os,'setsid'): #setsid not present on Windows
            self.proc = subprocess.Popen(self.cmdline, stdin=subprocess.PIPE, preexec_fn=os.setsid)
        else:
            self.proc = subprocess.Popen(self.cmdline, stdin=subprocess.PIPE)

    def capture_frame(self, frame):
        if not isinstance(frame, (np.ndarray, np.generic)):
            raise InvalidFrame('Wrong type {} for {} (must be np.ndarray or np.generic)'.format(type(frame), frame))
        if frame.shape != self.frame_shape:
            raise InvalidFrame("Your frame has shape {}, but the VideoRecorder is configured for shape {}.".format(
                frame.shape, self.frame_shape))
        if frame.dtype != np.uint8:
            raise InvalidFrame("Your frame has data type {}, but we require uint8 (i.e. RGB values from 0-255).".format(
                frame.dtype))

        self.proc.stdin.write(frame.tobytes())

    def close(self):
        self.proc.stdin.close()
        ret = self.proc.wait()
        if ret != 0:
            logger.error("VideoRecorder encoder exited with status {}".format(ret))
