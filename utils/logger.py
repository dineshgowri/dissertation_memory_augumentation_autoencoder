# utils/logger.py
from __future__ import absolute_import, print_function
import tensorflow as tf
import numpy as np
from io import BytesIO
from PIL import Image

# Make TF2 behave like TF1 for summaries
tf.compat.v1.disable_eager_execution()


class Logger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.compat.v1.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.compat.v1.Summary(
            value=[tf.compat.v1.Summary.Value(tag=tag, simple_value=value)]
        )
        self.writer.add_summary(summary, step)

    def image_summary(self, tag, images, step):
        """Log images if valid; otherwise log as histogram."""
        img_summaries = []

        for i, img in enumerate(images):
            arr = np.array(img)

            # Decide if this looks like an image
            is_image = (
                arr.ndim == 2 or
                (arr.ndim == 3 and arr.shape[-1] in [1, 3])
            )

            if is_image:
                arr = np.clip(arr, 0, 1)  # [0,1]
                arr_uint8 = (arr * 255).astype(np.uint8)

                # grayscale expand to RGB
                if arr_uint8.ndim == 2:
                    arr_uint8 = np.expand_dims(arr_uint8, -1)
                if arr_uint8.shape[-1] == 1:
                    arr_uint8 = np.repeat(arr_uint8, 3, axis=-1)

                s = BytesIO()
                Image.fromarray(arr_uint8).save(s, format="PNG")

                img_sum = tf.compat.v1.Summary.Image(
                    encoded_image_string=s.getvalue(),
                    height=arr_uint8.shape[0],
                    width=arr_uint8.shape[1],
                )
                img_summaries.append(
                    tf.compat.v1.Summary.Value(tag=f"{tag}/{i}", image=img_sum)
                )

            else:
                # Fallback: log as histogram instead of crashing
                counts, bin_edges = np.histogram(arr, bins=100)
                hist = tf.compat.v1.HistogramProto()
                hist.min = float(np.min(arr))
                hist.max = float(np.max(arr))
                hist.num = int(np.prod(arr.shape))
                hist.sum = float(np.sum(arr))
                hist.sum_squares = float(np.sum(arr ** 2))
                bin_edges = bin_edges[1:]
                for edge in bin_edges:
                    hist.bucket_limit.append(edge)
                for c in counts:
                    hist.bucket.append(c)
                img_summaries.append(
                    tf.compat.v1.Summary.Value(tag=f"{tag}_hist/{i}", histo=hist)
                )

        if img_summaries:
            summary = tf.compat.v1.Summary(value=img_summaries)
            self.writer.add_summary(summary, step)

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram explicitly."""
        counts, bin_edges = np.histogram(values, bins=bins)
        hist = tf.compat.v1.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))
        bin_edges = bin_edges[1:]
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)
        summary = tf.compat.v1.Summary(
            value=[tf.compat.v1.Summary.Value(tag=tag, histo=hist)]
        )
        self.writer.add_summary(summary, step)
        self.writer.flush()

