import cv2 as cv
import numpy as np

from tests import PVITestCase

IMAGE_SHAPE = (499, 640)

# Manually annotated ball outline in the YOLO segmentation format: class x1 y1 x2 y2 ... (normalized coordinates)
REFERENCE_PATH = "data/tennis_ball2_small.txt"

# Bounding box (x, y, w, h) and centroid (cx, cy) of the manually annotated ball in pixels
BALL_BBOX = (95, 90, 286, 280)
BALL_CENTROID = (241.6, 222.3)


def load_reference_mask() -> np.ndarray:
    with open(PVITestCase.rpath(REFERENCE_PATH)) as f:
        values = f.readline().split()
    height, width = IMAGE_SHAPE
    polygon = np.array(values[1:], dtype=np.float64).reshape(-1, 2) * (width, height)
    mask = np.zeros(IMAGE_SHAPE, dtype=np.uint8)
    cv.fillPoly(mask, [np.round(polygon).astype(np.int32)], 255)
    return mask > 0


def mask_iou(mask: np.ndarray, expected_mask: np.ndarray) -> float:
    return float((mask & expected_mask).sum() / (mask | expected_mask).sum())


def bbox_iou(bbox: tuple, expected_bbox: tuple) -> float:
    x1, y1, w1, h1 = bbox
    x2, y2, w2, h2 = expected_bbox
    iw = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    ih = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    return float(iw * ih / (w1 * h1 + w2 * h2 - iw * ih))


class TestLoadImage(PVITestCase):
    def test_image(self):
        rgb = self.params["rgb"]
        self.assertIsInstance(rgb, np.ndarray, msg="Image must be a numpy array")
        self.assertEqual(rgb.shape, (*IMAGE_SHAPE, 3), msg="Image has unexpected shape")
        self.assertEqual(rgb.dtype, np.uint8, msg="Image must be of type uint8")

    def test_channel_order(self):
        rgb = self.params["rgb"]
        # The ball is yellow-green and has almost no blue, so swapped channels show up as red < blue
        r, _, b = rgb[load_reference_mask()].mean(axis=0)
        self.assertGreater(r, b, msg="The ball must have more red than blue. Is the image still in BGR order?")


class TestBboxAndCentroid(PVITestCase):
    def setUp(self):
        self.mask = cv.imread(self.rpath("tests/data/colors-bbox_and_centroid_input.png"), cv.IMREAD_GRAYSCALE)
        self.expected_bbox = (30, 20, 226, 166)
        self.expected_centroid = (118.52, 131.61)

    def test_bbox(self):
        bbox, _ = self.params["bbox_and_centroid_fn"](self.mask)
        self.assertEqual(len(bbox), 4, msg="Bounding box must be a 4-tuple (x, y, w, h)")
        self.assertArraysClose(
            np.array(bbox, dtype=np.float64),
            np.array(self.expected_bbox, dtype=np.float64),
            rtol=0,
            atol=0.5,
            msg="bbox_and_centroid returned incorrect bounding box (x, y, w, h)",
        )

    def test_centroid(self):
        _, centroid = self.params["bbox_and_centroid_fn"](self.mask)
        self.assertEqual(len(centroid), 2, msg="Centroid must be a pair (cx, cy)")
        self.assertArraysClose(
            np.array(centroid, dtype=np.float64),
            np.array(self.expected_centroid, dtype=np.float64),
            rtol=0,
            atol=0.5,
            msg="bbox_and_centroid returned incorrect centroid (cx, cy)",
        )


class TestSegmentation(PVITestCase):
    # Minimal Jaccard index (IoU) of the mask and of the bounding box w.r.t. the reference
    min_mask_iou = 0.0
    min_bbox_iou = 0.0
    # Maximal distance of the centroid from the reference centroid in pixels
    max_centroid_distance = 0.0

    def test_mask(self):
        mask = self.params["mask"]
        self.assertIsInstance(mask, np.ndarray, msg="Mask must be a numpy array")
        self.assertEqual(mask.shape, IMAGE_SHAPE, msg="Mask has unexpected shape")
        self.assertLessEqual(len(np.unique(mask)), 2, msg="Mask must be binary")

        iou = mask_iou(mask > 0, load_reference_mask())
        print(f"Mask IoU: {iou:.3f}")
        self.assertGreaterEqual(
            iou,
            self.min_mask_iou,
            msg=f"Mask too different from the reference (IoU={iou:.3f} < {self.min_mask_iou})",
        )

    def test_bbox(self):
        bbox = self.params["bbox"]
        self.assertEqual(len(bbox), 4, msg="Bounding box must be a 4-tuple (x, y, w, h)")

        iou = bbox_iou(bbox, BALL_BBOX)
        print(f"Bounding box IoU: {iou:.3f}")
        self.assertGreaterEqual(
            iou,
            self.min_bbox_iou,
            msg=f"Bounding box {tuple(bbox)} too different from the reference {BALL_BBOX} "
            f"(IoU={iou:.3f} < {self.min_bbox_iou})",
        )

    def test_centroid(self):
        centroid = self.params["centroid"]
        self.assertEqual(len(centroid), 2, msg="Centroid must be a pair (cx, cy)")

        distance = float(np.hypot(centroid[0] - BALL_CENTROID[0], centroid[1] - BALL_CENTROID[1]))
        print(f"Centroid distance: {distance:.1f} px")
        self.assertLessEqual(
            distance,
            self.max_centroid_distance,
            msg=f"Centroid ({centroid[0]:.1f}, {centroid[1]:.1f}) too far from the reference {BALL_CENTROID} "
            f"({distance:.1f} px > {self.max_centroid_distance} px)",
        )


class TestSegmentationRGB(TestSegmentation):
    min_mask_iou = 0.75
    min_bbox_iou = 0.85
    max_centroid_distance = 20.0


class TestSegmentationHSV(TestSegmentation):
    # Brightness thresholding removes also the shaded part of the ball, hence the looser bounding box limit
    min_mask_iou = 0.75
    min_bbox_iou = 0.75
    max_centroid_distance = 20.0


class TestSegmentationBackprojection(TestSegmentation):
    # Histogram of a small region does not cover all shades of the ball, hence the looser limits
    min_mask_iou = 0.7
    min_bbox_iou = 0.7
    max_centroid_distance = 30.0


class TestHueHistogram(PVITestCase):
    def test_format(self):
        hist = self.params["hist"]
        self.assertIsInstance(hist, np.ndarray, msg="Histogram must be a numpy array")
        self.assertEqual(hist.squeeze().ndim, 1, msg="Histogram must be one-dimensional (hue only)")
        self.assertEqual(hist.size, 180, msg="Hue histogram must have 180 bins")

    def test_normalization(self):
        values = self.params["hist"].ravel()
        self.assertGreaterEqual(values.min(), 0, msg="Histogram must not contain negative values")
        self.assertAlmostEqual(
            float(values.max()), 255, delta=1, msg="Histogram must be normalized to the range 0..255"
        )

    def test_peak(self):
        peak = int(self.params["hist"].ravel().argmax())
        print(f"Histogram peak: H={peak}")
        # The yellow-green ball has hue around 35 in OpenCV units (0..179)
        self.assertTrue(
            28 <= peak <= 42,
            msg=f"Histogram peaks at H={peak}, expected a peak between 28 and 42. "
            "Is the reference region inside the ball and is the histogram computed from hue?",
        )


class TestBackprojection(PVITestCase):
    def test_backprojection(self):
        backproj = self.params["backproj"]
        self.assertIsInstance(backproj, np.ndarray, msg="Backprojection must be a numpy array")
        self.assertEqual(backproj.shape, IMAGE_SHAPE, msg="Backprojection has unexpected shape")

        ball = load_reference_mask()
        inside, outside = float(backproj[ball].mean()), float(backproj[~ball].mean())
        print(f"Mean backprojection: {inside:.1f} on the ball, {outside:.1f} on the background")
        self.assertGreater(
            inside, 10 * outside, msg="Backprojection must be much higher on the ball than on the background"
        )
