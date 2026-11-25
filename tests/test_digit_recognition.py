import functools
import json
import sys

import cv2 as cv
import numpy as np

from tests import PVITestCase

NUM_CLASSES = 10
NUM_FEATURES = 64


@functools.cache
def load_dataset() -> tuple[np.ndarray, np.ndarray]:
    from sklearn.datasets import load_digits

    digits = load_digits()
    return digits.data, digits.target


@functools.cache
def load_image(path: str) -> np.ndarray:
    return cv.cvtColor(cv.imread(PVITestCase.rpath(path)), cv.COLOR_BGR2RGB)


def load_reference() -> tuple[np.ndarray, np.ndarray]:
    """
    Reference bounding boxes (x1, y1, x2, y2) of the ink of the digits in the image (Otsu threshold and connected
    components) and their labels (row of the grid).
    """
    with open(PVITestCase.rpath("tests/data/digit_recognition-digits.json")) as f:
        digits = json.load(f)
    return np.array([d["bbox"] for d in digits], dtype=np.float64), np.array([d["label"] for d in digits])


def match_boxes(boxes: np.ndarray, ref_bboxes: np.ndarray) -> np.ndarray:
    """
    For each reference digit, the index of the box containing its center and having its own center inside the digit's
    bounding box (-1 if none or several boxes do).
    """
    ref_centers = (ref_bboxes[:, :2] + ref_bboxes[:, 2:]) / 2
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    inside = (
        (boxes[:, None, :2] <= ref_centers[None, :, :]).all(axis=2)
        & (boxes[:, None, 2:] >= ref_centers[None, :, :]).all(axis=2)
        & (ref_bboxes[None, :, :2] <= centers[:, None, :]).all(axis=2)
        & (ref_bboxes[None, :, 2:] >= centers[:, None, :]).all(axis=2)
    )
    return np.where(inside.sum(axis=0) == 1, inside.argmax(axis=0), -1)


class DigitsTestCase(PVITestCase):
    def check_no_libraries(self, *fns) -> None:
        # sklearn may be imported in the notebook for loading the dataset, but must not be used in the functions
        imported = sorted(m for m in ("torch", "tensorflow", "keras", "jax", "xgboost", "lightgbm") if m in sys.modules)
        self.assertEqual(imported, [], msg=f"Libraries {imported} are not allowed, implement the classifier in numpy")
        for fn in fns:
            try:
                source = self.get_source(fn)
            except (OSError, TypeError):
                continue
            for module in ("sklearn", "torch", "tensorflow", "keras", "jax", "xgboost", "lightgbm", "scipy"):
                self.assertNotIn(module, source, msg=f"Function {fn.__qualname__} must not use {module}")

    def predict(self, inputs: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
        predicted = np.asarray(self.params["predict_fn"](inputs, weights, bias)).ravel()
        self.assertEqual(predicted.shape, (len(inputs),), msg="predict must return one label per sample")
        return predicted

    def accuracy(self, predicted: np.ndarray, labels: np.ndarray) -> float:
        accuracy = self.params["accuracy_fn"](predicted.copy(), labels.copy())
        self.assertEqual(np.ndim(accuracy), 0, msg="accuracy must return a single number")
        return float(accuracy)

    def parameters(self) -> tuple[np.ndarray, np.ndarray]:
        weights = np.asarray(self.params["weights"], dtype=np.float64)
        bias = np.asarray(self.params["bias"], dtype=np.float64).ravel()
        self.assertEqual(weights.shape, (NUM_FEATURES, NUM_CLASSES), msg="weights must be of shape (64, 10)")
        self.assertEqual(bias.shape, (NUM_CLASSES,), msg="bias must be of shape (10,)")
        self.assertTrue(np.isfinite(weights).all() and np.isfinite(bias).all(), msg="Parameters contain NaN or inf")
        return weights, bias


class TestSplit(DigitsTestCase):
    """The split function on dummy data."""

    num_samples = 200

    def split(self, val_fraction: float = 0.2, seed: int = 0):
        inputs = np.arange(self.num_samples * 3, dtype=np.float64).reshape(self.num_samples, 3)
        labels = np.arange(self.num_samples) % NUM_CLASSES
        result = self.params["split_fn"](inputs.copy(), labels.copy(), val_fraction=val_fraction, seed=seed)
        self.assertEqual(
            len(result), 4, msg="split_data must return (inputs_train, inputs_val, labels_train, labels_val)"
        )
        inputs_train, inputs_val, labels_train, labels_val = (np.asarray(a) for a in result)
        self.assertEqual(
            inputs_train.ndim == 2 and inputs_train.shape[1], 3, msg="inputs_train must be of shape (N_train, D)"
        )
        self.assertEqual(inputs_val.ndim == 2 and inputs_val.shape[1], 3, msg="inputs_val must be of shape (N_val, D)")
        self.assertEqual(labels_train.shape, (len(inputs_train),), msg="labels_train must have one label per sample")
        self.assertEqual(labels_val.shape, (len(inputs_val),), msg="labels_val must have one label per sample")
        # the rows of the dummy inputs are 3 * index + (0, 1, 2)
        return inputs_train[:, 0].astype(int) // 3, inputs_val[:, 0].astype(int) // 3, labels_train, labels_val

    def test_partition(self):
        train, val, labels_train, labels_val = self.split()
        self.assertEqual(
            len(val), round(0.2 * self.num_samples), msg="Validation set must have val_fraction of samples"
        )
        self.assertEqual(
            sorted(np.concatenate([train, val]).tolist()),
            list(range(self.num_samples)),
            msg="Each sample must be either in the training or in the validation set, exactly once",
        )
        self.assertTrue(
            (labels_train == train % NUM_CLASSES).all() and (labels_val == val % NUM_CLASSES).all(),
            msg="Labels must be shuffled together with the samples",
        )
        self.assertEqual(len(self.split(val_fraction=0.5)[1]), self.num_samples // 2, msg="val_fraction is not used")

    def test_random(self):
        _, val, _, _ = self.split()
        self.assertFalse(
            set(val.tolist()) in ({*range(len(val))}, {*range(self.num_samples - len(val), self.num_samples)}),
            msg="The samples must be shuffled before splitting",
        )
        self.assertTrue((self.split(seed=0)[1] == val).all(), msg="The same seed must give the same split")
        self.assertFalse(
            set(self.split(seed=1)[1].tolist()) == set(val.tolist()), msg="Different seeds must give different splits"
        )


class TestData(DigitsTestCase):
    """The training and validation data of the notebook are the digits dataset."""

    def test_data(self):
        dataset_inputs, dataset_labels = load_dataset()
        inputs = {k: np.asarray(self.params[f"inputs_{k}"], dtype=np.float64) for k in ("train", "val")}
        labels = {k: np.asarray(self.params[f"labels_{k}"]).ravel() for k in ("train", "val")}
        for k in ("train", "val"):
            self.assertEqual(
                inputs[k].ndim == 2 and inputs[k].shape[1], NUM_FEATURES, msg=f"inputs_{k} must be of shape (N, 64)"
            )
            self.assertEqual(labels[k].shape, (len(inputs[k]),), msg=f"labels_{k} must have one label per sample")
        num = len(dataset_inputs)
        self.assertEqual(len(inputs["train"]) + len(inputs["val"]), num, msg=f"Use all {num} samples of the dataset")
        self.assertTrue(
            0.1 * num <= len(inputs["val"]) <= 0.4 * num, msg="Use 10 to 40 % of the samples for validation"
        )
        # the samples may be scaled by a constant (e.g. to the range 0..1)
        scale = dataset_inputs.max() / max(inputs["train"].max(), 1e-12)
        self.assertTrue(
            np.allclose(inputs["val"].max() * scale, dataset_inputs.max()),
            msg="inputs_train and inputs_val must be scaled the same way",
        )
        # the samples of the dataset are all different
        index = {row.tobytes(): i for i, row in enumerate(dataset_inputs)}
        used = []
        for k in ("train", "val"):
            rows = np.round(inputs[k] * scale)
            self.assertTrue(
                np.allclose(rows, inputs[k] * scale, atol=1e-6), msg=f"inputs_{k} are not samples of the dataset"
            )
            for row, label in zip(rows, labels[k], strict=True):
                i = index.get(row.tobytes())
                self.assertIsNotNone(i, msg=f"inputs_{k} contains a sample, which is not in the dataset")
                self.assertEqual(label, dataset_labels[i], msg=f"labels_{k} do not correspond to the samples")
                used.append(i)
        self.assertEqual(len(set(used)), num, msg="Each sample must be used exactly once")


class TestPredict(DigitsTestCase):
    # Hard-coded inputs and expected values (no reference implementation in the tests)
    predict_labels = np.array([0, 1, 1, 1, 2])

    def test_predict(self):
        predicted = self.predict(
            np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.5], [0.0, -1.0]]),
            np.array([[1.0, -1.0, 0.0], [0.0, 2.0, -1.0]]),
            np.array([0.0, 0.5, 0.2]),
        )
        self.assertTrue(
            (predicted == self.predict_labels).all(),
            msg=f"Expected labels {self.predict_labels.tolist()}, got {predicted.tolist()}. Is the predicted class the "
            "one with the highest score?",
        )
        self.assertNoLoops(self.params["predict_fn"])

    def test_accuracy(self):
        for predicted, labels, expected in (
            (np.array([0, 1, 1, 1, 2]), np.array([0, 1, 2, 1, 0]), 0.6),
            (np.array([3, 3, 3, 3]), np.array([3, 3, 3, 3]), 1.0),
            (np.array([1, 2, 0]), np.array([2, 0, 1]), 0.0),
            (np.array([9, 4, 7, 4, 0, 5, 5, 1]), np.array([9, 4, 1, 4, 0, 5, 3, 1]), 0.75),
        ):
            accuracy = self.accuracy(predicted, labels)
            self.assertAlmostEqual(
                accuracy,
                expected,
                places=9,
                msg=f"Expected accuracy {expected} for predicted {predicted.tolist()} and labels {labels.tolist()}, "
                f"got {accuracy}. Is it the fraction (0 to 1) of correctly classified samples?",
            )
        self.check_no_libraries(self.params["accuracy_fn"])
        self.assertNoLoops(self.params["accuracy_fn"])


class TestLoss(DigitsTestCase):
    # Hard-coded inputs (no reference implementation in the tests)
    inputs = np.array([[0.0, 0.5, 1.0], [1.0, 0.25, 0.0], [0.5, 0.5, 0.5], [0.0, 1.0, 0.75]])
    labels = np.array([0, 2, 1, 2])
    weights = np.array([[0.5, -1.0, 0.25], [1.0, 0.0, -0.5], [-0.25, 0.75, 1.5]])
    bias = np.array([0.1, -0.2, 0.3])
    # Maximal relative error of the analytic gradient with respect to the numerical one
    max_relative_error = 1e-5

    def loss(self, inputs, labels, weights, bias) -> tuple[float, np.ndarray, np.ndarray]:
        result = self.params["loss_fn"](inputs.copy(), labels.copy(), weights.copy(), bias.copy())
        self.assertEqual(len(result), 3, msg="softmax_loss must return (loss, grad_weights, grad_bias)")
        loss = float(result[0])
        grad_weights = np.asarray(result[1], dtype=np.float64)
        grad_bias = np.asarray(result[2], dtype=np.float64)
        self.assertEqual(grad_weights.shape, weights.shape, msg="grad_weights must have the same shape as weights")
        self.assertEqual(grad_bias.shape, bias.shape, msg="grad_bias must have the same shape as bias")
        return loss, grad_weights, grad_bias

    def test_loss(self):
        loss, _, _ = self.loss(self.inputs, self.labels, self.weights, self.bias)
        self.assertAlmostEqual(loss, 1.412172839860896, places=6, msg="Loss must be the mean cross-entropy")
        self.check_no_libraries(self.params["loss_fn"])
        self.assertNoLoops(self.params["loss_fn"])

    def test_stability(self):
        with np.errstate(all="ignore"):
            loss, grad_weights, grad_bias = self.loss(self.inputs, self.labels, 1000 * self.weights, self.bias)
        self.assertTrue(
            np.isfinite(loss) and np.isfinite(grad_weights).all() and np.isfinite(grad_bias).all(),
            msg="Loss or gradient is not finite for large scores",
        )

    def test_gradient(self):
        weights, bias = self.weights.copy(), self.bias.copy()
        _, grad_weights, grad_bias = self.loss(self.inputs, self.labels, weights, bias)
        for name, param, grad in (("weights", weights, grad_weights), ("bias", bias, grad_bias)):
            # central differences of the student's own loss
            numerical = np.zeros_like(param)
            eps = 1e-5
            for i in np.ndindex(param.shape):
                old = param[i]
                param[i] = old + eps
                plus = self.loss(self.inputs, self.labels, weights, bias)[0]
                param[i] = old - eps
                minus = self.loss(self.inputs, self.labels, weights, bias)[0]
                param[i] = old
                numerical[i] = (plus - minus) / (2 * eps)
            error = np.abs(grad - numerical).max() / max(np.abs(numerical).max(), 1e-8)
            print(f"Relative error of grad_{name} {error:.2e}")
            self.assertLess(
                error,
                self.max_relative_error,
                msg=f"grad_{name} does not match the numerical gradient of the loss (relative error {error:.2e})",
            )


class TestAccuracy(DigitsTestCase):
    """The trained parameters of the notebook on the training and validation data."""

    min_train_accuracy = 0.95
    min_val_accuracy = 0.93

    def test_accuracy(self):
        self.check_no_libraries(self.params["predict_fn"], self.params["accuracy_fn"])
        weights, bias = self.parameters()
        for split, min_accuracy in (("train", self.min_train_accuracy), ("val", self.min_val_accuracy)):
            inputs = np.asarray(self.params[f"inputs_{split}"], dtype=np.float64)
            labels = np.asarray(self.params[f"labels_{split}"]).ravel()
            accuracy = self.accuracy(self.predict(inputs, weights, bias), labels)
            print(f"Accuracy on {split} data {accuracy:.3f}")
            self.assertGreaterEqual(
                accuracy,
                min_accuracy,
                msg=f"Low accuracy {accuracy:.3f} on {split} data, at least {min_accuracy} is required. Tune the "
                "learning rate and the number of iterations.",
            )


class TestOwnImage(DigitsTestCase):
    # The image with 10 x 10 digits
    image_path = "data/digits.jpg"
    # The extraction is tested on the image of the assignment and on its downscaled version
    scales = (1.0, 0.5)
    min_accuracy = 0.5

    def test_accuracy(self):
        weights, bias = self.parameters()
        ref_bboxes, ref_labels = load_reference()
        for scale in self.scales:
            with self.subTest(scale=scale):
                rgb = load_image(self.image_path)
                if scale != 1.0:
                    rgb = cv.resize(rgb, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
                result = self.params["extract_fn"](rgb.copy())
                self.assertEqual(len(result), 2, msg="extract_digits must return a pair (boxes, inputs)")
                boxes = np.asarray(result[0], dtype=np.float64)
                inputs = np.asarray(result[1], dtype=np.float64)
                num = len(ref_bboxes)
                self.assertEqual(boxes.shape, (num, 4), msg=f"Scale {scale}: expected {num} boxes (x1, y1, x2, y2)")
                self.assertEqual(
                    inputs.shape, (num, NUM_FEATURES), msg=f"Scale {scale}: expected inputs of shape ({num}, 64)"
                )
                matched = match_boxes(boxes, ref_bboxes * scale)
                self.assertTrue(
                    (matched >= 0).all() and len(set(matched)) == num,
                    msg=f"Scale {scale}: {np.sum(matched < 0)} digits do not correspond to exactly one box. Are all "
                    "digits found as single boxes? Do the thresholds depend on the size of the image?",
                )
                labels = np.empty(num, dtype=np.int64)
                labels[matched] = ref_labels
                accuracy = self.accuracy(self.predict(inputs, weights, bias), labels)
                print(f"Scale {scale}: accuracy {accuracy:.2f} on {self.image_path}")
                self.assertGreaterEqual(
                    accuracy,
                    self.min_accuracy,
                    msg=f"Scale {scale}: low accuracy {accuracy:.2f} on {self.image_path}. Do the samples look like "
                    "the samples of the dataset (size, position, range of values, bright digit on dark background)?",
                )
