"""BBox-only postprocessing, including real Transformers equivalence when installed."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from PIL import Image

from sam3_runtime.service import SAM3Detector


class BBoxTests(unittest.TestCase):
    def test_detector_uses_boxes_without_mask_postprocessing(self):
        detector = SAM3Detector.__new__(SAM3Detector)
        detector.torch, detector.device, detector.threshold = torch, 'cpu', .3
        class Inputs(dict):
            def to(self, device):
                return self
        detector.processor = Mock(return_value=Inputs(original_sizes=torch.tensor([[20, 30]])))
        detector.processor.post_process_object_detection.return_value = [dict(
            boxes=torch.tensor([[1., 2., 10., 12.], [1., 2., 10., 12.], [20., 10., 30., 20.]]),
            scores=torch.tensor([.9, .8, .7]))]
        detector.model = Mock(return_value=object())
        rows = detector.detect(Image.new('RGB', (30, 20)), ['carrot'])
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['bbox'], [1., 2., 10., 12.])
        detector.processor.post_process_instance_segmentation.assert_not_called()
        detector.processor.post_process_object_detection.assert_called_once_with(
            detector.model.return_value, threshold=.3, target_sizes=[[20, 30]])

    def test_real_postprocessors_produce_identical_boxes_and_scores(self):
        try:
            from transformers.models.sam3.image_processing_sam3_fast import Sam3ImageProcessorFast
        except ImportError:
            self.skipTest('Run this test in rewardbench-sam3 for the real SAM3 processor')
        processor = Sam3ImageProcessorFast()
        torch.manual_seed(4)
        for presence in (None, torch.tensor([[1.], [-1.]])):
            for threshold in (0., .3, 1.):
                with self.subTest(presence=presence is not None, threshold=threshold):
                    outputs = SimpleNamespace(pred_logits=torch.randn(2, 5),
                        pred_boxes=torch.rand(2, 5, 4), pred_masks=torch.randn(2, 5, 8, 8),
                        presence_logits=presence)
                    sizes = [(24, 32), (32, 16)]
                    old = processor.post_process_instance_segmentation(outputs, threshold=threshold, target_sizes=sizes)
                    new = processor.post_process_object_detection(outputs, threshold=threshold, target_sizes=sizes)
                    for a, b in zip(old, new):
                        self.assertTrue(torch.equal(a['boxes'], b['boxes']))
                        self.assertTrue(torch.equal(a['scores'], b['scores']))
                        self.assertNotIn('masks', b)


if __name__ == '__main__':
    unittest.main()
