# Reviewed example output

- [tracks.json](tracks.json): accepted detector boxes, temporary ID `V1`, image-space center points, class labels, and source-relative timestamps.
- [processing_summary.json](processing_summary.json): 180 accepted observations over 180 source frames, processing settings, and limitations. Coverage is not accuracy.
- [render_summary.json](render_summary.json): the 557-frame overview, replay, and hold sequence, including the final video's checksum.

The [main README](../README.md) explains reproduction and interpretation. Finished media is in [docs](../docs/README.md). Missing detections would remain absent; the pipeline does not fabricate observations or infer crash physics.
