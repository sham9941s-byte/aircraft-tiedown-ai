# Annotation contract

The annotation layer should support both bounding boxes and instance masks.

For each object:

- `class`
- `bbox`
- optional `mask`
- `visibility`
- `occlusion`
- `quality` when applicable
- `connection_id` for relationship reasoning where applicable

Relationships are important because a strap existing in an image is not enough; the system must reason about whether it is connected to the correct engine/trailer connection point.
