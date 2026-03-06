Place offline-generated template tensors here when available:

- `mean_xyz.npy`
- `std_xyz.npy`
- `mean_feat.npy`
- `std_feat.npy`

These are not required for the MVP because `packages/core/src/template.ts` ships a procedural default template. Swap that default with loaded tensors once you export real templates from `/tools/template_builder`.
