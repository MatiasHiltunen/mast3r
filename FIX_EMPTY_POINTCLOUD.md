# Fix for Empty Pointcloud Error

## Issue
When running the MASt3R demo with certain images, especially those with:
- Very different resolutions (e.g., 1324x1766 and 160x214)
- Poor quality or insufficient overlap
- Images that fail to produce valid 3D reconstructions

The following error would occur:
```
ValueError: zero-size array to reduction operation maximum which has no identity
```

## Root Cause
The error occurs when the 3D reconstruction process produces no valid points (all points are NaN or filtered out by the confidence threshold), and trimesh tries to export an empty pointcloud/mesh to GLB format.

## Fix Applied
The fix adds proper validation and error handling in `mast3r/demo.py`:

1. **Check for valid points before creating geometry** - Only create pointclouds/meshes if there are valid points after filtering
2. **Validate scene before export** - Check if the scene has any valid geometry (not just cameras) before attempting export
3. **Catch export errors** - Handle ValueError exceptions from trimesh when exporting scenes with invalid geometry
4. **Provide informative warnings** - Display helpful messages when the reconstruction fails

## How to Avoid This Issue

1. **Use similar resolution images** - Try to use images with comparable resolutions
2. **Ensure good image overlap** - Images should have sufficient overlap for reconstruction
3. **Adjust confidence threshold** - Lower the `min_conf_thr` parameter if too many points are being filtered out
4. **Check image quality** - Ensure images are not blurry or too small
5. **Use appropriate image sizes** - Consider using `--image_size 224` for faster processing and lower memory usage

## Example Usage
```bash
# If you encounter empty pointcloud issues, try:
python demo.py --model_name MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric --device mps --image_size 224
```

Then in the UI, adjust the `min_conf_thr` slider to a lower value (e.g., 0.5 or 1.0) to include more points in the reconstruction. 