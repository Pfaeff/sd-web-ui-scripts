# sd-web-ui-scripts
Custom scripts for the stable diffusion web ui by AUTOMATIC1111

## Mosaicing

The algorithm in action:

https://www.youtube.com/watch?v=t7nopq27uaM

UI:
![UI](/ui.png?raw=true "UI")

### Installation

Download `mosaic.py` and place it in the `stable-diffusion-webui/scripts` folder, then restart webui.

### Usage

- Patches will be resized for processing from `patch size` to whatever your `width` and `height` settings are and then scaled back down for the mosaic.
- elliptical mask shape with lead to less visible seems/patches, but requires a larger overlap in order to not produce gaps
- more overlap means more patches to process and thus longer computation
- This will not change the size of the image. If you want to use this as an upscaler, you have to use any other upscaler before applying this algorithm.


### Results / Examples / Experimentation

https://imgur.com/a/y0A6qO1

### Original reddit thread

https://www.reddit.com/r/StableDiffusion/comments/xa48o6/enhancing_local_detail_and_cohesion_by_mosaicing/
