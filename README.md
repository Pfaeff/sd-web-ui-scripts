# sd-web-ui-scripts
Custom scripts for the [stable diffusion web ui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) by [AUTOMATIC1111](https://github.com/AUTOMATIC1111)

## Mosaicing

![example](/example.png?raw=true "example")

The algorithm in action:

https://www.youtube.com/watch?v=t7nopq27uaM

UI:

![UI](/ui_0.2.png?raw=true "UI")

### Installation

Download `mosaic.py` and place it in the `stable-diffusion-webui/scripts` folder, then restart webui.

### Usage

You can find the algorithm in the img2img tab under "Scripts".

- Patches will be resized for processing from `patch size` to whatever your `width` and `height` settings are and then scaled back down for the mosaic.
- elliptical mask shape will lead to less visible seems/patches, but requires a larger overlap in order to not produce gaps
- more overlap means more patches to process and thus longer computation
- Use preview mode to view how a single patch would look after processing, before committing the entire image.
- "Upscale amount" will upscale the image using bilinear upscaling before processing. For better results, I'd recommend manual upscaling with a better upscaler.


### Results / Examples / Experimentation

https://imgur.com/a/y0A6qO1

### Original reddit thread

https://www.reddit.com/r/StableDiffusion/comments/xa48o6/enhancing_local_detail_and_cohesion_by_mosaicing/
