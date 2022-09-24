################################
# mosaic.py - 0.4 by Pfaeff
################################

import math
import random

import modules.scripts as scripts
import gradio as gr
from PIL import Image
import numpy as np 
import cv2

from modules import images, devices
from modules.processing import Processed, process_images
from modules.shared import opts, state


def filter_mask(mask, mask_feather_x, mask_feather_y):
    sigma_x = 0.3*((mask_feather_x-1)*0.5 - 1) + 0.8
    kernel_x = cv2.getGaussianKernel(mask_feather_x, sigma_x)

    sigma_y = 0.3*((mask_feather_y-1)*0.5 - 1) + 0.8
    kernel_y = cv2.getGaussianKernel(mask_feather_y, sigma_y)

    mask = cv2.sepFilter2D(mask, ddepth=cv2.CV_64F, kernelX=kernel_x, kernelY=kernel_y).astype(np.float32)
    
    mask = np.clip(mask, 0, 1)

    return mask


def build_rectangular_mask(width, height, mask_border_x, mask_border_y, mask_feather_x, mask_feather_y):
    mask = np.zeros((height, width, 3), dtype=np.float32)
    mask = cv2.rectangle(mask, (mask_border_x, mask_border_y), (width - mask_border_x, height - mask_border_y), color=(1.0, 1.0, 1.0), thickness=cv2.FILLED)

    if (mask_feather_x > 0 ) != (mask_feather_y > 0):
        raise Exception("Mask feather can't be zero in just one dimension!")

    if mask_feather_x > 0 and mask_feather_y > 0:
        return filter_mask(mask, mask_feather_x, mask_feather_y)
    else:
        return mask


def build_elliptical_mask(width, height, mask_border_x, mask_border_y, mask_feather_x, mask_feather_y):
    mask = np.zeros((height, width, 3), dtype=np.float32)

    mask = cv2.ellipse(
        mask, 
        (width//2, height//2), 
        ((width - 2 * mask_border_x) // 2, (height - 2 * mask_border_y) // 2), 
        0, 0, 360,
        color=(1.0, 1.0, 1.0), 
        thickness=cv2.FILLED)

    if (mask_feather_x > 0 ) != (mask_feather_y > 0):
        raise Exception("Mask feather can't be zero in just one dimension!")

    if mask_feather_x > 0 and mask_feather_y > 0:
        return filter_mask(mask, mask_feather_x, mask_feather_y)
    else:
        return mask


def check_positions(image_width, image_height):
    def check_positions_func(position):
        left, top, right, bottom = position

        return left >= 0 and left < image_width and \
               top >= 0 and top < image_height and \
               left < right and top < bottom

    return check_positions_func


def out_of_bounds(p, s, m):
    return p >= m or (p + s) <= 0


def indices_to_position(
    x, y, 
    image_width,
    image_height,    
    patch_width,
    patch_height, 
    overlap_x, 
    overlap_y, 
    randomize_position_x, 
    randomize_position_y):

    left = x * (patch_width - overlap_x) - patch_width // 2
    top  = y * (patch_height - overlap_y) - patch_height // 2

    # TODO prevent this from happening in the first place
    if out_of_bounds(left, patch_width, image_width):
        return None

    # TODO prevent this from happening in the first place
    if out_of_bounds(top, patch_height, image_height):
        return None        

    if randomize_position_x > 0:
        new_left = left + random.randint(0, randomize_position_x) - randomize_position_x // 2

        while (out_of_bounds(new_left, patch_width, image_width)):
            new_left = left + random.randint(0, randomize_position_x) - randomize_position_x // 2

        left = new_left

    if randomize_position_y > 0:
        new_top = top + random.randint(0, randomize_position_y) - randomize_position_y // 2

        while (out_of_bounds(new_top, patch_height, image_height)):
            new_top = top + random.randint(0, randomize_position_y) - randomize_position_y // 2

        top = new_top


    # Determine the region to crop
    right  = left + patch_width
    bottom = top + patch_height             

    return (left, top, right, bottom) 


def generate_row_positions(
    image_width, 
    image_height, 
    patch_width, 
    patch_height, 
    overlap_x, 
    overlap_y, 
    randomize_position_x, 
    randomize_position_y):

    num_tiles_x = int(math.ceil(image_width / (patch_width  - overlap_x))) + 1
    num_tiles_y = int(math.ceil(image_height / (patch_height - overlap_y))) + 1

    positions = []
    for y in range(num_tiles_y):
        for x in range(num_tiles_x):
            position = indices_to_position(
                x, 
                y, 
                image_width,
                image_height,
                patch_width, 
                patch_height, 
                overlap_x, 
                overlap_y, 
                randomize_position_x, 
                randomize_position_y)
            if position:
                positions.append(position)

    return positions


def generate_radial_positions(
    image_width, 
    image_height, 
    patch_width, 
    patch_height, 
    overlap_x, 
    overlap_y, 
    randomize_position_x, 
    randomize_position_y):

    positions = generate_row_positions(
        image_width, 
        image_height, 
        patch_width, 
        patch_height, 
        overlap_x, 
        overlap_y, 
        randomize_position_x, 
        randomize_position_y)

    positions = np.array(positions)

    center_positions = np.array(
        (0.5 * (positions[:, 2] + positions[:, 0]), 
         0.5 * (positions[:, 3] + positions[:, 1]))).T 
    image_center = np.array((image_width / 2, image_height / 2))
    d = center_positions - image_center

    # Compute manhattan distance
    distances = np.sum(np.abs(d), axis=1)
    idx = np.argsort(distances)
    positions = positions[idx, :]

    return positions


class Script(scripts.Script):
    def title(self):
        return "Moisaicing (by Pfaeff)"


    def show(self, is_img2img):
        return is_img2img


    def ui(self, is_img2img):
        if not is_img2img:
            return None

        patch_size         = gr.Slider(label="Patch size",         minimum=64, maximum=1024, step=8,      value=512)
        overlap            = gr.Slider(label="Overlap",            minimum=0,  maximum=0.99, step=0.01,   value=0.75)
        mask_shape         = gr.Radio(label='Mask shape ',         choices=['elliptical', 'rectangular'], value='elliptical', type="index", visible=False)
        mask_border        = gr.Slider(label="Mask border",        minimum=0,  maximum=0.49, step=0.01,   value=0.1)
        order              = gr.Radio(label='Processing order',    choices=['radial', 'row-by-row'],      value='radial',     type="index", visible=False)
        randomize_position = gr.Slider(label="Randomize position", minimum=0,  maximum=0.25, step=0.01,   value=0.06)
        upscale_factor     = gr.Slider(label="Upscale amount",     minimum=1,  maximum=16, step=1, value=1)
        preview_mode       = gr.Checkbox(label='Single patch preview mode', value=False)
        mask_preview       = gr.Checkbox(label='Mask preview mode (white means a pixel is going to get regenerated)', value=False)

        return [patch_size, overlap, mask_shape, mask_border, order, randomize_position, upscale_factor, preview_mode, mask_preview]


    def run(self, p, patch_size, overlap, mask_shape, mask_border, order, randomize_position, upscale_factor, preview_mode, mask_preview):
        if p.seed:
            random.seed(p.seed)

        patch_width = patch_size
        patch_height = patch_size
        overlap_x = int(round(overlap * patch_width))
        overlap_y = int(round(overlap * patch_height))
        randomize_position_x = int(round(randomize_position * patch_width))
        randomize_position_y = int(round(randomize_position * patch_height))
        mask_border_x = int(round(mask_border * patch_width))
        mask_border_y = int(round(mask_border * patch_height))

        mask_feather_x = mask_border_x - 1
        mask_feather_y = mask_border_y - 1

        # TODO allow user-drawn masks to be used
        # TODO Allow other methods of inpainting_fill

        p.batch_size = 1
        p.batch_count = 1
        p.do_not_save_grid = True
        p.do_not_save_samples = True        
        p.inpaint_full_res = False
        p.inpainting_fill = 1
        p.mask_blur = 0 # We want to control the blur ourselfs for now

        generate_positions = generate_radial_positions if order==0 else generate_row_positions
        build_mask = build_elliptical_mask if mask_shape==0 else build_rectangular_mask

        mask = build_mask(patch_width, patch_height, mask_border_x, mask_border_y, mask_feather_x, mask_feather_y)

        devices.torch_gc()
        ############################################################

        img = np.asarray(p.init_images[0]).astype(np.float32) / 255.0

        if upscale_factor > 1.0:
            img = cv2.resize(img, None, fx=upscale_factor, fy=upscale_factor)

        if mask_preview:
            img[:, :, :] = 0.0

        image_width = img.shape[1]
        image_height = img.shape[0]

        print("image_width:", image_width)
        print("image_height:", image_height)

        positions = generate_positions(
            image_width, 
            image_height, 
            patch_width, 
            patch_height, 
            overlap_x, 
            overlap_y, 
            randomize_position_x, 
            randomize_position_y)

        if not preview_mode:
            print(f"Mosaicing will process a total of {len(positions)} images tiled as {patch_width}x{patch_height}.")
            state.job_count = len(positions)           
        else:
            print(f"If it were not in preview mode, mosaicing would now process a total of {len(positions)} images tiled as {patch_width}x{patch_height}.")
            state.job_count = 1

        for idx, (left, top, right, bottom) in enumerate(positions):
            pad_left = -left if left < 0 else 0
            pad_top  = -top if top < 0 else 0

            pad_right  = max(0, (right - image_width)) if right > image_width else 0
            pad_bottom = max(0, (bottom - image_height)) if bottom > image_height else 0        

            left = max(0, left)
            top  = max(0, top)
            right  = min(right, image_width)
            bottom = min(bottom, image_height)

            crop = np.copy(img[top:bottom, left:right, :])

            # Add padding if necessary
            if ((pad_left > 0) or (pad_right > 0) or (pad_top > 0) or (pad_bottom > 0)):
                crop = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=0)

            crop_image = Image.fromarray((crop * 255).astype(np.uint8))
            mask_image = Image.fromarray((mask * 255).astype(np.uint8))

            p.init_images = [crop_image]
            p.image_mask = mask_image
            p.latent_mask = mask_image

            state.job = f"Batch {idx + 1} out of {len(positions)}"

            if not mask_preview:
                processed = process_images(p)

                initial_seed = processed.seed
                initial_info = processed.info

                p.seed = processed.seed + 1
                random.seed(p.seed)

                output_images = processed.images
                output_image_np = np.asarray(output_images[0]).astype(np.float32) / 255.0
                output_image_np = cv2.resize(output_image_np, (patch_width, patch_height))
                output_image_np = np.multiply(output_image_np, mask) + np.multiply(crop, 1.0 - mask)

                if len(output_images) == 0:
                    return None
            else:
                initial_seed = -1
                initial_info = ""
                output_image_np = mask
                output_image_np = 1.0 - np.multiply(1.0 - output_image_np, 1.0 - crop)
    
            # Unpad
            if ((pad_left > 0) or (pad_right > 0) or (pad_top > 0) or (pad_bottom > 0)):
                output_image_np = output_image_np[pad_top:patch_height - pad_bottom,
                                                  pad_left:patch_width - pad_right, :]  


            if not preview_mode:
                img[top:bottom, left:right, :] = output_image_np
            else:
                img = output_image_np
                break

        result = Image.fromarray((img * 255).astype(np.uint8))

        if not preview_mode and not mask_preview and opts.samples_save:
            images.save_image(result, p.outpath_samples, "", initial_seed, p.prompt, opts.grid_format, info=initial_info, p=p)

        processed = Processed(p, [result], initial_seed, initial_info)

        return processed
