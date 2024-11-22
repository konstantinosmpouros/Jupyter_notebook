# Computer Vision Generator


## Overview

**Computer Vision Generator** is a cutting-edge project that leverages multiple Large Language Models (LLMs) from Hugging Face and OpenAI, and more, to generate high-quality images, refine them, and create videos. This project integrates various models for image generation, enhancement, and sequential video creation to deliver smooth visual outputs.


## Features

- **Image Generation**: Using state-of-the-art LLMs to generate images from textual descriptions.
- **Image Refinement**: Apply further enhancements to the generated images for better resolution, quality, binary masks and more.
- **Video Creation**: Sequentially generate frames and compile them into videos, creating dynamic visual content from still images and textual description.


## Model's Task

   - Text-to-image generation models.
   - Image binary mask generation.
   - Image upscaling models.
   - Image inpainting models.
   - Text + Image to Video generation.
   - Text to Video generation.


## Image Generator Models Used

1. **Hugging Face Models**

   - **Stable Diffusion 5**: A text-to-image diffusion model capable of generating high-quality, realistic images from textual prompts. <br>

   - **Flux.1**: A custom-trained model for generating stylized artistic images with a focus on creative interpretations from a textual description. <br>

   - **Flux.1 LoRA AntiBlur**: A fine-tuned version of Flux.1 with LoRA (Low-Rank Adaptation) to reduce image blur and enhance sharpness.<br>

   - **Flux.1 Details**: Specializes in adding fine-grained details and textures to generated images, enhancing realism.<br>

   - **Flux.1 FilmPortrait**: A model designed for generating cinematic, portrait-style images with a filmic look and feel.<br>
   
   - **RealVisXL 5 Lightning**: An advanced model designed for generating ultra-realistic images with an emphasis on speed and detail. It excels in rendering complex scenes, making it ideal for creative applications requiring high-quality visuals. <br>
   
   - **LCM Dreamshaper 7**: A model focused on producing imaginative and artistic interpretations from textual prompts. It specializes in generating dream-like, surreal images, allowing for a high degree of creativity and abstraction. <br>

   - **Stable Diffusion 3**: Medium: Designed as a mid-tier version with a balance between processing speed and image quality, ideal for prototyping and generating consistent visuals. <br>
   
   

2. **OpenAI Models**
   - **DALL-E 3**: An advanced text-to-image generation model from OpenAI that creates highly detailed and realistic images based on text descriptions.<br>


## Image Refining Models Used

1. **Upscaling**

   - **AuraSR V2**: A super-resolution model that enhances low-resolution images by refining details and improving clarity.<br>

2. **Binary Mask Generation**

   - **Mask RCNN Resnet50**: A powerful object detection model that identifies and segments objects in images, providing pixel-level masks alongside bounding boxes for accurate localization.<br>

   - **Detectron2**: A robust framework for object detection and segmentation tasks that offers state-of-the-art performance with flexible architecture and support for various model types, including Faster R-CNN and Mask R-CNN.<br>

3. **Inpainting**

   - **Kandinsky2**: An inpainting model that specializes in reconstructing missing or corrupted parts of an image by utilizing deep understanding of texture and color harmony, providing seamless blending with existing parts of the image. <br>

   - **Dreamshaper 8**: A versatile inpainting model known for its ability to refine and reshape image areas, creating natural-looking modifications while preserving the image’s overall aesthetic and consistency. <br>


## Video Generator Models Used

1. **Image to Video**

   - **CogVideoX-5B**: A video generation model designed for producing high-quality videos from textual descriptions and image, leveraging advanced neural architectures. <br>
    
   - **I2VGenXL**: A video generation model designed for converting images into videos, enabling seamless animation and storytelling from static visuals. <br>

   - **Stable Video Diffusion I2V XT**: A video generation model designed for creating videos by extending image-to-video capabilities, producing smooth transitions and maintaining high fidelity across frames. <br>

2. **Text to Video**

   - **AnimateLCM**: A text-to-video generation model that transforms descriptive text prompts into dynamic video sequences, offering high-quality frame generation and realistic motion continuity for engaging visual storytelling. <br>


