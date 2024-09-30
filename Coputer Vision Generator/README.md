# Computer Vision Generator

## Overview

**Computer Vision Generator** is a cutting-edge project that leverages multiple Large Language Models (LLMs) from Hugging Face and OpenAI to generate high-quality images, refine them, and create videos. This project integrates various models for image generation, enhancement, and sequential video creation to deliver smooth visual outputs.

## Features

- **Image Generation**: Using state-of-the-art LLMs to generate images from textual descriptions.
- **Image Refinement**: Apply further enhancements to the generated images for better resolution and quality.
- **Video Creation**: Sequentially generate frames and compile them into videos, creating dynamic visual content from still images and textual description.

## Model's Task

1. **Hugging Face Models**:
   - Text-to-image generation models.
   - Image refinement and upscaling models.
   - Frame-by-frame generation for video synthesis.

2. **OpenAI Models**:
   - LLMs for generating creative and detailed prompts for the images.
   - Advanced models for enhancing image features and resolution.

## Image Generator Models Used

1. **Hugging Face Models**:
   - **Stable Diffusion 5**: A text-to-image diffusion model capable of generating high-quality, realistic images from textual prompts. <br>

   - **Flux.1**: A custom-trained model for generating stylized artistic images with a focus on creative interpretations from a textual description. <br>

   - **Flux.1 LoRA AntiBlur**: A fine-tuned version of Flux.1 with LoRA (Low-Rank Adaptation) to reduce image blur and enhance sharpness.<br>

   - **Flux.1 Details**: Specializes in adding fine-grained details and textures to generated images, enhancing realism.<br>

   - **Flux.1 FilmPortrait**: A model designed for generating cinematic, portrait-style images with a filmic look and feel.<br>
   
   - **AuraSR V2**: A super-resolution model that enhances low-resolution images by refining details and improving clarity.<br>

2. **OpenAI Models**:
   - **DALL-E 3**: An advanced text-to-image generation model from OpenAI that creates highly detailed and realistic images based on text descriptions.<br>

## Video Generator Models Used

1. **Hugging Face Models**:
   - **CogVideoX-5B**: A video generation model designed for producing high-quality videos from textual descriptions, leveraging advanced neural architectures. <br>
    
   - **I2VGenXL**: A video generation model designed for converting images into videos, enabling seamless animation and storytelling from static visuals. <br>

   - **Stable Video Diffusion I2V XT**: A video generation model designed for creating videos by extending image-to-video capabilities, producing smooth transitions and maintaining high fidelity across frames. <br>



