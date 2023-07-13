from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch

# model_id = "runwayml/stable-diffusion-v1-5"
model_id = "prompthero/openjourney"
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float32)
# pipe = pipe.to("cuda")
prompt = "Yandex LLC is a Russian multinational technology company providing Internet-related products and services, including an Internet search engine,"
# prompt = "Picsart is an all-in-one platform where people can create, customize and share images and videos."
# prompt = "10Web AI Website Builder to create your website 10x faster with AI generated content and images."
# prompt = """Generate  main image of company by it's description. The main image of a website is the the most prominent image in the homepage of the site, usually in the top section of the homepage.
# Description: Mrketing company in USA"""
image = pipe(prompt).images[0]

image.save("MAIN_IMAGE.png")
