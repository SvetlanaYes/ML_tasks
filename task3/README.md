# Generate main image of website from company description

This script generates the main image of a company's website based on a brief description of the company. It uses the `diffusers` library to perform stable diffusion and the `prompthero/openjourney` model for image generation.

## Prerequisites
- Python 3.x
- PyTorch
- diffusers

## Installation
1. Open .ipnb file on collab
2. From Edit->Network->Hardware accelerator settings choose GPU for faster execution
3. Add description of company in the 3rd code cell
4. Run colab cells Runtime->Run all
5. The code will be downloaded as image_for_website.py
6. Main image will be generated as main_image.png

## Usage
```
python image_for_website.py -d DESCRIPTION
```

### Arguments
- `-d DESCRIPTION`, `--description DESCRIPTION`: Brief description of the company.

## Example
To generate the main image for a company's website based on the description "A leading technology company specializing in AI solutions", run the following command:
```
python image_for_website.py -d "A leading technology company specializing in AI solutions"
```

After running the script, the generated main image will be saved as `main_image.png` in the current directory.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.