# Colour Pallete of Website

This script takes a URL of a website and captures a screenshot of either the full webpage or just the first page. It then extracts the color palette from the screenshot using K-means clustering and displays the color palette in a window using PyQt5.

## Prerequisites
- Python 3.x
- Selenium
- Pillow
- NumPy
- scikit-learn
- PyQt5
- Google Chrome browser (with corresponding version of chromedriver)

## Installation
1. Clone the repository or download the code file.

2. Install the required dependencies by running the following command:
   ```
   pip install selenium Pillow numpy scikit-learn PyQt5
   ```

3. Download the appropriate version of [chromedriver](https://sites.google.com/a/chromium.org/chromedriver/) that matches your Chrome browser version.

## Usage
```
python extract_colour_pallete_of_url.py [-h] [-m MODE] -u URL [-cn COLOR_NUMBER]
```

### Arguments
- `-h`, `--help`: Show the help message and exit.
- `-m MODE`, `--mode MODE`: Specify the screenshot mode. 'True' for a full screenshot, 'False' for only the first page. (Default: False)
- `-u URL`, `--url URL`: URL of the website to capture.
- `-cn COLOR_NUMBER`, `--color_number COLOR_NUMBER`: Number of colors to extract from the screenshot. (Default: 5)

## Example
To capture a full screenshot of a website and extract the top 5 colors from the screenshot, run the following command:
```
python extract_colour_pallete_of_url.py --mode True --url "https://example.com" --color_number 5
```

To capture only the first page of a website and extract the default number of colors (5), run the following command:
```
python extract_colour_pallete_of_url.py --url "https://example.com"
```
