"""
Implemented by : Svetlana Yesayan
Description:

    Script extracts colour pallete from given website url

Dependencies & Supported versions:

    See in requirements.txt

Usage:

    See details in README.md

"""
try:
    import sys
    from selenium import webdriver
    import time
    from PIL import Image
    import numpy as np
    from sklearn.cluster import KMeans
    from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
    from PyQt5.QtGui import QColor
    from io import BytesIO
    import argparse
except ImportError as exception:
        print("You should import certain libraries")
        exit(0)


def argument_parser():
    parser = argparse.ArgumentParser("Script extracts colour pallete from given website url")
    parser.add_argument("-m", "--mode", type=bool, required=False, default=False, help="Full screenshot or only the first page of the given website. 'True' for full screenshot, 'False' for first page")
    parser.add_argument("-u", "--url", type=str,  required=True, help="URL of the website")
    parser.add_argument("-cn", "--color_number", type=str, required=False, default=5, help="Number of colours to extract")
    return parser.parse_args()


def take_screenshot(url, mode):
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    driver = webdriver.Chrome(options=options)
    driver.get(url)
    time.sleep(5)
    if mode:
        total_height = driver.execute_script("return document.body.scrollHeight")

        driver.set_window_size(driver.get_window_size()["width"], total_height)

        segment_height = driver.get_window_size()["height"]
        screenshots = []
        for y in range(0, total_height, segment_height):
            if y + segment_height > total_height:
                segment_height = total_height - y
            driver.execute_script(f"window.scrollTo(0, {y});")
            screenshots.append(driver.get_screenshot_as_png())
        screenshot = combine_screenshots(screenshots)
    else:
        screenshot = Image.open(BytesIO(driver.get_screenshot_as_png()))
    driver.quit()
    return screenshot


def combine_screenshots(screenshots):
    combined_image = None

    for screenshot in screenshots:
        image = Image.open(BytesIO(screenshot))
        if combined_image is None:
            combined_image = image
        else:
            combined_image = combine_images(combined_image, image)

    return combined_image


def combine_images(image1, image2):
    new_width = max(image1.width, image2.width)
    new_height = image1.height + image2.height
    combined_image = Image.new("RGB", (new_width, new_height))
    combined_image.paste(image1, (0, 0))
    combined_image.paste(image2, (0, image1.height))

    return combined_image


def save_screenshot(screenshot_bytes, file_path):
    with open(file_path, 'wb') as file:
        file.write(screenshot_bytes)


def extract_color_palette(image_bytes, num_colors):
    image = Image.open(BytesIO(image_bytes))

    if image.mode != "RGB":
        image = image.convert("RGB")

    pixels = list(image.getdata())

    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_

    colors = colors.astype(int)

    return colors


def create_palette_window(palette):
    app = QApplication(sys.argv)
    window = QWidget()
    layout = QVBoxLayout()
    window.setLayout(layout)

    for color in palette:
        color_label = QLabel()
        color_label.setFixedHeight(50)
        color_label.setStyleSheet(f"background-color: rgb({color[0]}, {color[1]}, {color[2]});")
        layout.addWidget(color_label)

    window.show()
    sys.exit(app.exec_())


def main(args):
    screenshot = take_screenshot(args.url, args.mode)
    screenshot_bytes = BytesIO()
    screenshot.save(screenshot_bytes, format="PNG")
    screenshot_path = 'screenshot.png'
    save_screenshot(screenshot_bytes.getvalue(), screenshot_path)
    palette = extract_color_palette(screenshot_bytes.getvalue(), args.color_number)
    create_palette_window(palette)


if __name__ == '__main__':
    args = argument_parser()
    main(args)
