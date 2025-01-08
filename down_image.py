import cv2

def downsample_image(image_path, scale_factor, output_path):
    # Load the image
    image = cv2.imread(image_path)

    # Check if the image is loaded
    if image is None:
        print("Error: Image not loaded properly.")
        return

    # Get the dimensions of the image
    height, width = image.shape[:2]

    # Calculate the new dimensions
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    # Resize the image to the new dimensions
    downsampled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Save the downsampled image
    cv2.imwrite(output_path, downsampled_image)

    print(f"Downsampled image saved as {output_path}.")

# Example usage
image_path = '/home/cuonghoang/Downloads/mcg-2.0/pre-trained/demos/ILSVRC2012_val_00000502.JPEG'  # Replace with your image path
output_path = '/home/cuonghoang/Downloads/mcg-2.0/pre-trained/demos/ILSVRC2012_val_00000502_down.JPEG'  # Replace with your desired output path
scale_factor = 0.5  # Replace with your desired scale factor (e.g., 0.5 for half size)

downsample_image(image_path, scale_factor, output_path)
