import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import matplotlib
from tqdm import tqdm

def draw_bounding_boxes(csv_path, imgs_dir, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Group by image name
    grouped = df.groupby('img_name')

    # Define fixed colors for each author_id
    color_map = {
        3: 'blue',    # 禾荣华
        4: 'green',   # 赵辉
        5: 'red',     # 贺代碧
        7: 'orange'   # 梁春霞
    }

    for img_name, group in tqdm(grouped):
        # Load the image
        img_path = os.path.join(imgs_dir, img_name)
        if not os.path.exists(img_path):
            print(f"Image {img_name} not found.")
            continue

        img = Image.open(img_path)
        fig, ax = plt.subplots(1)
        
        # Set figure size to match image dimensions (inches)
        plt.gcf().set_size_inches(img.width / 100, img.height / 100)  # Assuming 100 DPI
        
        ax.imshow(img, alpha=0.5)  # 设置图像透明度为50%
        ax.axis('off')  # Turn off axis
        
        # Add title with bounding box count
        # ax.set_title(f"(Boxes: {len(group)})")

        # Draw bounding boxes
        for idx, row in group.iterrows():
            # Replace eval with safe parsing using numpy
            bbox_str = row['worker_bbox'].strip('[]')  # Remove brackets
            bbox = np.fromstring(bbox_str, sep=' ')  # Parse space-separated values
            # Get color from author_id
            author_id = row['author_id']
            color = color_map.get(author_id, 'gray')  # Default to gray if author_id not found
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                                     linewidth=10, edgecolor=color, facecolor='none')
            ax.add_patch(rect)

        # Save without padding
        output_path = os.path.join(output_dir, img_name)
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close()

if __name__ == "__main__":
    csv_path = 'datasets/Medical/WorkerData.csv'  # Path to your CSV file
    imgs_dir = 'imgs'  # Directory containing images
    output_dir = 'imgs_edit'  # Output directory for edited images

    draw_bounding_boxes(csv_path, imgs_dir, output_dir)