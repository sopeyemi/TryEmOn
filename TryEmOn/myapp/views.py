from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from .serializers import ImageSerializer
from django.core.files.base import ContentFile
import numpy as np
import cv2
from io import BytesIO
from ultralytics import YOLO
import os
from cv2 import rectangle, putText, getTextSize, FONT_HERSHEY_SIMPLEX, LINE_AA, Canny
import copy
from sklearn.cluster import KMeans
import math
import webcolors
import torch
from torchvision import transforms, models
from PIL import Image
import random

# Paths to AI models
file_path = os.path.join('.', 'media', 'clothing_finder.pt')
file_path2 = os.path.join('.', 'media', 'fit_classifier128.pt')

# Class names for object detection
names = ["short-sleeve shirt", "long-sleeve shirt", "short-sleeveoutwear", "long-sleeveoutwear", "pair of shorts", "pair of pants", "skirt", "hat", "shoe"]
model = YOLO(file_path)
have_a_model = True
BOX_COLOR = (0, 255, 0)  # Green
TEXT_COLOR = (255, 255, 255)  # White

class ImageUploadView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        serializer = ImageSerializer(data=request.data)
        if serializer.is_valid():
            # Read the uploaded image file
            image_file = request.FILES['image']
            image_array = np.fromstring(image_file.read(), np.uint8)
            img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            # Rate image
            boxed_img, class_names, color_names, complexity, aesthetic, aesthetics, errors, rating, confidence, overlay_img = rate_my_fit(img)

            # Encode and save
            success, buffer = cv2.imencode('.jpg', boxed_img)
            if not success:
                return Response({"error": "Image processing failed"}, status=400)
            new_image_file = ContentFile(buffer.tobytes(), image_file.name)

            serializer.save(image=new_image_file)

            # Encode and save the overlayed image
            success, overlay_buffer = cv2.imencode('.jpg', overlay_img)
            if not success:
                return Response({"error": "Overlay image processing failed"}, status=400)
            overlay_image_file = ContentFile(overlay_buffer.tobytes(), "overlay.jpg")

            # Prepare JSON response data
            response_data = serializer.data
            response_data['class_names'] = class_names
            response_data['color_names'] = color_names
            response_data['OverallComplexity'] = complexity
            response_data['OverallAesthetic'] = aesthetic
            response_data['Aesthetics'] = aesthetics
            response_data['ColorTheoryErrors'] = errors
            response_data['ai_rating'] = rating
            response_data['confidence'] = confidence
            response_data['overlay_image_url'] = overlay_image_file.url

            return Response(response_data, status=201)  # Return the JSON response with a success status
        return Response(serializer.errors, status=400)  # Return validation errors

# Rates user outfit
# --- rate_my_fit function ---
def rate_my_fit(img):
    outfit = []
    class_names = []

    rating, confidence = ai_rater(img)

    if have_a_model:
        pred = model(img)
        for results in pred:
            box = results.boxes.cpu().numpy()
            for b in box:
                bbox = list(b.xywh[0])
                h, w, channels = img.shape
                bbox[0] *= 1/w
                bbox[1] *= 1/h
                bbox[2] *= 1/w
                bbox[3] *= 1/h
                class_name = names[int(list(b.cls)[0])]
                if float(list(b.conf)[0]) > 0.65:
                    outfit.append((class_name, bbox))
                    class_names.append(class_name)
    else:
        print('Error: No Model Detected.')
        exit()

    color_names, complexity, aesthetic, aesthetics, errors = generateRating(img, outfit)

    # Apply overlays
    overlay_img = img.copy() 
    for class_name, bbox in outfit:
        img = visualize_bbox(img, bbox, class_name)  # Image with boxes

    # Overlay long-sleeve shirt regardless of detected clothing
    long_sleeve_overlay_path = find_clothing_overlay("long-sleeve_shirt", overlay_folder="overlays")
    if long_sleeve_overlay_path:
        # Calculate approximate top area for overlay
        h, w, _ = img.shape
        top_bbox = [0.5, 0.25, 1, 0.5]  # Center at middle, 25% height, full width, 50% height
        overlay_img = overlay_clothing(overlay_img, top_bbox, long_sleeve_overlay_path)

    # Overlay pants regardless of detected clothing
    pants_overlay_path = find_clothing_overlay("pair_of_pants", overlay_folder="overlays")
    if pants_overlay_path:
        # Calculate approximate bottom area for overlay
        h, w, _ = img.shape
        bottom_bbox = [0.5, 0.75, 1, 0.5]  # Center at middle, 75% height, full width, 50% height
        overlay_img = overlay_clothing(overlay_img, bottom_bbox, pants_overlay_path)

    return img, class_names, color_names, complexity, aesthetic, aesthetics, errors, rating, confidence, overlay_img
# Color Theory analysis
def generateRating(img, outfit):
    main_colors = []
    only_colors = []
    complexities = []

    for category, bbox in outfit:
        cropped_article = cropToBbox(img, bbox)
        complexities.append(get_complexity(cropped_article))
        colors = get_colors(cropped_article)

        if colors != None:
            main_colors.append((category, colors[0]))
            only_colors.append((colors[0]))

    color_names = []
    for category, color in main_colors: 
        # print(str(category) + ": " + str(get_colour_name(color)) + ": " + str(color) + "\n")
        color_names.append(str(get_colour_name(color)))

    aesthetics = []

    for color in only_colors:
        color_attributes = [
            color,
            isGloomy(color),
            isNeutral(color),
            isVibrant(color)
        ]
        aesthetics.append(color_attributes)

    errors = []
    for _, c1 in main_colors:
        for _, c2 in main_colors:
            if not areCompatible(c1, c2): 
                errors.append((get_colour_name(c1), get_colour_name(c2)))

    comp = round(np.mean(complexities)*100, 2)
    if(comp > 100):
        comp = 100

    return color_names, (str(comp) + " %"), str(getAesthetic(only_colors)), aesthetics, len(errors)

# Generates an AI rating and the ratings confidence
def ai_rater(img):
    # Transformation pipeline for the input image
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Set up the device and load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(file_path2, map_location=torch.device('cpu'), weights_only=True))
    model.to(device)
    model.eval()

    import torch.nn.functional as F

    def predict_rating(img):
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        image = Image.fromarray(image) # Convert NumPy array to PIL Image
        image = transform(image).unsqueeze(0).to(device) # Apply the transformation pipeline

        with torch.no_grad():
            outputs = model(image)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        # 0 == Bad, 1 == Good
        rating = "Do Better" if predicted.item() == 0 else "Good"
        confidence_percentage = confidence.item() * 100

        # Confidence Rating Buffer
        if(confidence_percentage > 95):
            conf = 'Extremely Confident'
        elif(confidence_percentage <= 95 and confidence_percentage > 90):
            conf = 'Very Confident'
        elif(confidence_percentage <= 90 and confidence_percentage > 80):
            conf = 'Fairly Confident'
        elif(confidence_percentage <= 80 and confidence_percentage > 70):
            conf = 'Relatively Confident'
        elif(confidence_percentage <= 70 and confidence_percentage > 50):
            conf = 'meh Confident'

        return rating, conf

    return predict_rating(img)


# BGR (opposite of RGB, opencv image format) to color name
def get_colour_name(bgr_tuple):

    rgb_tuple = (bgr_tuple[2], bgr_tuple[1], bgr_tuple[0])
    print(rgb_tuple)
    print("get_colour_name")
    
    min_colours = {}
    for name in webcolors.names("css3"):
        r_c, g_c, b_c = webcolors.name_to_rgb(name)
        rd = (r_c - rgb_tuple[0]) ** 2
        gd = (g_c - rgb_tuple[1]) ** 2
        bd = (b_c - rgb_tuple[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def bgr_to_hsv(c):
    b, g, r = c
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = (df/mx)*100
    v = mx*100
    return h, s, v

def isGloomy(color, tolerance=40):
    h,s,v = bgr_to_hsv(color)
    return (v < tolerance)

def isNeutral(color, tolerance=65):
    h,s,v = bgr_to_hsv(color)
    return (s < tolerance)

def isVibrant(color):
    return isBright(color) and (not isNeutral(color))

def isBright(color, tolerance=60):
    h,s,v = bgr_to_hsv(color)
    return (v >= tolerance)

# Returns the overall aesthetic of all of the clothing items found
def getAesthetic(colors, tolerance=0.5):
    aesthetics = [0, 0] #vibrant or gloomy
    for c in colors:
        if isVibrant(c): aesthetics[0] += 1
        elif isGloomy(c): aesthetics[1] += 1
    aesthetics[0] *= 1/len(colors)
    aesthetics[1] *= 1/len(colors)
    if aesthetics[0] > tolerance: return "vibrant"
    elif aesthetics[1] > tolerance: return "gloomy"
    else: return "neutral"

#return True if basically the same color
def areTheSame(c1, c2, tolerance=20):    
    # Convert RGB to XYZ color space
    def rgb_to_xyz(rgb):
        b, g, r = [x / 255.0 for x in rgb]
        r = r ** 2.2
        g = g ** 2.2
        b = b ** 2.2
        
        x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
        y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
        z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
        return (x, y, z)

    # Calculate Delta E
    def delta_e(c1, c2):
        x1, y1, z1 = rgb_to_xyz(c1)
        x2, y2, z2 = rgb_to_xyz(c2)
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)

    delta_e_value = delta_e(c1, c2)
    
    return (delta_e_value * 100) < tolerance

#return True if colors go well together based on color theory/color complements
def areCompatible(c1, c2):
    if areTheSame((255, 0, 0), c1, 40):
        if areTheSame((255, 0, 0), c2, 40): return True
        elif areTheSame((0, 255, 0), c2, 40): return False
        elif areTheSame((0, 0, 255), c2, 40): return False
    elif areTheSame((0, 255, 0), c1, 40):
        if areTheSame((255, 0, 0), c2, 40): return False
        elif areTheSame((0, 255, 0), c2, 40): return True
        elif areTheSame((0, 0, 255), c2, 40): return False
    elif areTheSame((0, 0, 255), c1, 40):
        if areTheSame((255, 0, 0), c2, 40): return False
        elif areTheSame((0, 255, 0), c2, 40): return False
        elif areTheSame((0, 0, 255), c2, 40): return True
    return True

# Finds at most 5 of the most frequent colors in the bboxed image (not always accurate)
def get_colors(img):

    flat_img = np.reshape(img, (-1, 3))

    kmeans = KMeans(n_clusters=5, random_state=0)
    kmeans.fit(flat_img)

    cluster_centers = kmeans.cluster_centers_

    percentages = (np.unique(kmeans.labels_,return_counts=True)[1])/flat_img.shape[0]

    p_and_c = list(zip(percentages,cluster_centers))

    sortedPC = sorted(p_and_c, key=lambda x: x[0], reverse=True)

    colors = []
    for p,c in sortedPC:
        if p > 0.2:
            colors.append(c)

    return colors

# Adds green boxes around the clothing items found
def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    img = copy.deepcopy(img)
    x_center, y_center, w, h = bbox
    height, width, colors = img.shape
    w *= width
    h *= height
    x_center *= width
    y_center *= height
    x_min = x_center - w/2
    y_min = y_center - h/2
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
   
    rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    
    ((text_width, text_height), _) = getTextSize(class_name, FONT_HERSHEY_SIMPLEX, 0.5, 1)    
    rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=FONT_HERSHEY_SIMPLEX,
        fontScale=0.5, 
        color=TEXT_COLOR, 
        lineType=LINE_AA,
    )
    return img

# --- find_clothing_overlay function ---
def find_clothing_overlay(class_name, overlay_folder="overlays"):
    """
    Finds all matching clothing overlay files for a given class name and selects one randomly.

    Args:
        class_name (str): The category of the clothing (e.g., "long-sleeve shirt").
        overlay_folder (str): Path to the folder containing overlay images.

    Returns:
        str: Path to the selected overlay image, or None if no matches are found.
    """
    pattern = class_name.replace(" ", "_")  # Replace spaces with underscores
    matching_files = [
        os.path.join(overlay_folder, f) for f in os.listdir(overlay_folder) 
        if f.startswith(pattern) and (f.endswith(".png") or f.endswith(".jpg"))
    ]
    if not matching_files:
        print(f"No overlay images found for class '{class_name}' in {overlay_folder}.")
        return None

    return random.choice(matching_files)

# --- overlay_clothing function ---
def overlay_clothing(img, bbox, overlay_path):
    """
    Overlays a clothing image onto the input image based on the bounding box.

    Args:
        img (numpy.ndarray): The base image.
        bbox (list): Bounding box [center_x, center_y, width, height] (normalized).
        overlay_img_path (str): Path to the clothing image.

    Returns:
        numpy.ndarray: The image with the overlay applied.
    """
    clothing_img = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)

    if clothing_img is None:
        print(f"Error: Unable to load clothing image from {overlay_path}")
        return img

    h, w, _ = img.shape
    center_x = int(bbox[0] * w)
    center_y = int(bbox[1] * h)
    box_width = int(bbox[2] * w)
    box_height = int(bbox[3] * h)

    top_left_x = int(center_x - box_width / 2)
    top_left_y = int(center_y - box_height / 2)

    resized_clothing = cv2.resize(clothing_img, (box_width, box_height))

    if resized_clothing.shape[2] == 4:  # Check for alpha channel
        clothing_bgr = resized_clothing[:, :, :3]
        clothing_alpha = resized_clothing[:, :, 3] / 255.0  # Normalize alpha

        roi = img[top_left_y:top_left_y + box_height, top_left_x:top_left_x + box_width]

        for c in range(3):  # For each color channel
            roi[:, :, c] = (
                clothing_bgr[:, :, c] * clothing_alpha
                + roi[:, :, c] * (1 - clothing_alpha)
            )

        img[top_left_y:top_left_y + box_height, top_left_x:top_left_x + box_width] = roi
    else:
        img[top_left_y:top_left_y + box_height, top_left_x:top_left_x + box_width] = resized_clothing

    return img

def cropToBbox(img, bbox):
    x_center, y_center, w, h = bbox
    height, width, colors = img.shape
    w *= width
    h *= height
    x_center *= width
    y_center *= height
    x_min = x_center - w/2
    y_min = y_center - h/2
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
    crop_img = img[y_min:y_max, x_min:x_max]
    return crop_img

# More edges in outfit means higher complexity, and vice versa
def get_complexity(img):
    edges = Canny(img,50,150,apertureSize = 3)
    w, h = edges.shape
    return 7*np.sum(edges)/(w*h*255)

    
from django.shortcuts import render

def index(request):
    return render(request, 'index.html')
