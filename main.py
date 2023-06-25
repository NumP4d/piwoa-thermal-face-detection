import tensorflow as tf
import numpy as np
from object_detection.utils import visualization_utils as vis_util
import os
import cv2

def bounding_box_thresholded(
    bounding_boxes,
    class_confidences,
    max_boxes=20,
    confidence_threshold=0.5):
    
    output_boxes = []
    if not max_boxes:
        max_boxes_to_draw = bounding_boxes.shape[0]
    for i in range(bounding_boxes.shape[0]):
        if max_boxes == len(output_boxes):
            break
        if class_confidences is None or class_confidences[i] > confidence_threshold:
            output_boxes.append(tuple(bounding_boxes[i].tolist()))
    
    return output_boxes

def denormalize_boxes(
    image_shape,
    bounding_boxes):

    x_size, y_size = (image_shape[1] - 1), (image_shape[0] - 1)
    output_boxes = []
    for i in range(len(bounding_boxes)):
        ymin, xmin, ymax, xmax = bounding_boxes[i]
        xmin = round(xmin * x_size)
        xmax = round(xmax * x_size)
        ymin = round(ymin * y_size)
        ymax = round(ymax * y_size)
        denormalized_box = (ymin, xmin, ymax, xmax)
        output_boxes.append(denormalized_box)

    return output_boxes

def detect_temperatures(
    image,
    bounding_boxes,
    reference_temperature,
    reference_pixel,
    gain,
    min_temp=34,
    max_temp=44):

    temperatures = []
    for box in bounding_boxes:
        ymin, xmin, ymax, xmax = box
        pixels = image[ymin:(ymax+1), xmin:(xmax+1), 1].flatten().astype(float)
        pixels -= reference_pixel
        pixels *= gain
        pixels += reference_temperature
        threshold_indices = (pixels >= min_temp) & (pixels <= max_temp)
        temperature = np.mean(pixels[threshold_indices])
        temperatures.append(temperature)
    
    return temperatures

def reference_box(
    image,
    reference):

    ymin, xmin, ymax, xmax = reference
    pixels = image[ymin:(ymax+1), xmin:(xmax+1), 1].flatten()
    reference_pixel = np.mean(pixels)

    return reference_pixel

def draw_reference_box(
    image,
    reference,
    color=(0, 255, 0),
    thickness=4):

    ymin, xmin, ymax, xmax = reference
    image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness)

    return image

if __name__ == "__main__":

    thermal_model   = os.path.join('thermal_model', 'thermal_face_automl_edge_fast.tflite')
    img_path        = os.path.join('images', 'img09.jpg')

    reference = {
        'box':  (20, 20, 40, 40),
        'temp': 30
    }

    gain = 30 / 255

    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=thermal_model)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    resize_shape = (input_shape[1], input_shape[2])

    for detail in input_details:
        print(detail)
    for detail in output_details:
        print(detail)

    # Load thermal image
    image = cv2.imread(img_path)

    # Resize image to meet size requirements of NN
    image_resized = cv2.resize(image, resize_shape)

    # Resize and normalize image
    #input_data = np.reshape(image_resized, input_shape).astype(float) / 255

    # Resize image
    input_data = np.reshape(image_resized, input_shape)

    print(input_data.shape)

    # Set input tensor to NN
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Invoke NN to detect faces
    interpreter.invoke()

    # Get output tensors - only needed
    bounding_boxes      = np.squeeze(interpreter.get_tensor(output_details[0]['index']))
    class_confidences   = np.squeeze(interpreter.get_tensor(output_details[2]['index']))

    confidence_threshold = 0.3

    boxes = bounding_box_thresholded(
        bounding_boxes,
        class_confidences,
        max_boxes=20, confidence_threshold=confidence_threshold)

    boxes = denormalize_boxes(
        image.shape,
        boxes)

    print(boxes)

    for box in boxes:
        ymin, xmin, ymax, xmax = box
        image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=(255, 0, 0), thickness=4)

    image = draw_reference_box(image, reference['box'], color=(0, 255, 0), thickness=4)

    reference_pixel = reference_box(image, reference['box'])

    print(reference_pixel)

    temperatures = detect_temperatures(image, boxes,
        reference['temp'],
        reference_pixel,
        gain,
        min_temp=34,
        max_temp=44)

    for i in range(len(temperatures)):
        ymin, xmin, ymax, xmax = boxes[i]
        cv2.putText(image, str(round(temperatures[i], 1)), (xmin,ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    print(temperatures)

    cv2.imshow("image_with_boxes", image)
    cv2.waitKey(0)

    print("Done!")