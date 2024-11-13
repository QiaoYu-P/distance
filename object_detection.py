import cv2
import numpy as np
def estimate_distance(known_width, focal_length, width_in_image):
    return (known_width * focal_length) / width_in_image
def load_yolo():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layer_names = net.getLayerNames()
    unconnected_out_layers = net.getUnconnectedOutLayers()

    if isinstance(unconnected_out_layers, np.ndarray) and unconnected_out_layers.ndim == 1:
        output_layers = [layer_names[i - 1] for i in unconnected_out_layers]
    else:
        output_layers = [layer_names[unconnected_out_layers - 1]]

    return net, output_layers
def detect_objects(img, net, output_layers):
    height, width = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    return outs, width, height
def get_box_dimensions(outs, width, height):
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    return boxes, confidences, class_ids
def draw_labels(boxes, confidences, class_ids, classes, img, known_width, focal_length):
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (0, 255, 0)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

            # 计算距离
            distance = estimate_distance(known_width, focal_length, w)
            cv2.putText(img, f"{label}: {distance:.2f} m", (x, y - 5), font, 1, color, 1)
    return img

def main():
    known_width = 0.5  # 已知物体的宽度（单位：米）
    focal_length = 700  # 相机的焦距（单位：像素）

    net, output_layers = load_yolo()
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    cap = cv2.VideoCapture("222.mp4")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        outs, width, height = detect_objects(frame, net, output_layers)
        boxes, confidences, class_ids = get_box_dimensions(outs, width, height)
        frame = draw_labels(boxes, confidences, class_ids, classes, frame, known_width, focal_length)

        cv2.imshow("Image", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
