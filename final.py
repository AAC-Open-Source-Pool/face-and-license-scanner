import cv2
import os
import dlib
import numpy as np
import winsound


def capture_images_live(directory, prefix, num_images):
    cap = cv2.VideoCapture(0)
    if not os.path.exists(directory):
        os.makedirs(directory)

    for i in range(num_images):
        ret, img = cap.read()
        cv2.imshow('Capture', img)
        img_path = os.path.join(directory, f"{prefix}_{i}.jpg")
        cv2.imwrite(img_path, img)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()


def process_image(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    histogram = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    return histogram


def crop(image):
    img = cv2.imread(image)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_image, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        faces = img[y:y + h, x:x + w]
        cv2.imwrite('live.jpg', faces)


def preprocess_image(image_path):
    # Load the image and convert it to RGB format
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a face detector
    face_detector = dlib.get_frontal_face_detector()

    # Detect faces in the image
    face_locations = face_detector(image)

    if not face_locations:
        print("No face detected in the image.")
        return None

    # Assume there's only one face in the image
    face_location = face_locations[0]

    # Get the rectangle coordinates of the face
    x, y, w, h = face_location.left(), face_location.top(), face_location.width(), face_location.height()

    # Crop the face from the image
    face_chip = image[y:y+h, x:x+w]

    # Resize the face chip to 150x150
    face_chip = cv2.resize(face_chip, (150, 150))

    return face_chip


def face_recognition():
    # Load a known face image
    known_face_chip = preprocess_image('face1.jpg')

    if known_face_chip is None:
        return

    # Load the face recognition model
    facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

    # Compute the face descriptor for the known face
    known_face_descriptor = facerec.compute_face_descriptor(known_face_chip)

    # Capture a live image for face recognition
    capture_images_live("live", "live", 1)

    live_face_chip = preprocess_image('live/live_0.jpg')

    if live_face_chip is None:
        return

    # Compute face descriptors for detected faces
    live_face_descriptor = facerec.compute_face_descriptor(live_face_chip)

    # Calculate the Euclidean distance between known and live face descriptors
    similarity = np.linalg.norm(np.array(known_face_descriptor) - np.array(live_face_descriptor))
    print(similarity)
    if similarity < 0.6:  # Adjust the threshold as needed
        print("Face matched, go to the next step.")
    else:
        print("Face not matched.")
        winsound.Beep(1000, 1000)


def calculate_distance(hist1, hist2):
    eps = 1e-10
    return 0.5 * np.sum(((hist1 - hist2) ** 2) / (hist1 + hist2 + eps))


def main():
    # Capture License Images Live
    capture_images_live("license dataset/L", "L", 2)

    # Process License and Sample License Images
    license_histogram = process_image('license.jpg')
    sample_license_histogram = process_image('license.jpg')

    # Calculate Histogram Distances for License Images
    d1 = calculate_distance(license_histogram, sample_license_histogram)
    print(d1)
    if d1 < 2:  # Adjust the threshold as needed
        face_recognition()
    else:
        print("License not matched.")
        winsound.Beep(1000, 1000)


if __name__ == "__main__":
    main()
