# Import modules
import dlib
from skimage import io

predictor_path = "F:/PyCharm/Facial Recognition/shape_predictor_68_face_landmarks.dat"

# Import the image file path
file_name = "F:/PyCharm/Facial Recognition/will-ferrell.jpg"

# Create a HOG face detector using the dlib class
face_detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor(predictor_path)

win = dlib.image_window()

# Load the image
image = io.imread(file_name)

# Run the HOG face detector on the image
detected_faces = face_detector(image, 1)

print("Found {} faces in the image file {}".format(len(detected_faces), file_name))

# Show the desktop window the image
win.set_image(image)

# Loop through each face we found in the image
for i, face_rect in enumerate(detected_faces):

    # Detected faces are returned as an object with the coordinates.
    # of the top, left, right, and bottom edges.
    print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.top(),
                                                                              face_rect.right(), face_rect.bottom()))

    # Draw a box around each face we found
    win.add_overlay(face_rect)

    # Get the faces' pose
    pose_landmarks = face_pose_predictor(image, face_rect)

    # Draw the face landmarks on the screen.
    win.add_overlay(pose_landmarks)

dlib.hit_enter_to_continue()