import dlib
import cv2
import openface

predictor_path = "F:/PyCharm/Facial Recognition/shape_predictor_68_face_landmarks.dat"

# Import the image file path
file_name = "F:/PyCharm/Facial Recognition/will-ferrell.jpg"

# Create a HOG face detector using the dlib class
face_detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor(predictor_path)
face_aligner = openface.AlignDlib(predictor_path)

# Load the image
image = cv2.imread(file_name)

# Run the HOG face detector on the image
detected_faces = face_detector(image ,1)

print("Found {} faces in the image file {}".format(len(detected_faces), file_name))

for i, face_rect in enumerate(detected_faces):
    # Detected faces are returned as an object with the coordinates
    # of the top, left, right and bottom edges
    print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.top(),
                                                                             face_rect.right(), face_rect.bottom()))

    # Get the the face's pose
    pose_landmarks = face_pose_predictor(image, face_rect)

    # Use openface to calculate and perform the face alignment
    alignedFace = face_aligner.align(534, image, face_rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

    # Save the aligned image to a file
    cv2.imwrite("aligned_face_{}.jpg".format(i), alignedFace)


