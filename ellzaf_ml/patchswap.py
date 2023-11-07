import cv2
import mediapipe as mp
import numpy as np
import random

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

def extract_features(image, facial_landmarks, feature_indices_dict):
    features = {}
    for feature_name, indices in feature_indices_dict.items():
        mask = np.zeros_like(image)
        points = np.array([facial_landmarks[point] for point in indices], dtype=np.int32)
        cv2.fillConvexPoly(mask, points, (255, 255, 255))
        features[feature_name] = (cv2.bitwise_and(image, mask), mask)
    return features

def align_and_blend_features(image_src, image_dst, feature_src, mask_src, landmarks_src, landmarks_dst, feature_indices):
    # Validate that there are enough points for affine transformation
    if len(feature_indices) < 3:
        raise ValueError("Not enough landmark points for affine transformation.")

    # Ensure the correct points are used for the source and destination
    src_points = np.float32([landmarks_src[idx] for idx in feature_indices[:3]])
    dst_points = np.float32([landmarks_dst[idx] for idx in feature_indices[:3]])

    # Calculate the affine transform
    warp_mat = cv2.getAffineTransform(src_points, dst_points)

    # Apply the affine transformation to the source feature
    transformed_feature = cv2.warpAffine(feature_src, warp_mat, (image_dst.shape[1], image_dst.shape[0]))
    transformed_mask = cv2.warpAffine(mask_src, warp_mat, (image_dst.shape[1], image_dst.shape[0]))

    # Determine the correct center point for seamless cloning
    # This should be the centroid of the destination feature's landmarks
    center_point = tuple(np.mean(np.float32([landmarks_dst[idx] for idx in feature_indices]), axis=0).astype(int))

    # Perform seamless cloning using the transformed feature, mask, and center point
    blended_image = cv2.seamlessClone(transformed_feature, image_dst, transformed_mask, center_point, cv2.NORMAL_CLONE)
    
    return blended_image
    
def landmarks_to_points(face_landmarks, image):
    # Convert face landmarks to points on the image
    return [(int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])) for landmark in face_landmarks.landmark]

def read_and_process_images(path_img_a, path_img_b):
    # Read images and process them to find facial landmarks
    image_a, image_b = cv2.imread(path_img_a), cv2.imread(path_img_b)
    image_a_rgb, image_b_rgb = cv2.cvtColor(image_a, cv2.COLOR_BGR2RGB), cv2.cvtColor(image_b, cv2.COLOR_BGR2RGB)
    results_a, results_b = face_mesh.process(image_a_rgb), face_mesh.process(image_b_rgb)
    
    if not results_a.multi_face_landmarks or not results_b.multi_face_landmarks:
        raise ValueError("Could not detect faces in one or both images.")
    
    landmarks_a = landmarks_to_points(results_a.multi_face_landmarks[0], image_a)
    landmarks_b = landmarks_to_points(results_b.multi_face_landmarks[0], image_b)
    return image_a, image_b, landmarks_a, landmarks_b

feature_indices = {
    'right_eye': [463, 414, 286, 258, 257, 259, 260, 467, 446, 255, 339, 254, 253, 252, 256, 341],
    'left_eye': [243, 190, 56, 28, 27, 29, 30, 247, 226, 25, 110, 24, 23, 22, 26, 112],
    'nose': [8, 417, 465, 412, 399, 456, 420, 429, 279, 358, 327, 326, 2, 97, 98, 129, 49, 209, 198, 236, 174, 188, 245, 193],
    #'full_mouth': [164, 393, 391, 322, 410, 287, 273, 335, 406, 313, 18, 83, 182, 106, 43, 57, 186, 92, 165, 167],
    'lips': [0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61, 185, 40, 39, 37]
}

# Read and process the images
try:
    image_a, image_b, landmarks_a, landmarks_b = read_and_process_images('face_live/facespoof_train/ori/000446.jpg',
                                                                         'face_live/facespoof_train/ori/000459.jpg')

    # Extract features from both images
    features_a = extract_features(image_a, landmarks_a, feature_indices)
    features_b = extract_features(image_b, landmarks_b, feature_indices)

    for feature in feature_indices.keys():
        image_b = align_and_blend_features(image_a, image_b, features_a[feature][0], features_a[feature][1], landmarks_a, landmarks_b, feature_indices[feature])
        image_a = align_and_blend_features(image_b, image_a, features_b[feature][0], features_b[feature][1], landmarks_b, landmarks_a, feature_indices[feature])
    
    # Display the output
    cv2.imshow('Image A with features from B', image_a)
    cv2.imshow('Image B with features from A', image_b)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
except ValueError as e:
    print(e)