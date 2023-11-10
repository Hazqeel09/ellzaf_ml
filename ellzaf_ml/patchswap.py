import os
import cv2
import mediapipe as mp
import numpy as np

class PatchSwap:
    def __init__(self, static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            min_detection_confidence=min_detection_confidence
        )
        self.feature_indices = {
            'right_eye': [463, 414, 286, 258, 257, 259, 260, 467, 446, 255, 339, 254, 253, 252, 256, 341],
            'left_eye': [243, 190, 56, 28, 27, 29, 30, 247, 226, 25, 110, 24, 23, 22, 26, 112],
            'nose': [8, 417, 465, 412, 399, 456, 420, 429, 279, 358, 327, 326, 2, 97, 98, 129, 49, 209, 198, 236, 174, 188, 245, 193],
            'full_mouth': [164, 393, 391, 322, 410, 287, 273, 335, 406, 313, 18, 83, 182, 106, 43, 57, 186, 92, 165, 167],
            'lips': [0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61, 185, 40, 39, 37]
        }
        
     # Method to release the resources properly
    def close(self):
        self.face_mesh.close()

    def is_feature_visible(self, image, landmarks, feature_indices):
        height, width, _ = image.shape
        for idx in feature_indices:
            x, y = landmarks[idx]
            if x < 0 or y < 0 or x >= width or y >= height:
                return False
        return True

    def extract_features(self, image, facial_landmarks, feature_indices_dict):
        features = {}
        for feature_name, indices in feature_indices_dict.items():
            mask = np.zeros_like(image)
            points = np.array([facial_landmarks[point] for point in indices], dtype=np.int32)
            cv2.fillConvexPoly(mask, points, (255, 255, 255))
            features[feature_name] = (cv2.bitwise_and(image, mask), mask)
        return features

    def clamp(self, value, min_value, max_value):
        return max(min_value, min(value, max_value))

    def get_valid_roi(self, image, roi):
        x, y, w, h = roi
        height, width = image.shape[:2]

        # Clamp the coordinates to the image dimensions
        x = self.clamp(x, 0, width - 1)
        y = self.clamp(y, 0, height - 1)
        w = self.clamp(w, 1, width - x)
        h = self.clamp(h, 1, height - y)

        return x, y, w, h

    def get_valid_center_point(self, center_points, image_shape):
        # Ensure center point is within image boundaries
        center_x, center_y = np.mean(center_points, axis=0).astype(int)
        center_x = self.clamp(center_x, 0, image_shape[1] - 1)
        center_y = self.clamp(center_y, 0, image_shape[0] - 1)
        return center_x, center_y

    def align_and_blend_features(self, image_src, image_dst, feature_src, mask_src, landmarks_src, landmarks_dst, feature_indices):
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
        center_points = np.float32([landmarks_dst[idx] for idx in feature_indices if 0 <= landmarks_dst[idx][0] < image_dst.shape[1] and 0 <= landmarks_dst[idx][1] < image_dst.shape[0]])
        if center_points.size == 0:
            return image_dst, 0
        center_x, center_y = self.get_valid_center_point(center_points, image_dst.shape)
        center_point = (center_x, center_y)

        # Validate and clamp the ROI if needed
        bounding_rect = cv2.boundingRect(center_points)
        x, y, w, h = self.get_valid_roi(image_dst, bounding_rect)
        
        if w > 0 and h > 0:
            blended_image = cv2.seamlessClone(transformed_feature, image_dst, transformed_mask, center_point, cv2.NORMAL_CLONE)
            return blended_image, 1
        else:
            # If ROI is not valid, return the original destination image
            return image_dst, 0

    def landmarks_to_points(self, face_landmarks, image):
        # Convert face landmarks to points on the image
        return [(int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])) for landmark in face_landmarks.landmark]

    def read_and_process_images(self, path_img_a, path_img_b):
        # Read images and process them to find facial landmarks
        image_a, image_b = cv2.imread(path_img_a), cv2.imread(path_img_b)
        image_a_rgb, image_b_rgb = cv2.cvtColor(image_a, cv2.COLOR_BGR2RGB), cv2.cvtColor(image_b, cv2.COLOR_BGR2RGB)
        results_a, results_b = self.face_mesh.process(image_a_rgb), self.face_mesh.process(image_b_rgb)
        
        if not results_a.multi_face_landmarks or not results_b.multi_face_landmarks:
            raise ValueError("Could not detect faces in one or both images.")
        
        landmarks_a = self.landmarks_to_points(results_a.multi_face_landmarks[0], image_a)
        landmarks_b = self.landmarks_to_points(results_b.multi_face_landmarks[0], image_b)
        return image_a, image_b, landmarks_a, landmarks_b

    def swap_features(self, path_img_a, path_img_b, features_to_swap=None):
        if features_to_swap is None:
            features_to_swap = ["right_eye", "left_eye", "nose", "lips"]
        
        if not isinstance(features_to_swap, list):
            raise TypeError("features_to_swap must be a list of feature names.")
            
        try:
            image_a, image_b, landmarks_a, landmarks_b = self.read_and_process_images(path_img_a, path_img_b)
            features_a = self.extract_features(image_a, landmarks_a, self.feature_indices)
            features_b = self.extract_features(image_b, landmarks_b, self.feature_indices)
            
            # Process each feature separately
            for feature in features_to_swap:
                if feature not in self.feature_indices:
                    raise ValueError(f"Feature '{feature}' not recognized. Available features are: {list(self.feature_indices.keys())}")
                # Check if the feature is visible in both images before swapping
                if self.is_feature_visible(image_a, landmarks_a, self.feature_indices[feature]) and \
                self.is_feature_visible(image_b, landmarks_b, self.feature_indices[feature]):
                    # Swap feature from image A to image B
                    swap_result_b, success_b = self.align_and_blend_features(
                        image_a, image_b, features_a[feature][0], features_a[feature][1], landmarks_a, landmarks_b, self.feature_indices[feature]
                    )
                    if success_b:
                        image_b = swap_result_b
                    else:
                        image_b = None
                    # Swap feature from image B to image A
                    swap_result_a, success_a = self.align_and_blend_features(
                        image_b, image_a, features_b[feature][0], features_b[feature][1], landmarks_b, landmarks_a, self.feature_indices[feature]
                    )
                    if success_a:
                        image_a = swap_result_a
                    else:
                        image_a = None
            
            # Return both images and the success flags for each
            return image_a, image_b
        except ValueError as e:
            print(e)
            # In case of an exception, return the original images and failure flags
            return None, None


    def show_image(self, imageA, titleA, imageB, titleB):
        cv2.imshow(titleA, imageA)
        cv2.imshow(titleB, imageB)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def swap_features_in_directory(self, input_dir, output_dir, shuffle=True, shuffle_seed=39, additional_name="zswapped_", diff_counter=True, features_to_swap=None):
        if features_to_swap is None:
            features_to_swap = ["right_eye", "left_eye", "nose", "lips"]
            
        # Ensure the output directory exists
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"The input directory '{input_dir}' does not exist.")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Collect all image filenames
        image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if input_dir == output_dir:
            image_files = [f for f in image_files if not f.startswith(additional_name)]
            
        if not image_files:
            raise FileNotFoundError("No image files found in the input directory.")

        if shuffle:
            # Optionally shuffle the list if you don't want alphabetical order
            np.random.seed(shuffle_seed)
            np.random.shuffle(image_files)

        # Initialize counters for the images successfully generated
        total_new_images = 0

        # Perform feature swapping in a chained manner
        for i, file_name in enumerate(image_files):
            file_a = os.path.join(input_dir, file_name)
            file_b = os.path.join(input_dir, image_files[(i + 1) % len(image_files)])  # wrapping
            try:
                image_a, image_b = self.swap_features(file_a, file_b, features_to_swap)
                if image_a is not None:
                    cv2.imwrite(os.path.join(output_dir, f"{additional_name}{i if diff_counter else ''}_a_{file_name}"), image_a)
                    total_new_images += 1
                if image_b is not None:
                    cv2.imwrite(os.path.join(output_dir, f"{additional_name}{i if diff_counter else ''}_b_{file_name}"), image_b)
                    total_new_images += 1
            except Exception as e:
                print(f"Failed to swap features for images {file_a} and {file_b}: {e}")

        print(f"Chained feature swapping completed. {total_new_images} augmented images saved to {output_dir}.")
