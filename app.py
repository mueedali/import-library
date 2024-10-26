import cv2
import face_recognition
import os
import csv
from datetime import datetime

# Function to find encodings of the images
def find_encodings(images):
    encodings_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings_list.append(face_recognition.face_encodings(img)[0])
    return encodings_list

# Function to mark attendance in a CSV file
def mark_attendance(name):
    current_date = datetime.now().strftime('%Y-%m-%d')
    with open('attendance.csv', 'r+') as file:
        data_list = file.readlines()
        attendance_list = [line.split(',')[0] for line in data_list if current_date in line]
        
        if name not in attendance_list:  # Ensure the name is marked once per day
            time_now = datetime.now().strftime('%H:%M:%S')
            file.write(f'{name},{current_date},{time_now}\n')

# Load images and names from a directory
def load_images_from_directory(directory):
    images = []
    class_names = []
    for file_name in os.listdir(directory):
        img = cv2.imread(f'{directory}/{file_name}')
        images.append(img)
        class_names.append(os.path.splitext(file_name)[0])
    return images, class_names

# Main function
def main():
    # Set the path to the images folder
    path = 'images'  # Update this with the correct path to your images folder
    images, class_names = load_images_from_directory(path)
    known_encodings = find_encodings(images)

    # Initialize webcam
    cap = cv2.VideoCapture(0)

    while True:  # Keep running the loop until 'q' is pressed
        success, img = cap.read()
        img_small = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        img_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

        faces_current_frame = face_recognition.face_locations(img_small)
        encodings_current_frame = face_recognition.face_encodings(img_small, faces_current_frame)

        for encode_face, face_loc in zip(encodings_current_frame, faces_current_frame):
            matches = face_recognition.compare_faces(known_encodings, encode_face)
            face_distances = face_recognition.face_distance(known_encodings, encode_face)

            # Get the best match index based on minimum distance
            match_index = min(range(len(face_distances)), key=face_distances.__getitem__)

            # Check if the face matches any known encodings
            if matches[match_index] and face_distances[match_index] < 0.6:  # Threshold to reduce false positives
                name = class_names[match_index].capitalize()
                mark_attendance(name)  # Mark attendance for known individuals
            else:
                name = "Unknown"  # Set label to "Unknown" for unrecognized individuals

            # Display the name on the webcam feed
            y1, x2, y2, x1 = face_loc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4  # Scale back to original size
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Webcam', img)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()  # Release the webcam
    cv2.destroyAllWindows()  # Close all OpenCV windows

# Create the attendance CSV file if not exists
def initialize_csv():
    if not os.path.exists('attendance.csv'):
        with open('attendance.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Name', 'Date', 'Time'])

if __name__ == "__main__":
    initialize_csv()  # Initialize CSV file
    main()  # Start the webcam and face recognition
