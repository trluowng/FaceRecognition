
import cv2
import numpy as np
import pickle
from mtcnn import MTCNN
from keras_facenet import FaceNet
from tensorflow.keras.models import load_model
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

#tải mô hình MLP và bộ mã hóa nhãn
model = load_model('D:/NCKH/NCKH 25-26/Source/mlp_face_classifier.h5')
with open('D:/NCKH/NCKH 25-26/Source/label_encoder.pkl', 'rb') as f:
    out_encoder = pickle.load(f)

#tính tâm của mỗi lớp ở file faces_embedding_4classes.npz
centroids = {}
have_centroids = False
try:
    data = np.load('D:/NCKH/NCKH 25-26/Source/faces_embedding_4classes.npz')
    #kiểm tra key trong npz để lấy embedding và label
    if 'embeddings' in data and 'labels' in data:
        X, y = data['embeddings'], data['labels']
    else:
        #đặt tên arr_0, arr_1 nếu ảnh ko tên
        X, y = data['arr_0'], data['arr_1']
    data.close()
    #đảm bảo y là mảng numpy dạng số để tính trọng tâm
    y = np.array(y)
    for label in np.unique(y):
        #tính vector tâm cho lớp = trung bình của các embedding thuộc lớp đó
        class_embeddings = X[y == label]
        if len(class_embeddings) > 0:
            centroids[label] = class_embeddings.mean(axis=0)
    have_centroids = True
except Exception as e:
    print("Không tìm thấy hoặc không thể đọc file faces_embedding_4classes.npz")
    have_centroids = False

#khởi tạo MTCNN
detector = MTCNN()
embedder = FaceNet()

#mở cam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Không thể mở webcam.")
    exit()

print("Bắt đầu nhận diện. Nhấn 'q' để thoát.")
while True:
    #Đọcframe từcam
    ret, frame = cap.read()
    if not ret:
        break

    #chuyển sang RGB để MTCNN xử lý
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #phát hiện các khuôn mặt trong frame
    faces = detector.detect_faces(rgb_frame)

    #duyệt qua từng khuôn mặt
    for face in faces:
        #lấy tọa độ bounding box (x, y, width, height)
        x, y, w, h = face['box']
        # Đảm bảo tọa độ nằm trong khung hình
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = x1 + w, y1 + h
        x2 = min(x2, frame.shape[1])
        y2 = min(y2, frame.shape[0])

        #chuẩn hóa kích thước 160x160
        face_crop = frame[y1:y2, x1:x2]              #crop khuôn mặt (ở dạng BGR)
        face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)  #chuyển sang RGB cho FaceNet
        face_resized = cv2.resize(face_crop_rgb, (160, 160))        #resize về 160x160 (đầu vào FaceNet)

        # trích xuất embedding
        embeddings = embedder.embeddings([face_resized])
        if embeddings is None or len(embeddings) == 0:
            #bỏ qua nếu ko trích được embedding
            continue
        embedding_vector = embeddings[0]  # vector embedding của khuôn mặt

        
        #thêm một chiều batch cho vector embedding
        embedding_vector_expanded = np.expand_dims(embedding_vector, axis=0)
        #Sử dụng mô hình MLP để dự đoán lớp khuôn mặt
        yhat = model.predict(embedding_vector_expanded)  # dự đoán xác suất các lớp
        probabilities = yhat[0]
        class_index = np.argmax(probabilities)
        class_prob = probabilities[class_index]
        label_text = "Unknown"

        #kiểm tra ngưỡng xác suất cao nhất 0.75
        if class_prob >= 0.75:
            #lấy tên lớp dự đoán từ label encoder
            class_name = out_encoder.inverse_transform([class_index])[0]
            label_text = f"{class_name} {class_prob*100:.2f}%"

            #nếu dùng centroids, kiểm tra khoảng cách embedding đến tâm lớp
            if have_centroids and class_index in centroids:
                dist = np.linalg.norm(embedding_vector - centroids[class_index])
                if dist > 0.7:
                    #Nếu centroid xa hơn 0.7, coi như Unknown
                    label_text = "Unknown"

        #ngưỡng ảnh mờ
        face_crop = frame[y1:y2, x1:x2]  # ảnh BGR  
        gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        #tính độ sắc nét bằng Laplacian
        lap_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()

        #Vẽ bounding box và nhãn tên + xác suất
        if label_text == "Unknown":
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
        elif lap_var < 45:
            label_text = "Vui long giu nguyen vi tri"
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0, 140, 255),2)
        else:
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

        #vẽ nhãn
        text_y = y1 - 10 if y1 - 10 > 10 else y2 + 20
        cv2.putText(frame, label_text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

    #hiển thị
    cv2.imshow("Face Recognition", frame)
    #tạo nút thoát 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
