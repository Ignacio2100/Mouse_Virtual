import cv2
import mediapipe as mp
import pyautogui

# Inicializar Mediapipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1, 
    min_detection_confidence=0.5,  # Ajustar la confianza mínima de detección
    min_tracking_confidence=0.5     # Ajustar la confianza mínima de seguimiento
)

# Obtener el tamaño de la pantalla una vez
screen_width, screen_height = pyautogui.size()

# Inicializar la cámara con una resolución más baja
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Anchura
cap.set(4, 480)  # Altura

# Definir el tamaño deseado para la visualización
desired_width, desired_height = 800, 600

# Configurar el tamaño de la ventana de visualización
cv2.namedWindow('Hand Tracking', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Hand Tracking', desired_width, desired_height)

# Variables para guardar el estado de los dedos
index_finger_up = False
middle_finger_up = False
pinky_up = False
ring_finger_up = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("No se puede recibir el video")
        break

    # Convertir la imagen de BGR a RGB una sola vez fuera del bucle
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detectar manos en la imagen
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Dibujar landmarks de la mano
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Verificar el estado de los dedos una sola vez
            index_finger_tip_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * screen_height)
            middle_finger_tip_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * screen_height)
            pinky_tip_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * screen_height)
            ring_finger_tip_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * screen_height)

            index_finger_up = index_finger_tip_y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * screen_height
            middle_finger_up = middle_finger_tip_y < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y * screen_height
            pinky_up = pinky_tip_y < hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y * screen_height
            ring_finger_up = ring_finger_tip_y < hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y * screen_height

            # Mover el ratón si el dedo índice está levantado
            if index_finger_up:
                # Obtener las coordenadas del dedo índice
                index_finger_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * screen_width)
                pyautogui.moveTo(index_finger_x, index_finger_tip_y, duration=0.1)

            # Calcular la distancia entre los puntos correspondientes al dedo índice y el pulgar
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            distance_threshold = 0.05  # Umbral de distancia para el clic (ajustar según sea necesario)

            distance = ((index_finger_tip.x - thumb_tip.x)**2 + (index_finger_tip.y - thumb_tip.y)**2)**0.5

            # Hacer clic si la distancia es menor que el umbral
            if distance < distance_threshold:
                pyautogui.click()

            # Hacer scroll hacia arriba si el meñique está levantado
            if pinky_up:
                pyautogui.scroll(35)

            # Hacer scroll hacia abajo si el anular está levantado
            if ring_finger_up:
                pyautogui.scroll(-35)

    # Mostrar el frame
    cv2.imshow('Hand Tracking', frame)

    # Salir del bucle con 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()
