
import torch
import cv2
import matplotlib.pyplot as plt

midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small', trust_repo = True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )
midas.to(device)
midas.eval()

transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = transforms.small_transform


cap = cv2.VideoCapture(0) # stellt verbindung zur erst besten Kamera her
while cap.isOpened(): # Schleife die so lange läuft wie die Kamera geöffnet ist
    ret, frame = cap.read()
    
    # Transform input for midas
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgbatch = transform(img).to(device)
    
    # Make a prediction
    with torch.inference_mode():
        prediction = midas(imgbatch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size = img.shape[:2],
            mode = 'bicubic',
            align_corners = False
        ).squeeze()
        
        output = prediction.to(device).numpy()
    
    plt.imshow(output, cmap='viridis')    
    cv2.imshow('CV2Frame', frame)
    
    
    plt.pause(0.00001)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release() # gibt Kamera für andere Anwendungen wieder frei
        cv2.destroyAllWindows()

plt.show

