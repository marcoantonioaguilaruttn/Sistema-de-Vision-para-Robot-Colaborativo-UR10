#Asegurarse de haber activado el entorno virtual tonotos

#Mientras no se haya detectado ninguna instancia de la clase caja, se va a grabar en video.

#Una vez detectada una instancia, se toma un frame y se espera hasta que el robot haya terminado su movimiento
#antes de volver a capturar el video

#PONER COORDENADAS REEALES PARA LA COLOCACION DE OBJETOS
#ESTABLECER PUNTO SEGURO
#ESTABLECER CANTIDAD DE LEVANTAMIENTO 

import robotiqGripper as rg
import time
import numpy as np
import depthai as dai
import cv2

from calibración import intrinsicosCamara as intr

from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive


R_cam2base, t_cam2base = np.load("calibración/matriz.npy", allow_pickle=True)

def convertirCoordenada(coordenadasB):
    camera_matrix, dist_coeffs = intr.intrinsicosCamara()
    
    xmin, xmax = coordenadasB[0], coordenadasB[2]
    ymin, ymax = coordenadasB[1], coordenadasB[3]
    
    profundidad = None  #MEDIR PROFUNDIAD DE LA CAMARA A LA MESA
    centroX, centroY = (xmin + xmax) / 2, (ymin + ymax) / 2
    
    
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    
    
    x_cam = (centroX - cx) / fx * profundidad
    y_cam = (centroY - cy) / fy * profundidad
    z_cam = profundidad
    
    P_cam = np.array([[x_cam],
                      [y_cam],
                      [z_cam]])
    
    
    P_robot = R_cam2base @ P_cam + t_cam2base
    
    return P_robot

def movimientoCoordenada(coord, espacio):
    cantidadElevacionPuntoSeguro, tiempoAcciones = 0.5, 2
    
    coord = list(coord)
    espacio = list(espacio)
    
    rtdec.moveL(coord, 0.5, 0.3)
    time.sleep(tiempoAcciones)
    
    gripper.close()
    
    coord[2] += cantidadElevacionPuntoSeguro  
    rtdec.moveL(coord, 0.5, 0.3)
    time.sleep(tiempoAcciones)
    
    arribaEspacio = list(espacio)
    arribaEspacio[2] += cantidadElevacionPuntoSeguro  
    
    rtdec.moveL(arribaEspacio, 0.5, 0.3)
    time.sleep(tiempoAcciones)
    
    rtdec.moveL(espacio, 0.5, 0.3)
    gripper.open()
    time.sleep(tiempoAcciones)
    
    rtdec.moveL(arribaEspacio, 0.5, 0.3)
    rtdec.moveL(puntoSeguro, 0.5, 0.3)

def frameFinal(frame):
    coordenadas, clases = displayFrame("Objetos detectados", frame, True)
    
    print("Se han detectado los siguientes objetos: ")
    for i, coordenada in enumerate(coordenadas):
        print(f"{clases[i]} con coordenadas {coordenada}")
    
    input("Presiona cualquier tecla para continuar con el ordenamiento de las cajas")
    
    for i, coordenada in enumerate(coordenadas):
        clase = clases[i]
        j = coordenadasModulo[clase][2]
        
        if j > 1:
            print(f"Error de deteccion: Todos los espacios para la figura {clase} han sido cubiertos")
            continue
        
        coordenadaColocamiento = coordenadasModulo[clase][j]
        
        print(f"Moviendo caja con figura {clase} a la coordenada {coordenadaColocamiento}")
        
        coordenadasRobot = convertirCoordenada(coordenada)
        movimientoCoordenada(coordenadasRobot, coordenadaColocamiento)
        
        coordenadasModulo[clase][2] += 1

coordenadasModulo = {
    "cuadrado":  [(0, 0, 0), (0, 0, 0), 0],
    "triangulo": [(0, 0, 0), (0, 0, 0), 0],
    "circulo":   [(0, 0, 0), (0, 0, 0), 0],
    "estrella":  [(0, 0, 0), (0, 0, 0), 0]
}

ur_ip = None
rtde_frequency = 500.0

rtdec = RTDEControl(ur_ip, rtde_frequency, RTDEControl.FLAG_USE_EXT_UR_CAP)
gripper = rg.RobotiqGripper(rtdec)

gripper.activate()
gripper.open()

puntoSeguro = None
rtdec.moveL(puntoSeguro, 0.5, 0.3)

modeloUsado = None
rutaModelo = f"/models/{modeloUsado}"

with dai.Pipeline() as pipeline:
    cameraNode = pipeline.create(dai.node.Camera).build()
    detectionNetwork = pipeline.create(dai.node.DetectionNetwork).build(cameraNode, dai.NNModelDescription(rutaModelo))

    
    mapaClases = detectionNetwork.getClasses()

    
    qRgb = detectionNetwork.passthrough.createOutputQueue()
    qDet = detectionNetwork.out.createOutputQueue()

    pipeline.start()

    frame = None
    detections = []
    startTime = time.monotonic()
    counter = 0
    color2 = (255, 0, 255)

    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[[0, 2]] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)
    
    def displayFrame(name, frame, rtn=False):
        color = (255, 0, 0)
        coords, clases = [], []
        
        for detection in detections:
            bbox = frameNorm(
                frame,
                (detection.xmin, detection.ymin, detection.xmax, detection.ymax),
            )
            
            cv2.putText(
                frame,
                mapaClases[detection.label],
                (bbox[0] + 10, bbox[1] + 20),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.5,
                255,
            )
    
            coords.append(bbox)
            clases.append(mapaClases[detection.label])
            
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

        cv2.imshow(name, frame)
        
        if rtn:
            return coords, clases

    while pipeline.isRunning():
        inRgb = qRgb.get()
        inDet = qDet.get()

        if inRgb is not None:
            frame = inRgb.getCvFrame()
            cv2.putText(
                frame,
                "NN fps: {:.2f}".format(counter / (time.monotonic() - startTime)),
                (2, frame.shape[0] - 4),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.4,
                color2,
            )
        
        if inDet is not None:
            detections = inDet.detections
            counter += 1
        
        if frame is not None:
            displayFrame("rgb", frame)
            print("FPS: {:.2f}".format(counter / (time.monotonic() - startTime)))
            
            if inDet is not None and len(detections) > 0:
                print("Se ha detectado un objeto, iniciando temporizador de 5 segundos")
                time.sleep(5)
                frameFinal(frame)
        
        if cv2.waitKey(1) == ord("q"):
            pipeline.stop()
            break