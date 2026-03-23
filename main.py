import time
import math
import numpy as np
import depthai as dai
import cv2
from ultralytics import YOLO

from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from robotiqGripper import RobotiqGripper  

from calibracion import intrinsicosCamara as intr

R_cam2base, t_cam2base = np.load("calibración/matriz.npy", allow_pickle=True)

HOST = "192.168.56.101"

rtde_c = RTDEControlInterface(HOST)   
rtde_r = RTDEReceiveInterface(HOST)   

print("Robot conectado (ur_rtde)")

gripper = RobotiqGripper()
gripper.connect(HOST, 63352)          
gripper.activate()
gripper.move_and_wait_for_pos(0, 255, 255)   

elevacionEstandar         = -0.3485          
elevacionModuloPiezas     = -0.38451         
ejeModuloPiezas           = -0.365           
anguloGripperModuloPiezas = (2.226, 2.183, 0.0)

puntoSeguro = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   

robotOcupado = False
modeloUsado  = "models/best.pt"            
model        = YOLO(modeloUsado)                

def moveL(pose, speed=0.1, accel=0.5):
    """
    Movimiento lineal cartesiano.
    pose: [x, y, z, rx, ry, rz] en metros y radianes.
    ur_rtde usa moveL(pose, speed, accel, asynchronous=False).
    """
    rtde_c.moveL(pose, speed, accel)

def moveJ(joints_deg, speed=1.0, accel=1.4):
    """
    Movimiento por joints.
    joints_deg: lista de 6 ángulos en GRADOS (se convierten a radianes).
    """
    joints_rad = [math.radians(q) for q in joints_deg]
    rtde_c.moveJ(joints_rad, speed, accel)

def getTCPPose():
    """
    Retorna la pose TCP actual usando RTDEReceiveInterface.
    Equivale al struct.unpack manual del puerto 30003.
    """
    pose = rtde_r.getActualTCPPose()   
    x, y, z, rx, ry, rz = pose
    return {
        "x_m": x,    "y_m": y,    "z_m": z,
        "x_mm": x * 1000, "y_mm": y * 1000, "z_mm": z * 1000,
        "rx": rx,    "ry": ry,    "rz": rz
    }

def convertirCoordenada(centroX, centroY, anguloOBB):
    camera_matrix, dist_coeffs = intr.intrinsicosCamara()

    profundidad = None  

    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    x_cam = (centroX - cx) / fx * profundidad
    y_cam = (centroY - cy) / fy * profundidad
    z_cam = profundidad

    P_cam   = np.array([[x_cam], [y_cam], [z_cam]])
    P_robot = R_cam2base @ P_cam + t_cam2base

    rx, ry = anguloGripperModuloPiezas[0], anguloGripperModuloPiezas[1]
    rz     = anguloOBB

    return list(P_robot.flatten()) + [rx, ry, rz]

def movimientoCoordenada(coord, espacio):
    """
    Recoge un objeto en `coord` y lo deposita en `espacio`.
    Ambos son listas [x, y, z, rx, ry, rz] en metros/rad.
    """
    tiempoAcciones = 1
    velocidad      = 0.1
    aceleracion    = 0.1

    coord   = list(coord)
    espacio = list(espacio)

    def posicionSuperior(pose):
        """Eleva la coordenada Z al nivel estándar de seguridad."""
        p = list(pose)
        p[2] = elevacionEstandar
        return p

    def aplicarAnguloGripper(pose, angulo):
        """Sobreescribe el componente rz (índice 5) de la pose."""
        p = list(pose)
        p[5] = angulo
        return p

    puntoSuperiorCubo    = posicionSuperior(coord)
    puntoSuperiorEspacio = posicionSuperior(espacio)

    coordConAngulo      = aplicarAnguloGripper(coord,             coord[5])
    puntoSuperiorAngulo = aplicarAnguloGripper(puntoSuperiorCubo, coord[5])
    
    moveL(puntoSeguro,        speed=velocidad, accel=aceleracion)
    time.sleep(tiempoAcciones)

    moveL(puntoSuperiorAngulo, speed=velocidad, accel=aceleracion)
    time.sleep(tiempoAcciones)
    
    moveL(coordConAngulo,      speed=velocidad, accel=aceleracion)
    time.sleep(tiempoAcciones)

    gripper.move_and_wait_for_pos(255, 255, 255)
    time.sleep(tiempoAcciones)

    moveL(puntoSuperiorAngulo, speed=velocidad, accel=aceleracion)
    time.sleep(tiempoAcciones)
    
    puntoSuperiorModulo = aplicarAnguloGripper(puntoSuperiorCubo, anguloGripperModuloPiezas[2])
    moveL(puntoSuperiorModulo, speed=velocidad, accel=aceleracion)
    time.sleep(tiempoAcciones)

    moveL(puntoSuperiorEspacio, speed=velocidad, accel=aceleracion)
    time.sleep(tiempoAcciones)
      
    moveL(espacio,              speed=velocidad, accel=aceleracion)
    time.sleep(tiempoAcciones)
  
    gripper.move_and_wait_for_pos(0, 255, 255)
    time.sleep(tiempoAcciones)
    
    moveL(puntoSuperiorEspacio, speed=velocidad, accel=aceleracion)
    time.sleep(tiempoAcciones)
    
    moveL(puntoSeguro,          speed=velocidad, accel=aceleracion)
    time.sleep(tiempoAcciones)

def posModulo(x_mm):
    """
    Convierte coordenada X en mm al formato de pose en metros
    con los parámetros fijos del módulo.
    """
    x = x_mm / 1000.0
    y = ejeModuloPiezas / 1000.0    
    z = elevacionModuloPiezas       
    return [x, y, z, *anguloGripperModuloPiezas]

coordenadasModulo = {
    "cuadrado":  [posModulo(-731.4), posModulo(-787.8), 0],
    "triangulo": [posModulo(0),      posModulo(0),      0],
    "circulo":   [posModulo(-622.2), posModulo(-677.5), 0],
    "estrella":  [posModulo(-862),   posModulo(0),      0],
}

def frameFinal(frame, resultados_obb):
    coordenadas, clases = displayFrame("Objetos detectados", frame, resultados_obb, rtn=True)

    print("Se han detectado los siguientes objetos:")
    for i, coordenada in enumerate(coordenadas):
        print(f"  {clases[i]} con coordenadas {coordenada}")

    input("Presiona Enter para continuar con el ordenamiento de las cajas...")

    for i, coordenada in enumerate(coordenadas):
        clase = clases[i]
        j     = coordenadasModulo[clase][2]

        if j > 1:
            print(f"[ERROR] Todos los espacios para '{clase}' han sido cubiertos.")
            continue

        coordenadaColocamiento = coordenadasModulo[clase][j]
        print(f"Moviendo caja '{clase}' → {coordenadaColocamiento}")

        coordenadasRobot = convertirCoordenada(*coordenada)
        movimientoCoordenada(coordenadasRobot, coordenadaColocamiento)

        coordenadasModulo[clase][2] += 1

def displayFrame(name, frame, resultados_obb, rtn=False):
    """
    Dibuja los OBB detectados por Ultralytics sobre el frame.
    resultados_obb: results[0].obb  (lista de OBB del primer resultado)
    """
    color  = (255, 0, 0)
    coords = []
    clases = []

    if resultados_obb is not None:
        for box in resultados_obb:
            
            xywhr = box.xywhr[0].cpu().numpy()
            centroX   = int(xywhr[0])
            centroY   = int(xywhr[1])
            anguloOBB = float(xywhr[4])

            
            esquinas = box.xyxyxyxy[0].cpu().numpy().reshape((-1, 1, 2)).astype(int)
            cv2.polylines(frame, [esquinas], isClosed=True, color=color, thickness=2)

            clase = model.names[int(box.cls[0])]
            cv2.putText(frame, clase, (centroX + 10, centroY + 20),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))

            coords.append((centroX, centroY, anguloOBB))
            clases.append(clase)

    cv2.imshow(name, frame)

    if rtn:
        return coords, clases

moveL(puntoSeguro, speed=0.5, accel=0.3)
time.sleep(3)

with dai.Pipeline() as pipeline:
    
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(640, 640)
    cam.setInterleaved(False)
    cam.setFps(30)

    xoutRgb  = pipeline.create(dai.node.XLinkOut)
    xoutRgb.setStreamName("rgb")
    cam.preview.link(xoutRgb.input)

    pipeline.start()
    qRgb = pipeline.getOutputQueue("rgb", maxSize=1, blocking=False)

    frame        = None
    startTime    = time.monotonic()
    counter      = 0
    color2       = (255, 0, 255)
    ultimo_obb   = None   

    while pipeline.isRunning():
        inRgb = qRgb.tryGet()

        if inRgb is not None:
            frame = inRgb.getCvFrame()
            counter += 1

            results      = model(frame, verbose=False)
            ultimo_obb   = results[0].obb   

            fps_str = "FPS: {:.2f}".format(counter / (time.monotonic() - startTime))
            cv2.putText(frame, fps_str, (2, frame.shape[0] - 4),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.4, color2)

            displayFrame("rgb", frame, ultimo_obb)
            print(fps_str)

            if ultimo_obb is not None and len(ultimo_obb) > 0 and not robotOcupado:
                robotOcupado = True
                print("Objeto detectado, esperando 5 segundos")
                time.sleep(5)
    
                frameFinal(frame, ultimo_obb)
                robotOcupado = False

        if cv2.waitKey(1) == ord("q"):
            pipeline.stop()
            break

gripper.disconnect()
rtde_c.disconnect()
rtde_r.disconnect()