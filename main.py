import time
import math
import numpy as np
import depthai as dai
import cv2
import os
from ultralytics import YOLO

from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from robotiqGripper import RobotiqGripper

METODO_ACTIVO = "Park"  

#CORRECCIONES APLICADAS A LA TRANSFORMACION
CORR_A  =  0.939867
CORR_B  =  0.060054
CORR_TX = -0.012375
CORR_TY =  0.060002

#FUNCION PARA APLICAR LA CORRECCION
def aplicarCorreccionAfin(x, y):
    x_c = CORR_A * x - CORR_B * y + CORR_TX
    y_c = CORR_B * x + CORR_A * y + CORR_TY
    return x_c, y_c

DIR_RESULTADOS_CALIB = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "calibracion", "ResultadosCalib"
)
DIR_MATRIZ_LEGACY = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "calibracion", "matriznpy.npz"
)

#METODOS DISPONIBLES PARA LAS TRANSFORMACIONES ENTRE CÁMARA Y COORDENDAS BASE DEL ROBOT
NOMBRES_METODOS = ["Tsai", "Park", "Horaud", "Andreff", "Daniilidis"]


def cargarTodosLosMetodos():
    metodos = {}
    
    for nombre in NOMBRES_METODOS:
        ruta = os.path.join(DIR_RESULTADOS_CALIB, f"T_cam2gripper_{nombre}.npz")
        if os.path.exists(ruta):
            data = np.load(ruta)
            metodos[nombre] = (data["R"], data["T"].reshape(3, 1))
            print(f"  [OK] {nombre:12s} → {ruta}")
            
        else:
            print(f"  [--] {nombre:12s} no encontrado en {ruta}")

    if os.path.exists(DIR_MATRIZ_LEGACY):
        data = np.load(DIR_MATRIZ_LEGACY)
        metodos["Legacy"] = (data["R"], data["T"].reshape(3, 1))
        print(f"  [OK] {'Legacy':12s} → {DIR_MATRIZ_LEGACY}")

    if not metodos:
        raise FileNotFoundError(
            "No se encontró ningún archivo de calibración.\n"
            "Ejecuta primero calibracion_ojo_mano.py para generarlos."
        )
    return metodos

print("\nCargando matrices de calibración:")
CALIBRACIONES = cargarTodosLosMetodos()

print(f"Métodos disponibles: {list(CALIBRACIONES.keys())}\n")

if METODO_ACTIVO and METODO_ACTIVO in CALIBRACIONES:
    R_cam2base, t_cam2base = CALIBRACIONES[METODO_ACTIVO]
    print(f"Método activo para el robot: {METODO_ACTIVO}")
elif CALIBRACIONES:
    primer_metodo          = list(CALIBRACIONES.keys())[0]
    R_cam2base, t_cam2base = CALIBRACIONES[primer_metodo]
    print(f"METODO_ACTIVO no definido — usando '{primer_metodo}' por defecto.")


HOST   = "192.168.56.101" #CAMBIAR POR IP DEL ROBOT UR. ESTA SE PUEDE ENCONTRAR EN EL TEACH PENDANT
rtde_c = RTDEControlInterface(HOST) #SE ESTABLECE LA CONEXIÓN DE CONTROL CON EL ROBOT
rtde_r = RTDEReceiveInterface(HOST) #SE ESTABLECE LA CONEXIÓN DE RECEPCIÓN DE DATOS DEL ROBOT.

print("Robot conectado")

#SE CREA UNA INSTANCIA DE LA CLASE GRIPPER
gripper = RobotiqGripper()
gripper.connect(HOST, 63352)

gripper.activate()
gripper.move_and_wait_for_pos(0, 255, 255)

#VALORES ESTÁNDAR PARA LA COLOCACIÓN DE PIEZAS
elevacionEstandar         = 0.046
elevacionModuloPiezas     = 0.046
ejeModuloPiezas           = -0.365 #COORDENADAS DEL EJE Y DEL MODULO DE COLOCACIÓN
anguloGripperModuloPiezas = (2.226, 2.183, 0.0) #ANGULO DEL GRIPPER AL MOMENTO DE DEPOSITAR LOS CUBOS

RX_AGARRE = 2.193
RY_AGARRE = 2.223
RZ_AGARRE = 0.013

puntoSeguro     = [-0.615, -0.168, 0.069, 1.992, 2.396, 0] #COORDENDAS TCP A DONDE REGRESARA EL RBOOT DESPUES DE COLOCAR CADA PIEZA
robotOcupado    = False

modeloUsado     = "models/best.pt" #MODELO YOLO USADO PARA LA DETECCION DE PIEZAS
model           = YOLO(modeloUsado)

RESOLUCION_YOLO = (384, 384) #RESOLUCION, NO CAMBIAR BAJO NINGUNA CIRCUNSTANCIA

#FUNCION PARA CARGAR LOS PARAMETROS INTRINSECOS DE LA CÁMARA OAK-1 LITE
def intrinsicosCamara(resolucion=RESOLUCION_YOLO):
    with dai.Device() as device:
        calibData     = device.readCalibration()
        intrinsics    = calibData.getCameraIntrinsics(
            dai.CameraBoardSocket.CAM_A,
            resolucion[0],
            resolucion[1]
        )
        
        camera_matrix = np.array(intrinsics)
        
        dist_coeffs   = np.array(
            calibData.getDistortionCoefficients(dai.CameraBoardSocket.CAM_A)
        )
        
    print(f"Intrínsecos obtenidos para resolución {resolucion[0]}x{resolucion[1]}")
    
    return camera_matrix, dist_coeffs

#FUNCION PARA MOVER DE FORMA LINEAL EL ROBOT, TOMA COMO ARGUMENTOS COORDENADAS TCP
def moveL(pose, speed=0.1, accel=0.5):
    """Movimiento lineal cartesiano. pose: [x,y,z,rx,ry,rz] en metros/rad."""
    rtde_c.moveL(pose, speed, accel)

#FUNCION PARA MOVER LAS ARTICULACIONES DEL ROBOT, SE INTEGRA UNA CONVERSION DE GRADOS A RADIANES
def moveJ(joints, speed=1.0, accel=1.4):

    if max(abs(j) for j in joints) > 2 * math.pi:
        joints = [math.radians(q) for q in joints]
        
    rtde_c.moveJ(joints, speed, accel)

#FUNCION PARA OBTENER LA POSICIÓN TCP ACTUAL
def getTCPPose():
    pose = rtde_r.getActualTCPPose()
    x, y, z, rx, ry, rz = pose
    return {
        "x_m": x,       "y_m": y,       "z_m": z,
        "x_mm": x*1000, "y_mm": y*1000, "z_mm": z*1000,
        "rx": rx,       "ry": ry,       "rz": rz
    }

def waitActivo(segundos):
    """Reemplaza time.sleep() manteniendo ventanas de OpenCV responsivas."""
    fin = time.monotonic() + segundos
    while time.monotonic() < fin:
        cv2.waitKey(50)
        
#CONVIERTE LAS COORDENADAS EN PIXELES DEL MODELO YOLO A COORDENADAS TCP DEL ROBOT
def convertirCoordenada(centroX, centroY, anguloOBB, R=None, t=None):
    if R is None:
        R = R_cam2base
    if t is None:
        t = t_cam2base

    r20, r21, r22 = R[2, :]
    t2            = float(t[2, 0])

    x_cam_norm = (centroX - CX) / FX
    y_cam_norm = (centroY - CY) / FY

    denominador = r20 * x_cam_norm + r21 * y_cam_norm + r22
    
    #EVITA DIVISION POR CERO ASIGNANDO PROFUNDIDAD POR DEFECTO
    if abs(denominador) < 1e-6:
        z_cam = 0.37
    else:
        z_cam = (elevacionEstandar - t2) / denominador

    x_cam = x_cam_norm * z_cam
    y_cam = y_cam_norm * z_cam

    #TRANSFORMA PUNTO DE ESPACIO CAMARA A ESPACIO ROBOT
    P_cam   = np.array([[x_cam], [y_cam], [z_cam]])
    P_robot = R @ P_cam + t

    coords = list(P_robot.flatten())

    #APLICA CORRECCION AFIN SOLO CON LA CALIBRACION PRINCIPAL
    if R is R_cam2base:
        coords[0], coords[1] = aplicarCorreccionAfin(coords[0], coords[1])

    #RETORNA COORDENADAS XYZ + ORIENTACION DE AGARRE + ANGULO OBB
    return coords + [RX_AGARRE, RY_AGARRE, RZ_AGARRE, anguloOBB]
    
#EJECUTA LA SECUENCIA COMPLETA DE PICK AND PLACE PARA UN OBJETO
def movimientoCoordenada(coord, espacio):
    velocidad    = 0.2
    aceleracion  = 0.2
    tEspera      = 1

    elevacionAproximacion = 0.041   
    elevacionAgarre       = 0.01   

    coord   = list(coord)
    espacio = list(espacio)

    anguloOBB = coord[6]

    #GENERA POSE SEGURA SOBRE UN PUNTO A ALTURA DE APROXIMACION
    def posicionSuperior(pose, elevacion=elevacionAproximacion):
        p    = list(pose[:6])   
        p[2] = elevacion
        return p

    puntoSuperiorCubo    = posicionSuperior(coord)
    puntoSuperiorEspacio = posicionSuperior(espacio)
    puntoSuperiorModulo  = posicionSuperior(espacio)
    puntoSuperiorModulo[5] = anguloGripperModuloPiezas[2]

    #POSE DE AGARRE SOBRE EL OBJETO
    coordAgarre = [
        coord[0], coord[1], elevacionAgarre,
        RX_AGARRE, RY_AGARRE, RZ_AGARRE
    ]

    #POSE DE DEPOSITO EN EL SLOT DEL MODULO
    espacioDeposito = [
        espacio[0], espacio[1], elevacionAgarre,
        RX_AGARRE, RY_AGARRE, RZ_AGARRE
    ]

    print("Moviendo a punto seguro")
    moveL(puntoSeguro, speed=velocidad, accel=aceleracion); waitActivo(tEspera)

    print("Moviendo sobre el cubo")
    moveL(puntoSuperiorCubo, speed=velocidad, accel=aceleracion); waitActivo(tEspera)

    print(f"Rotando muñeca 3 → {anguloOBB:.3f} rad")
    
    #ALINEA LA MUNECA DEL ROBOT AL ANGULO DETECTADO POR OBB
    jointsActuales    = list(rtde_r.getActualQ())
    jointsActuales[5] = anguloOBB
    
    moveJ(jointsActuales, speed=0.1, accel=0.1); waitActivo(tEspera)

    print("Bajando al objeto")
    moveL(coordAgarre, speed=velocidad, accel=aceleracion); waitActivo(tEspera)

    print("Cerrando gripper")
    gripper.move_and_wait_for_pos(255, 255, 255); waitActivo(tEspera)

    #SUBE, TRASLADA Y DEPOSITA LA PIEZA EN EL MODULO
    moveL(puntoSuperiorCubo, speed=velocidad, accel=aceleracion); waitActivo(tEspera)
    moveL(puntoSuperiorModulo, speed=velocidad, accel=aceleracion); waitActivo(tEspera)
    moveL(espacioDeposito, speed=velocidad, accel=aceleracion); waitActivo(tEspera)

    print("Abriendo gripper")
    gripper.move_and_wait_for_pos(0, 255, 255); waitActivo(tEspera)

    moveL(puntoSuperiorEspacio, speed=velocidad, accel=aceleracion); waitActivo(tEspera)

    print("Regresando al punto seguro")
    moveL(puntoSeguro, speed=velocidad, accel=aceleracion); waitActivo(tEspera)

#CONVIERTE POSICION EN MM A COORDENADA TCP DEL MODULO DE PIEZAS
def posModulo(x_mm):
    return [
        x_mm / 1000.0,
        ejeModuloPiezas,
        elevacionModuloPiezas,
        *anguloGripperModuloPiezas,
        0.0   
    ]

#SLOTS DE DEPOSITO POR CLASE: [SLOT_0, SLOT_1, INDICE_ACTUAL]
coordenadasModulo = {
    "cuadrado":  [posModulo(-731.4),  posModulo(-787.8),   0],
    "triangulo": [posModulo(-973.85), posModulo(-1023.85), 0],
    "circulo":   [posModulo(-622.2),  posModulo(-677.5),   0],
    "estrella":  [posModulo(-862),    posModulo(-918),     0],
}

#DIBUJA DETECCIONES OBB EN EL FRAME Y RETORNA COORDENADAS Y CLASES SI SE SOLICITA
def displayFrame(name, frame, resultados_obb, rtn=False):
    color  = (255, 0, 0)
    coords = []
    clases = []

    if resultados_obb is not None and len(resultados_obb) > 0:
        xywhr_all    = resultados_obb.xywhr.cpu().numpy()
        xyxyxyxy_all = resultados_obb.xyxyxyxy.cpu().numpy()
        cls_all      = resultados_obb.cls.cpu().numpy()
        conf_all     = resultados_obb.conf.cpu().numpy()

        for i in range(len(cls_all)):
            centroX   = int(xywhr_all[i][0])
            centroY   = int(xywhr_all[i][1])
            anguloOBB = float(xywhr_all[i][4])
            clase     = model.names[int(cls_all[i])]
            confianza = float(conf_all[i])

            #DIBUJA CAJA OBB Y ETIQUETA SOBRE EL FRAME
            esquinas = xyxyxyxy_all[i].reshape((-1, 1, 2)).astype(int)
            cv2.polylines(frame, [esquinas], isClosed=True, color=color, thickness=2)
            cv2.putText(frame, f"{clase} {confianza:.2f}",
                        (centroX + 10, centroY + 20),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))

            coords.append((centroX, centroY, anguloOBB))
            clases.append(clase)

    cv2.imshow(name, frame)

    if rtn:
        return coords, clases

#MUESTRA DETECCIONES, COMPARA CALIBRACIONES Y EJECUTA EL ORDENAMIENTO DE PIEZAS
def frameFinal(frame, resultados_obb):
    coordenadas, clases = displayFrame("Objetos detectados", frame, resultados_obb, rtn=True)
    cv2.waitKey(1)

    print("\n" + "="*70)
    print("OBJETOS DETECTADOS")
    print("="*70)
    for i, coordenada in enumerate(coordenadas):
        print(f"  [{i}] {clases[i]:12s}  píxeles=({coordenada[0]}, {coordenada[1]})  "
              f"ángulo={coordenada[2]:.3f} rad")

    #IMPRIME TABLA COMPARATIVA DE COORDENADAS PARA TODOS LOS METODOS DE CALIBRACION
    print("\n" + "="*70)
    print("COMPARATIVA DE COORDENADAS POR MÉTODO DE CALIBRACIÓN")
    print("="*70)

    for i, coordenada in enumerate(coordenadas):
        clase = clases[i]
        print(f"\n  Objeto [{i}] — {clase}")
        print(f"  {'Método':<12}  {'X (mm)':>9}  {'Y (mm)':>9}  {'Z (mm)':>9}")
        print(f"  {'-'*45}")

        for nombre, (R_m, t_m) in CALIBRACIONES.items():
            try:
                c = convertirCoordenada(*coordenada, R=R_m, t=t_m)
                print(f"  {nombre:<12}  {c[0]*1000:>9.1f}  {c[1]*1000:>9.1f}  {c[2]*1000:>9.1f}")
            except Exception as e:
                print(f"  {nombre:<12}  ERROR: {e}")

    print("\n" + "="*70)
    print(f"Método activo: "
          f"{METODO_ACTIVO if METODO_ACTIVO else list(CALIBRACIONES.keys())[0]}")
    print("="*70)

    input("\nPresiona Enter para iniciar el ordenamiento...")
    cv2.waitKey(1)

    #RECORRE CADA OBJETO DETECTADO Y LO DEPOSITA EN SU SLOT CORRESPONDIENTE
    for i, coordenada in enumerate(coordenadas):
        clase = clases[i]
        if clase not in coordenadasModulo:
            print(f"  [OMITIDO] Clase '{clase}' no tiene posición de módulo definida.")
            continue

        info_modulo = coordenadasModulo[clase]
        idx_slot    = info_modulo[2]

        #OMITE LA CLASE SI YA SE LLENARON SUS DOS SLOTS
        if idx_slot > 1:
            print(f"  [OMITIDO] Clase '{clase}' ya tiene todos sus slots ocupados.")
            continue

        slot        = info_modulo[idx_slot]
        coord_robot = convertirCoordenada(*coordenada)

        print(f"\nRecogiendo {clase} → módulo slot {idx_slot}")
        print(f"  x={coord_robot[0]*1000:.1f}mm  y={coord_robot[1]*1000:.1f}mm  "
              f"z={coord_robot[2]*1000:.1f}mm  OBB={coord_robot[6]:.3f}rad")

        try:
            movimientoCoordenada(coord_robot, slot)
            #AVANZA EL INDICE DE SLOT TRAS DEPOSITAR EXITOSAMENTE
            coordenadasModulo[clase][2] = idx_slot + 1
        except Exception as e:
            print(f"  [ERROR] movimientoCoordenada falló para {clase}: {e}")
            input("  Presiona Enter para continuar con el siguiente objeto...")

    print("\nOrdenamiento completado.")

#SE CARGAN LOS INTRINSECOS DE LA CÁMARA OAK-1 LITE
camera_matrix, dist_coeffs = intrinsicosCamara(RESOLUCION_YOLO)
FX = camera_matrix[0, 0]
FY = camera_matrix[1, 1]
CX = camera_matrix[0, 2]
CY = camera_matrix[1, 2]

time.sleep(2)

#SE INICIALIZA EL PIPELINE DE DEPTHAI Y SE CONFIGURA LA SALIDA DE CAMARA
with dai.Pipeline() as pipeline:

    cam   = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    out_q = cam.requestOutput(
        RESOLUCION_YOLO,
        type=dai.ImgFrame.Type.BGR888p
    ).createOutputQueue(maxSize=1, blocking=False)

    pipeline.start()
    time.sleep(2)

    startTime  = time.monotonic()
    counter    = 0
    color2     = (255, 0, 255)
    ultimo_obb = None

    #BUCLE PRINCIPAL: CAPTURA FRAMES, CORRE YOLO Y DISPARA EL ORDENAMIENTO
    while pipeline.isRunning():
        inRgb = out_q.tryGet()

        if inRgb is not None:
            frame_yolo = inRgb.getCvFrame()
            counter   += 1

            #INFERENCIA YOLO-OBB SOBRE EL FRAME ACTUAL
            results    = model(frame_yolo, verbose=False, conf=0.6, iou=0.4)
            ultimo_obb = results[0].obb

            fps_str = "FPS: {:.2f}".format(counter / (time.monotonic() - startTime))
            cv2.putText(frame_yolo, fps_str, (2, frame_yolo.shape[0] - 4),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.4, color2)

            displayFrame("rgb", frame_yolo, ultimo_obb)

            #SI HAY DETECCIONES Y EL ROBOT ESTA LIBRE, INICIA EL ORDENAMIENTO
            if ultimo_obb is not None and len(ultimo_obb) > 0 and not robotOcupado:
                robotOcupado = True
                print("\nObjeto detectado — esperando 5 segundos para estabilizar...")
                waitActivo(5)
                frameFinal(frame_yolo, ultimo_obb)
                robotOcupado = False

        if cv2.waitKey(1) == ord("q"):
            pipeline.stop()
            break

#SE CIERRAN LAS VENTANAS DE OPENCV
cv2.destroyAllWindows()

#SE LIBERA EL GRIPPER Y SE DESCONECTAN LOS CLIENTES RTDE
gripper.move_and_wait_for_pos(0, 255, 255)
gripper.disconnect()

rtde_c.disconnect()
rtde_r.disconnect()

print("Sistema desconectado correctamente.")
