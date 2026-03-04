import cv2
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive
import numpy as np
import depthai as dai
import time
import os
import intrinsicosCamara as intr

robot_ip = "192.168.56.101"
rtde_frequency = 300.0

ctdFotos, index = 1000, 0

#interfazControl = RTDEControl(robot_ip)
#interfazReceive = RTDEReceive(robot_ip, rtde_frequency)

T_base2gripper, R_base2gripper = [], []

def guardarImagen(i, frame):
    i += 1
    direccionImagen = f"C:/Users/marco/Downloads/Proyectos/ProyectoIntegrador_2/calibración/imagenesDataset/imagen{i}.png"

    imgSize = (1280, 720)
    imagen = cv2.resize(frame, imgSize)
    cv2.imwrite(direccionImagen, imagen)

    print(f"Imagen {i} guardada exitosamente en {direccionImagen}")
    return i

def obtenerCoordenadas():
    coordenadasTCP = interfazReceive.getActualTCPPose()
    translacionGripper = np.array(coordenadasTCP[0:3])

    rotacionGripper = np.array(coordenadasTCP[3:])
    rotacionGripper, _ = cv2.Rodrigues(rotacionGripper)

    # Eye-to-Hand: invertir gripper→base para obtener base→gripper
    # R^-1 = R^T  (matrices de rotación son ortogonales)
    # t^-1 = -R^T * t
    R_base2gripper = rotacionGripper.T
    T_base2gripper = -R_base2gripper @ translacionGripper

    return T_base2gripper, R_base2gripper


def calibracionOjoMano(cnt):
    camera_matrix, dist_coeffs = intr.intrinsicosCamara()
    direccionMatriz = "calibración/matriz.npy"

    arrRvec, arrTvec = [], []
    R_base2gripper_validos, T_base2gripper_validos = [], []

    tamañoTablero = (9, 5)
    tamañoCuadro  = 0.025

    objp = np.zeros((tamañoTablero[0] * tamañoTablero[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:tamañoTablero[0], 0:tamañoTablero[1]].T.reshape(-1, 2) * tamañoCuadro

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for i in range(1, cnt + 1):
        dir = f"calibración/imagenes/imagen{i}.png"
        img = cv2.imread(dir, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"No se pudo leer {dir}, se omite.")
            continue

        ret, esquinasImagen = cv2.findChessboardCorners(img, tamañoTablero)

        if ret:
            esquinasImagen = cv2.cornerSubPix(img, esquinasImagen, (11, 11), (-1, -1), criteria)

            ret2, rvec, tvec = cv2.solvePnP(
                objp, esquinasImagen, camera_matrix, dist_coeffs
            )

            if ret2:
                arrRvec.append(rvec)
                arrTvec.append(tvec)

                R_base2gripper_validos.append(R_base2gripper[i - 1])
                T_base2gripper_validos.append(T_base2gripper[i - 1])
        else:
            print(f"No se encontraron esquinas en imagen{i}.png, se omite.")

    if len(arrRvec) < 3:
        print(f"ERROR: Solo {len(arrRvec)} imágenes válidas. Se necesitan al menos 3.")
        return

    print(f"Calibrando con {len(arrRvec)} imágenes válidas...")

    r_cam2base, t_cam2base = cv2.calibrateHandEye(
        R_base2gripper_validos,
        T_base2gripper_validos,
        arrRvec,
        arrTvec
    )

    np.save(direccionMatriz, (r_cam2base, t_cam2base))
    print(f"Matriz guardada en {direccionMatriz}")
    print("R_cam2base:\n", r_cam2base)
    print("T_cam2base:\n", t_cam2base)


with dai.Pipeline() as pipeline:

    camSocket = dai.CameraBoardSocket.CAM_A
    print("Camara conectada")

    cam = pipeline.create(dai.node.Camera).build(camSocket)
    cam_q_in = cam.inputControl.createInputQueue()

    highest_Res = cam.requestFullResolutionOutput(useHighestResolution=True)

    script = pipeline.create(dai.node.Script)
    highest_Res.link(script.inputs["in"])

    script.setScript("""
        while True:
            frame = node.inputs["in"].get()
            trigger = node.inputs["trigger"].tryGet()
            if trigger is not None:
                node.warn("Trigger recibido → enviando imagen full-res")
                node.io["highest_res"].send(frame)
    """)

    imgManip = pipeline.create(dai.node.ImageManip)
    highest_Res.link(imgManip.inputImage)
    imgManip.initialConfig.setOutputSize(1280, 720)
    fourmbSize = 1280 * 720 * 3
    imgManip.setMaxOutputFrameSize(fourmbSize)

    downscaled_res_q = imgManip.out.createOutputQueue(maxSize=8, blocking=False)
    highest_res_q    = script.outputs["highest_res"].createOutputQueue(maxSize=4)
    q_trigger        = script.inputs["trigger"].createInputQueue()

    pipeline.start()
    print("Pipeline comenzado")

    while pipeline.isRunning():
        if index >= ctdFotos:
            print("Límite de fotos alcanzado.")
            break

        os.system('cls')
        tecla = str(input("Presiona C para tomar una foto (o Enter para salir): ")).strip().upper()

        if tecla != "C":
            print("Saliendo del bucle...")
            break

        imagen = downscaled_res_q.tryGet()
        if imagen is not None:
            frame_preview = imagen.getCvFrame()

        q_trigger.send(dai.Buffer())
        time.sleep(0.3)

        if highest_res_q.has():
            highres_img = highest_res_q.get()
            frame_full  = highres_img.getCvFrame()
            index = guardarImagen(index, frame_full)

            #tImagen, rImagen = obtenerCoordenadas()
            #T_base2gripper.append(tImagen)   # listas globales base→gripper
            #R_base2gripper.append(rImagen)

    print(f"Proceso terminado. Se guardaron {index} imágenes.")

#calibracionOjoMano(index)