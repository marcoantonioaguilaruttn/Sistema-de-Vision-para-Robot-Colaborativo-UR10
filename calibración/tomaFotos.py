import cv2
import numpy as np
import depthai as dai
import time
import os

# ─────────────────────────────────────────────
#  CONFIGURACIÓN
# ─────────────────────────────────────────────
RESOLUCION      = (1280, 720)
DIRECTORIO      = "C:/Users/marco/Downloads/Proyectos/ProyectoIntegrador_2/dataset/imagenes"
LIMITE_FOTOS    = 1000
index           = 0

os.makedirs(DIRECTORIO, exist_ok=True)

# ─────────────────────────────────────────────
#  GUARDAR IMAGEN
# ─────────────────────────────────────────────
def guardarImagen(i, frame):
    i += 1
    direccionImagen = os.path.join(DIRECTORIO, f"imagen{i}.png")
    cv2.imwrite(direccionImagen, frame)
    print(f"[{i}] Imagen guardada → {direccionImagen} | Tamaño: {frame.shape}")
    return i

# ─────────────────────────────────────────────
#  PIPELINE DEPTHAI — solo 1280x720, sin 4K
# ─────────────────────────────────────────────
with dai.Pipeline() as pipeline:

    camSocket = dai.CameraBoardSocket.CAM_A
    print("Camara conectada")

    # Cámara directo en 1280x720 (no pedimos full-res)
    cam = pipeline.create(dai.node.Camera).build(camSocket)

    imgManip = pipeline.create(dai.node.ImageManip)
    cam.video.link(imgManip.inputImage)
    imgManip.initialConfig.setOutputSize(RESOLUCION[0], RESOLUCION[1])
    imgManip.setMaxOutputFrameSize(RESOLUCION[0] * RESOLUCION[1] * 3)

    # Cola de salida para preview y captura
    out_q = imgManip.out.createOutputQueue(maxSize=8, blocking=False)

    pipeline.start()
    print("Pipeline comenzado — resolución objetivo: 1280x720")
    print("─" * 50)

    ultimo_frame = None

    while pipeline.isRunning():
        if index >= LIMITE_FOTOS:
            print("Límite de fotos alcanzado.")
            break

        # Actualizar frame en buffer continuamente
        frame_dai = out_q.tryGet()
        if frame_dai is not None:
            ultimo_frame = frame_dai.getCvFrame()

        os.system('cls')
        print(f"Fotos tomadas: {index} / {LIMITE_FOTOS}")
        print(f"Resolución: {RESOLUCION[0]}x{RESOLUCION[1]}")
        print("─" * 50)
        tecla = str(input("Presiona C para capturar | cualquier otra tecla para salir: ")).strip().upper()

        if tecla != "C":
            print("Saliendo...")
            break

        # Obtener frame más reciente antes de guardar
        frame_dai = out_q.tryGet()
        if frame_dai is not None:
            ultimo_frame = frame_dai.getCvFrame()

        if ultimo_frame is not None:
            index = guardarImagen(index, ultimo_frame)
        else:
            print("⚠ No se recibió frame, intenta de nuevo.")

    print(f"\nProceso terminado. Se guardaron {index} imágenes en:\n{DIRECTORIO}")