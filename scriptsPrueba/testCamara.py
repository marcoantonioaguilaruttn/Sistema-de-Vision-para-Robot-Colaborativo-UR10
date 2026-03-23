#PROGAMA PARA VERIFICAR LA CONEXIÓN CON LA CÁMARA OAK 1 LITE

import cv2
import depthai as dai
import time

RESOLUCION = (3840, 2160)

with dai.Pipeline() as pipeline:

    focus_val = 115

    ctrl = dai.CameraControl()
    ctrl.setManualFocus(focus_val)

    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    cam_q_in = cam.inputControl.createInputQueue()
    cam_q_in.send(ctrl)

    out_q = cam.requestOutput(
                RESOLUCION,
                type=dai.ImgFrame.Type.BGR888p
            ).createOutputQueue(maxSize=8, blocking=False)

    pipeline.start()
    print("Calibración de foco | +/- ajuste de 5 | F/D ajuste fino de 1 | Q salir")
    print("─" * 50)

    cv2.namedWindow("Calibracion de foco", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Calibracion de foco", 1280, 720)

    def capturar_frame():
        frame_dai = None
        for _ in range(10):
            got = out_q.tryGet()
            if got is not None:
                frame_dai = got
        return frame_dai.getCvFrame() if frame_dai else None

    def mostrar_frame(frame, focus_val):
        display = frame.copy()
        cv2.putText(display, f"Focus: {focus_val}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display, "+/- ajuste | F/D fino | Q salir", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        cv2.imshow("Calibracion de foco", display)

    # Foto inicial
    time.sleep(1)
    frame = capturar_frame()
    if frame is not None:
        mostrar_frame(frame, focus_val)
        print(f"Focus inicial: {focus_val}")

    while pipeline.isRunning():
        key = cv2.waitKey(0) & 0xFF

        changed = True
        if key == ord('q'):
            print(f"\nValor final → ctrl.setManualFocus({focus_val})")
            break
        elif key == ord('+'):
            focus_val = min(255, focus_val + 5)
        elif key == ord('-'):
            focus_val = max(0, focus_val - 5)
        elif key == ord('f'):
            focus_val = min(255, focus_val + 1)
        elif key == ord('d'):
            focus_val = max(0, focus_val - 1)
        else:
            changed = False

        if changed:
            ctrl = dai.CameraControl()
            ctrl.setManualFocus(focus_val)
            cam_q_in.send(ctrl)
            print(f"Focus: {focus_val}", end='\r')

            time.sleep(0.3)
            frame = capturar_frame()
            if frame is not None:
                mostrar_frame(frame, focus_val)

cv2.destroyAllWindows()