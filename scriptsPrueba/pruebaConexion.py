
#Prueba básica d econexión y control del robot mediante la libreria UR_RTDE

from robotiq_gripper_control import RobotiqGripper
from rtde_control import RTDEControlInterface 
import math

import time

def gradosRadianes(angulos):
    return [math.radians(a) for a in angulos]

#SE EJECUTARAN LOS WAYPOINTS INDICADOS EN EL PROGRAMA posSistemaVision GUARDADO EN EL TEACH PENDANT
#VERIFICAR QUE NO HAYA COLISION ENTRE ESOS PUNTOS ANTES DE EJECUTAR ESTE PROGRAMA

posicionesPuntoSeguro = [
    [0.00, -90, 0, -90, 0, 0], #origen
    [-55.31, -90, -0.17, -89.91, -0.7, -0.06],
    [-142.38, -87.14, -56.02, -41.08, 89.58, 32.49],
    [-142.38, -87.26, -97.47, -76.34, 89.58, 32.49],
    [-154.34, -103.2, -121.62, -45, 91.2, 25.18]
]

velocidad, aceleracion = 1, 1
ip = "192.168.56.101"
print("Conectando...")

rtde_c = RTDEControlInterface(ip)  # Cambia por la IP mostrada en la configuración del UR
print("Conexion establecida con robot")

gripper = RobotiqGripper(rtde_c)
print("Gripper conectado")

gripper.activate()

print("Gripper activado")

for i, pos in enumerate(posicionesPuntoSeguro):
    
    rtde_c.moveJ(gradosRadianes(pos), velocidad, aceleracion)
    print(f"Pos {i} alcanzada")


i = 4

for pos in reversed(posicionesPuntoSeguro):
    rtde_c.moveJ(gradosRadianes(pos),velocidad, aceleracion)
    print(f"Pos {i} alcanzada")

    i -= 1


rtde_c.stopRobot()