# Sistema de Visión para Robot Colaborativo UR10

Sistema de pick & place automatizado que combina una cámara OAK con detección de objetos mediante YOLOv8-OBB y control de un robot UR10 con gripper Robotiq.

---

## 1. Preparación

En la presente sección se indicarán los pasos a seguir para utilizar correctamente el módulo. Es de suma importancia seguir cada paso

### 1.1 Instalar Python 3.11

> Descargar desde [python.org](https://www.python.org/downloads/release/python-3119/) 

Durante la instalación marcar la opción **"Add Python to PATH"**.

### 1.2 Instalar Visual Studio Code

Descargar desde [code.visualstudio.com](https://code.visualstudio.com/) e instalar la extensión **Python** (Microsoft).


### 1.3 Instalar CMake y Visual Studio Build Tools

`ur_rtde` se compila desde código fuente y requiere estas herramientas:

1. Descargar e instalar [CMake](https://cmake.org/download/) — marcar **"Add CMake to the system PATH"**
2. Descargar [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) e instalar el componente **"Desarrollo de escritorio con C++"**

### 1.4 Crear entorno virtual

Abrir una terminal en la carpeta del proyecto:

```bash
python -m venv .venv
```

Activar el entorno:

```bash
# Windows CMD
.venv\Scripts\activate.bat

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

El prompt debe mostrar `(.venv)` al inicio.

### 1.5 Instalar dependencias

Con el entorno virtual activado:

```bash
pip install ur_rtde
pip install ultralytics
pip install depthai
pip install opencv-python
pip install numpy
```

Verificar que todo se instaló correctamente:

```bash
python -c "from rtde_control import RTDEControlInterface; print('ur_rtde OK')"
python -c "from ultralytics import YOLO; print('ultralytics OK')"
python -c "import depthai; print('depthai OK')"
```

### 1.6 Estructura sugerida del proyecto

```
ProyectoIntegrador_2/
└── scriptsPrueba/
│   ├── pruebaConexion.py
│   └── pruebaGripper.py
│   ├── robotiqGripper.py
│   ├── robotiq_gripper_control.py
│   ├── robotiq_preamble.py
│   └── testCamara.py
├── main.py
├── robotiqGripper.py
├── models/
│   └── best.pt
└── calibracion/
    ├── calibracionCamara.py
    ├── intrinsicosCamara.py
    ├── imagenesDataset/
    └── matriz.npy
```

> `matriz.npy` se genera al ejecutar `calibracionCamara.py`. Si aún no existe, ver la sección de calibración más adelante.

---

## 2. Verificación con scripts de prueba

Antes de ejecutar el sistema completo, verificar cada componente por separado en el siguiente orden.

### 2.1 Verificar la cámara — `testCamara.py`

Conectar la cámara OAK por USB y ejecutar:

```bash
python testCamara.py
```

Se abrirá una ventana con el feed de la cámara. Usar los siguientes controles para ajustar el foco:

| Tecla | Acción |
|-------|--------|
| `+`   | Aumentar foco en 5 |
| `-`   | Reducir foco en 5 |
| `F`   | Aumentar foco en 1 (fino) |
| `D`   | Reducir foco en 1 (fino) |
| `Q`   | Salir y mostrar el valor final |

Anotar el valor de foco final. Si la imagen se ve nítida, la cámara funciona correctamente.

### 2.2 Verificar la conexión con el robot — `pruebaConexion.py`

> **Seguridad:** Despejar el área de trabajo del robot antes de ejecutar. El robot se moverá por una secuencia de posiciones predefinidas.

Verificar que la IP del robot en el archivo sea correcta (`192.168.56.101` por defecto) y que el robot esté en **modo remoto**.

```bash
python pruebaConexion.py
```

El robot debe moverse por 5 posiciones de prueba y regresar al origen. Si completa el ciclo sin errores, la conexión RTDE funciona correctamente.

### 2.3 Verificar el gripper — `pruebaGripper.py`

```bash
python pruebaGripper.py
```

El gripper debe cerrarse completamente y luego abrirse. La terminal mostrará la posición actual en cada paso. Si el gripper no responde, verificar:

- Que el ID del gripper sea **1** en el Teach Pendant (Installation → URCaps → Gripper)
- Que el robot esté encendido y conectado a la red
- El archivo robotiqGripper.py está presente en el mismo nivel de archivos que pruebaGripper.py

### 2.4 Calibración ojo-mano — `calibracionCamara.py`

Este paso solo es necesario si `calibracion/matriz.npy` no existe o si se cambia la posición de la cámara respecto al robot.

```bash
python calibracionCamara.py
```

El script guiará el proceso:

1. Se abre el pipeline de la cámara
2. Presionar `C` para capturar una foto (mover el robot manualmente a distintas posiciones entre fotos)
3. Capturar mínimo **15 fotos** con el tablero de ajedrez visible desde distintos ángulos
4. Presionar `Enter` para terminar la captura e iniciar la calibración automáticamente
5. El archivo `calibracion/matriz.npy` se genera al finalizar

> El tablero de calibración debe ser de **6×9 cuadros** con cuadros de **2.5 cm**. Se adjunta el archivo png proporcionado directamente por OpenCV.

![Tablero de calibración para módulo.](/imagenes/pattern.png "Tablero de calibración para módulo.")

---

## 3. Uso del módulo principal — `main.py`

### 3.1 Configurar parámetros antes de ejecutar

Abrir `main.py` y revisar las siguientes variables:

```python
# IP del robot — verificar que coincida con la configuración de red
HOST = "192.168.56.101"

# Ruta al modelo YOLOv8-OBB entrenado
modeloUsado = "models/best.pt"

# Profundidad de la cámara a la mesa de trabajo (metros)
# Buscar la línea: profundidad = None  y reemplazar con el valor medido
profundidad = 0.65  # ejemplo: 65 cm
```

Verificar también las coordenadas del módulo de piezas al fondo del archivo:

```python
coordenadasModulo = {
    "cuadrado":  [posModulo(-731.4), posModulo(-787.8), 0],
    "triangulo": [posModulo(0),      posModulo(0),      0],
    "circulo":   [posModulo(-622.2), posModulo(-677.5), 0],
    "estrella":  [posModulo(-862),   posModulo(0),      0],
}
```

### 3.2 Ejecutar

Con el entorno virtual activado y el robot en **modo remoto**:

```bash
python main.py
```

### 3.3 Flujo de ejecución

```
Inicio
  │
  ├─ Conecta al robot (RTDE) y al gripper
  ├─ Carga el modelo YOLO
  ├─ Mueve el robot al puntoSeguro
  │
  └─ Loop principal (cámara OAK)
       │
       ├─ Captura frame
       ├─ Corre inferencia OBB con YOLO
       ├─ Muestra detecciones en pantalla
       │
       └─ Si detecta objetos y robot libre:
            ├─ Espera 5 segundos para estabilizar
            ├─ Muestra detecciones finales
            ├─ [Enter] para confirmar el ordenamiento
            └─ Mueve cada caja a su posición en el módulo
```

### 3.4 Controles durante la ejecución

| Acción | Descripción |
|--------|-------------|
| `Q` en la ventana de video | Detiene el sistema y cierra conexiones |
| `Enter` en la terminal | Confirma el inicio del ordenamiento de cajas |

### 3.5 Salida esperada en terminal

```
Robot conectado (ur_rtde)
Se han detectado los siguientes objetos:
  cuadrado con coordenadas (312, 245, 0.34)
  circulo  con coordenadas (480, 310, -0.12)

Presiona Enter para continuar con el ordenamiento de las cajas...

Moviendo caja 'cuadrado' → [-0.7314, -0.365, -0.38451, 2.226, 2.183, 0.0]
Moviendo caja 'circulo'  → [-0.6222, -0.365, -0.38451, 2.226, 2.183, 0.0]
```

---

## Solución de problemas frecuentes

**`ur_rtde` no instala en Python 3.12/3.13**
Usar Python 3.11 instalado desde python.org (no Microsoft Store).

**Error 403 al hacer `git push`**
El usuario de Git no coincide con el dueño del repositorio. Corregir con:
```bash
git remote set-url origin https://TU_USUARIO@github.com/TU_USUARIO/repo.git
```

**El gripper no responde**
Verificar que el ID del gripper sea 1 en el Teach Pendant y que el archivo del gripper se encuentre en el directorio del archivo a ejecutar.
**`calibracion/matriz.npy` no encontrado**
Ejecutar `calibracionCamara.py` para generar el archivo de calibración.

**La cámara no detecta el tablero de calibración**
Asegurarse de que haya buena iluminación y que el tablero sea de exactamente 6×9 cuadros internos.
