# Parcial 3 - Informatica II (UdeA - Bioingenieria)

Programa en Python para analizar imagenes DICOM del Hospital San Vicente Fundacion.

## Que hace
- Carga carpetas DICOM y lee filas, columnas, paciente, modalidad, tamano del voxel y matriz de pixeles.
- Escala cada corte al doble y reconstruye el volumen 3D.
- Corrige el angulo de inclinacion del volumen (corregirAngulo).
- Normaliza a uint8 un corte, lo ecualiza y grafica los histogramas.
- Segmenta huecos anatomicos, hace una apertura, mide el area en mm2 y dibuja un circulo.

## Archivos
- clases.py: clases VolumenDicom y GestorVolumenes.
- main.py: menu para usar el programa.

## Como instalar
    pip install -r requirements.txt

## Como ejecutar
Poner la carpeta datos (con PPMI, Sarcoma y T2) junto a main.py y correr:

    python main.py

Los resultados quedan en la carpeta resultados.