"""
main.py - Implementación del Parcial 3 (menú).
Ejecutar:  python main.py
"""

import os
import matplotlib.pyplot as plt
from clases import GestorVolumenes

RAIZ_DATOS = "datos"   # carpeta con PPMI, Sarcoma y T2
SUBCARPETAS = ["PPMI", "Sarcoma", "T2"]  # las del enunciado (se ignora "series")


# ---------------------------- utilidades de entrada ----------------------------
def pedir_entero(mensaje, minimo=None, maximo=None, defecto=None):
    while True:
        txt = input(mensaje).strip()
        if txt == "" and defecto is not None:
            return defecto
        try:
            valor = int(txt)
            if (minimo is not None and valor < minimo) or (maximo is not None and valor > maximo):
                print(f"  Valor fuera de rango [{minimo}, {maximo}].")
                continue
            return valor
        except ValueError:
            print("  Ingrese un número entero.")


def pedir_float(mensaje):
    while True:
        try:
            return float(input(mensaje).strip().replace(",", "."))
        except ValueError:
            print("  Ingrese un número.")


def pedir_plano():
    print("  Plano: 1) transversal  2) coronal  3) sagital")
    op = pedir_entero("  Elija (1-3): ", 1, 3)
    return {1: "transversal", 2: "coronal", 3: "sagital"}[op]


# ---------------------------- acciones del menú ----------------------------
def cargar_carpeta(gestor):
    ruta = input("Ruta de la carpeta DICOM: ").strip().strip('"')
    if not os.path.isdir(ruta):
        print("  La carpeta no existe.")
        return
    nombre = input("Nombre para este volumen (Enter = nombre de la carpeta): ").strip() or None
    try:
        gestor.cargar(ruta, nombre).info()
    except ValueError as e:
        print(f"  Error: {e}")


def cargar_todo(gestor):
    if not os.path.isdir(RAIZ_DATOS):
        print(f"  No se encuentra la carpeta '{RAIZ_DATOS}'.")
        return
    cargados = []
    for sub in SUBCARPETAS:
        ruta = os.path.join(RAIZ_DATOS, sub)
        if os.path.isdir(ruta):
            cargados += gestor.cargar_arbol(ruta, prefijo=sub)
        else:
            print(f"  No se encontró {ruta}")
    print(f"\nSe cargaron {len(cargados)} volúmenes:")
    gestor.listar()


def seleccionar(gestor):
    gestor.listar()
    nombre = input("Nombre del volumen a usar: ").strip()
    try:
        gestor.seleccionar(nombre)
    except KeyError as e:
        print(f"  {e}")


def accion_angulo(gestor):
    vol = gestor.obtener_activo()
    angulo = pedir_float("Ángulo de inclinación en grados (+ antihorario): ")
    vol.corregirAngulo(angulo, mostrar=True)


def accion_ecualizar(gestor):
    vol = gestor.obtener_activo()
    plano = pedir_plano()
    n = vol.numero_cortes(plano)
    indice = pedir_entero(f"  Número de corte (0 a {n - 1}, Enter = central): ", 0, n - 1, n // 2)
    vol.ecualizarCorte(indice, plano)


def accion_segmentar(gestor):
    vol = gestor.obtener_activo()
    plano = pedir_plano()
    n = vol.numero_cortes(plano)
    indice = pedir_entero(f"  Número de corte (0 a {n - 1}, Enter = central): ", 0, n - 1, n // 2)
    txt = input("  Umbral 0-255 (Enter = automático Otsu): ").strip()
    umbral = int(txt) if txt else None
    kernel = pedir_entero("  Tamaño del kernel de la apertura (ej. 5): ", 1)
    iteraciones = pedir_entero("  Iteraciones de la apertura (ej. 2): ", 1)
    oscuros = input("  ¿Los huecos son zonas oscuras? (s/n, Enter = s): ").strip().lower() != "n"
    vol.segmentarHuecos(plano, indice, umbral, kernel, iteraciones, oscuros)


def demo_automatica(gestor):
    """Corre todo el flujo sobre cada volumen cargado y guarda los resultados de prueba."""
    if not gestor.volumenes:
        cargar_todo(gestor)
    for nombre, vol in gestor.volumenes.items():
        print(f"\n########## {nombre} ##########")
        vol.corregirAngulo(5)
        medio = vol.numero_cortes("transversal") // 2
        vol.ecualizarCorte(medio, "transversal", mostrar=False)
        if nombre.startswith("T2"):
            # en T2 el líquido (ventrículos) es brillante: se buscan zonas claras con umbral manual
            vol.segmentarHuecos("transversal", medio, 150, 5, 2, False, mostrar=False)
        else:
            vol.segmentarHuecos("transversal", medio, None, 5, 2, True, mostrar=False)
        if nombre.startswith("PPMI"):
            # el corte sagital funciona bien en PPMI (cabeza completa)
            vol.segmentarHuecos("sagital", vol.numero_cortes("sagital") // 2, None, 5, 2, True, mostrar=False)
    print("\nDemo terminada. Revise la carpeta 'resultados'.")


# ---------------------------- menú principal ----------------------------
def menu():
    gestor = GestorVolumenes("resultados")
    opciones = {
        "1": ("Cargar una carpeta DICOM", cargar_carpeta),
        "2": ("Cargar todas las carpetas de 'datos' (PPMI, Sarcoma, T2)", cargar_todo),
        "3": ("Listar volúmenes cargados", lambda g: g.listar()),
        "4": ("Seleccionar volumen activo", seleccionar),
        "5": ("Información del volumen activo", lambda g: g.obtener_activo().info()),
        "6": ("Corregir ángulo (corregirAngulo)", accion_angulo),
        "7": ("Normalizar y ecualizar un corte", accion_ecualizar),
        "8": ("Segmentar huecos, apertura, área y círculo", accion_segmentar),
        "9": ("Demo automática sobre todos los volúmenes", demo_automatica),
        "0": ("Salir", None),
    }
    while True:
        print("\n========== ANÁLISIS DE IMÁGENES DICOM ==========")
        if gestor.activo:
            print(f"Volumen activo: {gestor.activo}")
        for k, (texto, _) in opciones.items():
            print(f"  {k}. {texto}")
        op = input("Opción: ").strip()
        if op == "0":
            print("Hasta luego.")
            break
        if op not in opciones:
            print("Opción inválida.")
            continue
        try:
            opciones[op][1](gestor)
        except (RuntimeError, IndexError, ValueError, KeyError) as e:
            print(f"  Error: {e}")
        plt.close("all")


if __name__ == "__main__":
    menu()
