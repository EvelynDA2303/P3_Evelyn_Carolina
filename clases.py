"""
clases.py
Parcial 3 - Informática II (UdeA - Bioingeniería)

Contiene las clases para la gestión de volúmenes DICOM:
    - VolumenDicom    : un estudio (carpeta DICOM) con todos sus atributos y procesos.
    - GestorVolumenes : administra varios VolumenDicom (cargar, listar, seleccionar).

Librerías usadas (las vistas en clase): os, numpy, cv2, pydicom, matplotlib.
"""

import os
import numpy as np
import cv2
import pydicom
import matplotlib.pyplot as plt


class VolumenDicom:
    """Representa una carpeta de cortes DICOM como un volumen 3D.

    Convención del arreglo 3D:  volumen[z, y, x]
        z -> número de corte (SliceThickness)
        y -> filas             (PixelSpacing[0])
        x -> columnas          (PixelSpacing[1])

    Los planos "transversal", "coronal" y "sagital" se refieren a los ejes del
    arreglo (transversal = volumen[z,:,:], coronal = volumen[:,y,:],
    sagital = volumen[:,:,x]).
    """

    FACTOR_ESCALA = 2  # cada corte se escala al doble de su tamaño

    def __init__(self, carpeta, nombre=None, carpeta_salida="resultados"):
        # ---- Atributos de gestión
        self.carpeta = carpeta
        self.nombre = nombre if nombre else os.path.basename(os.path.normpath(carpeta))
        self.carpeta_salida = carpeta_salida
        os.makedirs(self.carpeta_salida, exist_ok=True)

        # ---- Atributos leídos directamente del dataset (ítem 2)
        self.filas = None
        self.columnas = None
        self.nombre_paciente = None
        self.modalidad = None
        self.pixel_spacing = None        # (alto_fila, ancho_columna) en mm - original
        self.slice_thickness = None      # mm
        self.tamano_voxel = None         # (alto, largo, ancho) en mm - original
        self.matriz_pixeles = None       # (n_cortes, filas, columnas) original

        # ---- Atributos del volumen escalado (ítem 3)
        self.tamano_voxel_escalado = None  # el pixel spacing se reduce a la mitad
        self.volumen = None                # (n_cortes, 2*filas, 2*columnas)

        # ---- Atributos de la corrección de ángulo (ítem 4)
        self.angulo = 0.0
        self.matriz_rotacion = None
        self.volumen_corregido = None

        # ---- Atributos de la ecualización (ítem 5)
        self.volumen_copia = None
        self.corte_ecualizado = None

        # ---- Atributos de la segmentación (ítem 6)
        self.ultima_segmentacion = None

        self._cargar_dicom()
        self._escalar_y_reconstruir()

    # ------------------------------------------------------------------
    # CARGA (ítem 2)
    # ------------------------------------------------------------------
    @staticmethod
    def _posicion_corte(ds):
        """Posición del corte a lo largo de la normal al plano (para ordenar)."""
        try:
            iop = np.array(ds.ImageOrientationPatient, dtype=float)
            ipp = np.array(ds.ImagePositionPatient, dtype=float)
            normal = np.cross(iop[:3], iop[3:])
            return float(np.dot(ipp, normal))
        except Exception:
            return float(getattr(ds, "InstanceNumber", None) or 0)

    def _cargar_dicom(self):
        datasets = []
        for archivo in os.listdir(self.carpeta):
            ruta = os.path.join(self.carpeta, archivo)
            if not os.path.isfile(ruta):
                continue
            try:
                ds = pydicom.dcmread(ruta)
                if hasattr(ds, "PixelData"):
                    datasets.append(ds)
            except Exception:
                continue  # no es un DICOM válido

        if not datasets:
            raise ValueError(f"No se encontraron archivos DICOM en: {self.carpeta}")

        # se ignoran archivos multicuadro / RGB (no son series de cortes 2D)
        datasets = [d for d in datasets if d.pixel_array.ndim == 2]
        if not datasets:
            raise ValueError(f"La carpeta {self.carpeta} no tiene cortes 2D en escala de grises")

        datasets.sort(key=self._posicion_corte)

        primero = datasets[0]
        self.filas = int(primero.Rows)
        self.columnas = int(primero.Columns)
        self.nombre_paciente = str(getattr(primero, "PatientName", "Desconocido"))
        self.modalidad = str(getattr(primero, "Modality", "Desconocida"))

        ps = getattr(primero, "PixelSpacing", [1.0, 1.0])
        self.pixel_spacing = (float(ps[0]), float(ps[1]))
        self.slice_thickness = float(getattr(primero, "SliceThickness", 1.0))
        # alto (fila), largo (columna) y ancho (grosor de corte), en mm
        self.tamano_voxel = (self.pixel_spacing[0], self.pixel_spacing[1], self.slice_thickness)

        cortes = [d.pixel_array for d in datasets
                  if d.pixel_array.shape == (self.filas, self.columnas)]
        if not cortes:
            raise ValueError(f"La carpeta {self.carpeta} no tiene cortes 2D en escala de grises")
        self.matriz_pixeles = np.stack(cortes, axis=0)

    # ------------------------------------------------------------------
    # ESCALADO Y RECONSTRUCCIÓN 3D (ítem 3)
    # ------------------------------------------------------------------
    def _escalar_y_reconstruir(self):
        f = self.FACTOR_ESCALA
        cortes_escalados = [
            cv2.resize(corte, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR)
            for corte in self.matriz_pixeles
        ]
        self.volumen = np.stack(cortes_escalados, axis=0)
        # al duplicar los pixeles, cada uno mide la mitad; el grosor de corte no cambia
        self.tamano_voxel_escalado = (self.pixel_spacing[0] / f,
                                      self.pixel_spacing[1] / f,
                                      self.slice_thickness)

    # ------------------------------------------------------------------
    # UTILIDADES INTERNAS
    # ------------------------------------------------------------------
    def _volumen_activo(self):
        """Usa el volumen corregido si ya se aplicó corregirAngulo; si no, el original."""
        return self.volumen_corregido if self.volumen_corregido is not None else self.volumen

    @staticmethod
    def _normalizar_plano(plano):
        p = plano.strip().lower()
        if p in ("transversal", "axial", "t", "1"):
            return "transversal"
        if p in ("coronal", "c", "2"):
            return "coronal"
        if p in ("sagital", "sagittal", "s", "3"):
            return "sagital"
        raise ValueError("Plano inválido. Use: transversal, coronal o sagital.")

    def numero_cortes(self, plano="transversal"):
        """Cantidad de cortes disponibles en el plano indicado."""
        plano = self._normalizar_plano(plano)
        nz, ny, nx = self._volumen_activo().shape
        return {"transversal": nz, "coronal": ny, "sagital": nx}[plano]

    def _obtener_corte(self, vol, plano, indice):
        """Extrae un corte del volumen y devuelve (corte, espaciado_vertical, espaciado_horizontal)."""
        plano = self._normalizar_plano(plano)
        sy, sx, sz = self.tamano_voxel_escalado
        n = {"transversal": vol.shape[0], "coronal": vol.shape[1], "sagital": vol.shape[2]}[plano]
        if not 0 <= indice < n:
            raise IndexError(f"Índice fuera de rango para el plano {plano}: 0 a {n - 1}")

        if plano == "transversal":
            return vol[indice, :, :], sy, sx
        if plano == "coronal":      # (z, x): se invierte z para que lo superior quede arriba
            return np.flipud(vol[:, indice, :]), sz, sx
        return np.flipud(vol[:, :, indice]), sz, sy   # sagital (z, y)

    @staticmethod
    def _a_isotropico(corte, esp_v, esp_h):
        """Corrige el 'aspect': remuestrea para que cada pixel sea cuadrado (esp_h x esp_h mm).
        Así los círculos y las áreas en mm² son correctos en cortes coronales/sagitales."""
        corte = np.ascontiguousarray(corte)
        factor = esp_v / esp_h
        if abs(factor - 1.0) > 1e-3:
            alto = max(1, int(round(corte.shape[0] * factor)))
            corte = cv2.resize(corte, (corte.shape[1], alto), interpolation=cv2.INTER_LINEAR)
        return corte, esp_h

    @staticmethod
    def _normalizar_uint8(corte):
        return cv2.normalize(corte.astype(np.float32), None, 0, 255,
                             cv2.NORM_MINMAX).astype(np.uint8)

    def info(self):
        print(f"\n=== {self.nombre} ===")
        print(f"Paciente          : {self.nombre_paciente}")
        print(f"Modalidad         : {self.modalidad}")
        print(f"Filas x Columnas  : {self.filas} x {self.columnas} (original)  np.shape -> {np.shape(self.matriz_pixeles)}")
        print(f"Número de cortes  : {self.matriz_pixeles.shape[0]}")
        print(f"PixelSpacing      : {self.pixel_spacing} mm | SliceThickness: {self.slice_thickness} mm")
        print(f"Voxel original    : {self.tamano_voxel} mm (alto, largo, ancho)")
        print(f"Voxel escalado    : {self.tamano_voxel_escalado} mm")
        print(f"Volumen 3D        : {self.volumen.shape} dtype={self.volumen.dtype}")
        print(f"Ángulo aplicado   : {self.angulo}°")

    # ------------------------------------------------------------------
    # CORRECCIÓN DE ÁNGULO (ítem 4)
    # ------------------------------------------------------------------
    ## Rota todos los cortes del volumen el angulo que escribe el usuario.
    # 1) Calcula la matriz de rotacion con el centro en la mitad de la imagen.
    # 2) Con warpAffine aplica esa matriz a cada corte.
    # 3) Guarda el resultado en self.volumen_corregido, partiendo siempre del volumen original.
    def corregirAngulo(self, angulo, mostrar=False):
        """Rota cada corte transversal 'angulo' grados (positivo = antihorario)
        alrededor del punto medio de la imagen. Resultado -> self.volumen_corregido."""
        alto, ancho = self.volumen.shape[1], self.volumen.shape[2]
        centro = (ancho / 2.0, alto / 2.0)
        self.matriz_rotacion = cv2.getRotationMatrix2D(centro, float(angulo), 1.0)
        relleno = float(self.volumen.min())

        rotados = [
            cv2.warpAffine(corte, self.matriz_rotacion, (ancho, alto),
                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                           borderValue=relleno)
            for corte in self.volumen   # siempre se rota desde el volumen original escalado
        ]
        self.volumen_corregido = np.stack(rotados, axis=0)
        self.angulo = float(angulo)

        # evidencia gráfica: corte central antes y después
        medio = self.volumen.shape[0] // 2
        fig, ejes = plt.subplots(1, 2, figsize=(10, 5))
        ejes[0].imshow(self.volumen[medio], cmap="gray")
        ejes[0].set_title("Original (corte central)")
        ejes[1].imshow(self.volumen_corregido[medio], cmap="gray")
        ejes[1].set_title(f"Corregido {angulo}°")
        for e in ejes:
            e.set_xlabel("columnas (px)")
            e.set_ylabel("filas (px)")
        fig.suptitle(f"{self.nombre} - corregirAngulo")
        ruta = os.path.join(self.carpeta_salida, f"{self.nombre}_rotacion_{angulo}.png")
        fig.savefig(ruta, dpi=120, bbox_inches="tight")
        print(f"Matriz de rotación:\n{self.matriz_rotacion}\nGuardado: {ruta}")
        if mostrar:
            plt.show()
        plt.close(fig)
        return self.volumen_corregido

    # ------------------------------------------------------------------
    # NORMALIZACIÓN + ECUALIZACIÓN (ítem 5)
    # ------------------------------------------------------------------
    # Hace una copia del volumen, toma un corte y lo pasa a uint8 (0 a 255).
    # Luego aplica equalizeHist para repartir mejor los grises y subir el contraste.
    # Al final grafica las dos imagenes con sus histogramas y guarda el grafico y la imagen.
    def ecualizarCorte(self, indice, plano="transversal", mostrar=True):
        """Copia el volumen, normaliza a uint8 el corte elegido y lo ecualiza.
        Grafica histogramas + imágenes y guarda el gráfico y la imagen ecualizada."""
        plano = self._normalizar_plano(plano)
        self.volumen_copia = self._volumen_activo().copy()

        corte, esp_v, esp_h = self._obtener_corte(self.volumen_copia, plano, indice)
        corte, _ = self._a_isotropico(corte, esp_v, esp_h)

        img8 = self._normalizar_uint8(corte)
        self.corte_ecualizado = cv2.equalizeHist(img8)

        # histogramas con cv2.calcHist (canal 0), como en el taller 1
        hist_orig = cv2.calcHist([img8], [0], None, [256], [0, 256])
        hist_eq = cv2.calcHist([self.corte_ecualizado], [0], None, [256], [0, 256])

        fig, ejes = plt.subplots(2, 2, figsize=(11, 9))
        ejes[0, 0].imshow(img8, cmap="gray", vmin=0, vmax=255)
        ejes[0, 0].set_title("Corte normalizado (uint8)")
        ejes[0, 1].imshow(self.corte_ecualizado, cmap="gray", vmin=0, vmax=255)
        ejes[0, 1].set_title("Corte ecualizado")
        ejes[1, 0].plot(hist_orig, color="gray")
        ejes[1, 0].set_title("Histograma original")
        ejes[1, 1].plot(hist_eq, color="gray")
        ejes[1, 1].set_title("Histograma ecualizado")
        for e in ejes[1]:
            e.set_xlim([0, 256])
            e.set_xlabel("Nivel de gris")
            e.set_ylabel("Frecuencia")
        fig.suptitle(f"{self.nombre} - plano {plano} - corte {indice}")

        base = f"{self.nombre}_{plano}_{indice}"
        ruta_fig = os.path.join(self.carpeta_salida, f"{base}_ecualizacion.png")
        ruta_img = os.path.join(self.carpeta_salida, f"{base}_ecualizada.png")
        fig.savefig(ruta_fig, dpi=120, bbox_inches="tight")
        cv2.imwrite(ruta_img, self.corte_ecualizado)
        print(f"Guardado: {ruta_fig}\nGuardado: {ruta_img}")
        if mostrar:
            plt.show()
        plt.close(fig)
        return self.corte_ecualizado

    # ------------------------------------------------------------------
    # SEGMENTACIÓN, APERTURA, COMPONENTES CONEXAS Y CÍRCULO (ítem 6)
    # ------------------------------------------------------------------
    # Busca huecos (zonas oscuras o claras) dentro de un corte.
    # 1) Normaliza el corte a uint8 y corrige el aspect para que el pixel sea cuadrado.
    # 2) Binariza con un umbral (el del usuario o Otsu si no se da).
    # 3) Hace una apertura con el kernel y las iteraciones elegidas para quitar ruido.
    # 4) Con connectedComponentsWithStats mide el area de cada region en pixeles.
    # 5) Pasa a RGB, dibuja el circulo sobre el hueco elegido y calcula el area en mm2.
    def segmentarHuecos(self, plano, indice, umbral=None, kernel=5, iteraciones=1,
                        oscuros=True, mostrar=True):
        """Segmenta 'huecos anatómicos' de un corte.

        umbral     : 0-255. Si es None se usa Otsu.
        kernel     : tamaño (kernel x kernel) de la apertura.
        iteraciones: iteraciones de la apertura.
        oscuros    : True -> los huecos son zonas oscuras (se invierte el umbral).
        """
        plano = self._normalizar_plano(plano)
        corte, esp_v, esp_h = self._obtener_corte(self._volumen_activo(), plano, indice)
        corte, px = self._a_isotropico(corte, esp_v, esp_h)   # pixel cuadrado de 'px' mm
        img8 = self._normalizar_uint8(corte)

        # 1) Binarización / umbralización
        tipo = cv2.THRESH_BINARY_INV if oscuros else cv2.THRESH_BINARY
        if umbral is None:
            umbral_usado, binaria = cv2.threshold(img8, 0, 255, tipo + cv2.THRESH_OTSU)
        else:
            umbral_usado, binaria = cv2.threshold(img8, int(umbral), 255, tipo)

        # 2) Apertura morfológica (kernel e iteraciones definidos por el usuario)
        ker = np.ones((int(kernel), int(kernel)), np.uint8)
        abierta = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, ker, iterations=int(iteraciones))

        # 3) Componentes conexas -> áreas
        n, etiquetas, stats, centroides = cv2.connectedComponentsWithStats(abierta, connectivity=8)
        alto, ancho = abierta.shape
        # Máscara del cuerpo: contorno externo más grande del tejido (lo que NO es hueco),
        # rellenado. Así se descartan la camilla y el aire de fuera del cuerpo.
        # En cortes sagitales/coronales el cuerpo suele quedar cortado arriba y abajo
        # (campo de visión limitado), así que ahí se añade una franja de tejido arriba y
        # abajo para que los huecos que tocan esos bordes cuenten como internos.
        tejido = cv2.bitwise_not(abierta)
        franja = 0 if plano == "transversal" else 3
        if franja:
            tejido = cv2.copyMakeBorder(tejido, franja, franja, 0, 0,
                                        cv2.BORDER_CONSTANT, value=255)
        # una apertura grande separa el cuerpo de objetos delgados pegados a él (camilla)
        k_cuerpo = max(15, (min(alto, ancho) // 25) | 1)
        tejido = cv2.morphologyEx(tejido, cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_cuerpo, k_cuerpo)))
        contornos, _ = cv2.findContours(tejido, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cuerpo = np.zeros_like(tejido)
        if contornos:
            mayor = max(contornos, key=cv2.contourArea)
            cv2.drawContours(cuerpo, [mayor], -1, 255, thickness=cv2.FILLED)
        if franja:
            cuerpo = cuerpo[franja:-franja, :]

        candidatas = []
        for i in range(1, n):  # 0 es el fondo
            x, y, w, h, area = stats[i]
            toca_lados = x == 0 or x + w >= ancho
            toca_borde = toca_lados or (franja == 0 and (y == 0 or y + h >= alto))
            dentro = cv2.countNonZero(cv2.bitwise_and(cuerpo, (etiquetas == i).astype(np.uint8) * 255)) == area
            candidatas.append((i, area, toca_borde, dentro))
        # primero: huecos completamente dentro del cuerpo; si no hay, los que no tocan el borde; si no, todos
        usar = ([c for c in candidatas if c[3] and not c[2]]
                or [c for c in candidatas if not c[2]]
                or candidatas)

        # 4) RGB para dibujar en color
        rgb = cv2.cvtColor(abierta, cv2.COLOR_GRAY2RGB)
        area_mm2, centro, radio = None, None, None
        if usar:
            etiqueta, area_px = max(usar, key=lambda c: c[1])[:2]
            x, y, w, h, _ = stats[etiqueta]
            centro = (int(centroides[etiqueta][0]), int(centroides[etiqueta][1]))
            radio = int(0.5 * max(w, h)) + 10
            area_mm2 = float(area_px) * px * px
            grosor = max(2, min(alto, ancho) // 150)
            cv2.circle(rgb, centro, radio, (255, 128, 0), grosor)

        alto_mm, ancho_mm = alto * px, ancho * px

        # 5) Gráfica
        fig, eje = plt.subplots(figsize=(8, 8))
        eje.imshow(rgb, aspect="equal")  # el corte ya viene con pixel cuadrado
        eje.set_title(f"{self.nombre} - plano {plano} - corte {indice}\n"
                      f"Umbral={umbral_usado:.0f}  Kernel={kernel}  Iteraciones={iteraciones}")
        eje.set_xlabel("columnas (px)")
        eje.set_ylabel("filas (px)")
        texto = (f"Dimensiones del corte: {ancho_mm:.1f} mm x {alto_mm:.1f} mm "
                 f"(PixelSpacing={self.pixel_spacing[0]:.3f} mm, SliceThickness={self.slice_thickness} mm)")
        texto += (f"\nÁrea del elemento de interés: {area_mm2:.2f} mm²" if area_mm2 is not None
                  else "\nNo se encontró ningún elemento de interés con estos parámetros")
        fig.text(0.5, 0.01, texto, ha="center", fontsize=9)

        base = f"{self.nombre}_{plano}_{indice}"
        ruta_fig = os.path.join(self.carpeta_salida, f"{base}_segmentacion.png")
        ruta_img = os.path.join(self.carpeta_salida, f"{base}_segmentada.png")
        fig.savefig(ruta_fig, dpi=120, bbox_inches="tight")
        cv2.imwrite(ruta_img, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        print(texto)
        print(f"Guardado: {ruta_fig}\nGuardado: {ruta_img}")
        if mostrar:
            plt.show()
        plt.close(fig)

        self.ultima_segmentacion = {
            "plano": plano, "indice": indice, "umbral": float(umbral_usado),
            "kernel": int(kernel), "iteraciones": int(iteraciones),
            "area_mm2": area_mm2, "centro": centro, "radio": radio,
            "dimensiones_mm": (ancho_mm, alto_mm),
        }
        return self.ultima_segmentacion

# Esta clase guarda varios volumenes DICOM a la vez en un diccionario.
# Permite cargar una carpeta, cargar todo el arbol de datos, listarlos y elegir cual usar.
class GestorVolumenes:
    """Administra varios VolumenDicom cargados al mismo tiempo."""

    def __init__(self, carpeta_salida="resultados"):
        self.carpeta_salida = carpeta_salida
        self.volumenes = {}   # nombre -> VolumenDicom
        self.activo = None    # nombre del volumen con el que se trabaja

    def cargar(self, carpeta, nombre=None):
        vol = VolumenDicom(carpeta, nombre, self.carpeta_salida)
        nombre = vol.nombre
        if nombre in self.volumenes:  # evita sobrescribir nombres repetidos
            nombre = f"{nombre}_{len(self.volumenes)}"
            vol.nombre = nombre
        self.volumenes[nombre] = vol
        self.activo = nombre
        return vol

    def cargar_arbol(self, raiz, prefijo=""):
        """Carga cada subcarpeta que contenga archivos DICOM directamente."""
        cargados = []
        for ruta, _, archivos in os.walk(raiz):
            if any(a.lower().endswith(".dcm") for a in archivos):
                # nombre = últimas carpetas de la ruta, ej. PPMI_3105_MPRAGE_GRAPPA
                partes = os.path.relpath(ruta, raiz).split(os.sep)
                nombre = "_".join(partes) if partes != ["."] else os.path.basename(raiz)
                if prefijo and not nombre.startswith(prefijo):
                    nombre = f"{prefijo}_{nombre}"
                try:
                    cargados.append(self.cargar(ruta, nombre))
                except ValueError as e:
                    print(f"Omitida {ruta}: {e}")
        return cargados

    def listar(self):
        if not self.volumenes:
            print("No hay volúmenes cargados.")
            return
        for nombre, v in self.volumenes.items():
            marca = "*" if nombre == self.activo else " "
            print(f" {marca} {nombre:35s} {v.modalidad}  {v.volumen.shape}  paciente={v.nombre_paciente}")

    def seleccionar(self, nombre):
        if nombre not in self.volumenes:
            raise KeyError(f"No existe el volumen '{nombre}'")
        self.activo = nombre

    def obtener_activo(self):
        if self.activo is None:
            raise RuntimeError("Primero cargue una carpeta DICOM.")
        return self.volumenes[self.activo]
