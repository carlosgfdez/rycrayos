import streamlit as st
import pydicom
import numpy as np
import cv2
import matplotlib.pyplot as plt
from io import BytesIO
from pydicom.filebase import DicomBytesIO

st.set_page_config(layout="wide")
st.title("Conversión de Mamografía DICOM a Color")

# 📌 Función para detectar tipo de imagen
def detectar_tipo_imagen(ds, nombre_archivo=""):
    series_desc = getattr(ds, "SeriesDescription", "").lower()
    image_type = getattr(ds, "ImageType", [])
    nombre = nombre_archivo.lower()

    if "ticem" in nombre or "ticem" in series_desc or any("subtracted" in str(x).lower() for x in image_type):
        return "Substracción (TiCEM)"
    elif "insight" in nombre or "low" in series_desc or "le" in series_desc:
        return "Baja energía (INSIGHT)"
    else:
        return "Desconocido"

# 🔽 Subida de archivo
archivo_dicom = st.file_uploader("Sube un archivo DICOM", type=None)

if archivo_dicom is not None:
    try:
        dcm_data = archivo_dicom.read()
        dicom_file = DicomBytesIO(dcm_data)
        ds = pydicom.dcmread(dicom_file, force=True)

        try:
            img = ds.pixel_array.astype(np.float32)
        except Exception as e:
            st.error(f"⚠️ No se pudo obtener la imagen del DICOM: {e}")
            st.stop()

        # Detectar tipo de imagen
        tipo_imagen = detectar_tipo_imagen(ds, archivo_dicom.name)

        # Mostrar metadatos
        st.markdown(f"**📄 Archivo cargado:** `{archivo_dicom.name}`")
        st.markdown(f"**🧾 NHC:** `{getattr(ds, 'PatientID', 'Desconocido')}`")
        st.markdown(f"**🩻 Modalidad:** `{getattr(ds, 'Modality', 'N/A')}`")
        st.markdown(f"**📐 Tamaño:** `{img.shape}`")
        st.markdown(f"**🖼️ Tipo de imagen:** `{tipo_imagen}`")

        # Procesamiento adaptado
        if tipo_imagen == "Substracción (TiCEM)":
            vmin = np.percentile(img, 5)
            vmax = np.percentile(img, 99.9)
            colormap = cv2.COLORMAP_TURBO
        elif tipo_imagen == "Baja energía (INSIGHT)":
            vmin = np.percentile(img, 1)
            vmax = np.percentile(img, 98)
            colormap = cv2.COLORMAP_JET
        else:
            vmin = np.percentile(img, 1)
            vmax = np.percentile(img, 99.5)
            colormap = cv2.COLORMAP_VIRIDIS  # por defecto

        # Escalado y coloreado
        img_clip = np.clip(img, vmin, vmax)
        norm = cv2.normalize(img_clip, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        colored = cv2.applyColorMap(norm, colormap)
        colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

        # Visualización
        st.image(colored_rgb, caption=f"Visualización con {tipo_imagen}", use_column_width=True)

        # Preparar descarga como PNG
        nombre_base = archivo_dicom.name.split(".")[0]
        nombre_salida = f"{nombre_base}_color.png"
        buffer = BytesIO()
        _, img_encoded = cv2.imencode(".png", colored_rgb)
        buffer.write(img_encoded)

        st.download_button(
            label="⬇️ Descargar imagen coloreada como PNG",
            data=buffer.getvalue(),
            file_name=nombre_salida,
            mime="image/png"
        )

    except Exception as e:
        st.error(f"❌ Error general al procesar el archivo DICOM: {e}")
