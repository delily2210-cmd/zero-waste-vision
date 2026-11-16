import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Zero Waste Vision", page_icon="🍎", layout="centered")

st.title("🍎 Zero Waste Vision")
st.write("Detektor kesegaran buah berbasis AI untuk mengurangi limbah makanan.")

@st.cache_resource
def load_ai_model():
    return tf.keras.models.load_model("model_buah.h5", compile=False)

model = load_ai_model()

# Urutan label wajib sama persis dengan hasil training di Colab
CLASS_NAMES = ["consume_soon", "fresh", "spoiled"]

pilihan = st.radio(
    "Pilih cara memasukkan foto buah:",
    ["Upload Foto", "Gunakan Kamera"],
    horizontal=True
)

if pilihan == "Upload Foto":
    gambar_input = st.file_uploader("Pilih foto buah (JPG/PNG)", type=["jpg", "jpeg", "png"])
else:
    gambar_input = st.camera_input("Ambil foto buah")

if gambar_input is not None:
    image = Image.open(gambar_input).convert("RGB")
    st.image(image, caption="Foto Buah yang Dianalisis", width=350)

    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("AI sedang menganalisis kesegaran buah..."):
        predictions = model.predict(img_array)
        index_tertinggi = np.argmax(predictions[0])
        hasil_kelas = CLASS_NAMES[index_tertinggi]
        tingkat_keyakinan = float(predictions[0][index_tertinggi]) * 100

    st.subheader("Hasil Deteksi AI:")

    if hasil_kelas == "fresh":
        st.success(f"**FRESH (Segar)** — Keyakinan: {tingkat_keyakinan:.1f}%")
        st.info("💡 **Saran Zero Waste:** Buah masih sangat segar. Simpan di tempat sejuk dan kering agar awet lebih lama.")
    elif hasil_kelas == "consume_soon":
        st.warning(f"**CONSUME SOON (Segera Konsumsi)** — Keyakinan: {tingkat_keyakinan:.1f}%")
        st.info("💡 **Saran Zero Waste:** Buah sudah mulai matang. Segera makan hari ini atau olah menjadi jus, smoothie, maupun kue agar tidak terbuang.")
    else:
        st.error(f"**SPOILED (Busuk)** — Keyakinan: {tingkat_keyakinan:.1f}%")
        st.info("💡 **Saran Zero Waste:** Buah sudah tidak layak konsumsi. Pisahkan dari buah lain dan olah menjadi pupuk kompos atau eco-enzyme.")