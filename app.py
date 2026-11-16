import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from PIL import Image

st.set_page_config(page_title="Zero Waste Vision", page_icon="🍎", layout="centered")

st.title("🍎 Zero Waste Vision")
st.write("Detektor kesegaran buah berbasis AI untuk mengurangi limbah makanan.")

@st.cache_resource
def load_ai_model():
    return tf.keras.models.load_model("model_buah.h5", compile=False)

model = load_ai_model()

# Urutan label sesuai abjad folder di Colab
CLASS_NAMES = ["consume_soon", "fresh", "spoiled"]
BATAS_KEYAKINAN = 75.0  # Minimal keyakinan 75% agar dianggap buah yang valid

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
    st.image(image, caption="Foto yang Dianalisis", width=350)

    # Preprocessing pakai rumus bawaan MobileNetV2 (sama dengan di Colab)
    img_resized = image.resize((224, 224), Image.Resampling.BILINEAR)
    img_array = np.array(img_resized, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    with st.spinner("AI sedang menganalisis gambar..."):
        predictions = model.predict(img_array)[0]
        index_tertinggi = int(np.argmax(predictions))
        hasil_kelas = CLASS_NAMES[index_tertinggi]
        tingkat_keyakinan = float(predictions[index_tertinggi]) * 100

    st.subheader("Hasil Deteksi AI:")

    # Filter untuk menolak benda random atau gambar yang ragu-ragu
    if tingkat_keyakinan < BATAS_KEYAKINAN:
        st.warning(f"⚠️ **Objek Tidak Dikenali / Gambar Kurang Jelas** (Keyakinan tertinggi hanya {tingkat_keyakinan:.1f}%)")
        st.info(
            "💡 **Tips Pemindaian:** Pastikan objek adalah **1 buah utuh**, pencahayaan terang, "
            "dan arahkan kamera lebih dekat ke bagian permukaan buah."
        )
    else:
        if hasil_kelas == "fresh":
            st.success(f"**FRESH (Segar)** — Keyakinan: {tingkat_keyakinan:.1f}%")
            st.info("💡 **Saran Zero Waste:** Buah masih sangat segar. Simpan di tempat sejuk dan kering agar awet lebih lama.")
        elif hasil_kelas == "consume_soon":
            st.warning(f"**CONSUME SOON (Segera Konsumsi)** — Keyakinan: {tingkat_keyakinan:.1f}%")
            st.info("💡 **Saran Zero Waste:** Buah sudah mulai matang. Segera makan hari ini atau olah menjadi jus, smoothie, maupun kue agar tidak terbuang.")
        else:
            st.error(f"**SPOILED (Busuk)** — Keyakinan: {tingkat_keyakinan:.1f}%")
            st.info("💡 **Saran Zero Waste:** Buah sudah tidak layak konsumsi. Pisahkan dari buah lain dan olah menjadi pupuk kompos atau eco-enzyme.")

    # Tampilkan rincian skor semua kategori
    with st.expander("📊 Lihat Rincian Skor Prediksi AI"):
        label_tampil = {
            "fresh": "Fresh (Segar)",
            "consume_soon": "Consume Soon (Segera Konsumsi)",
            "spoiled": "Spoiled (Busuk)"
        }
        for i, nama_kelas in enumerate(CLASS_NAMES):
            skor = float(predictions[i])
            st.write(f"**{label_tampil[nama_kelas]}:** {skor * 100:.1f}%")
            st.progress(skor)