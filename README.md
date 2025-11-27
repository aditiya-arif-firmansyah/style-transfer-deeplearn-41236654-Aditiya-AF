# style-transfer-deeplearn-41236654-Aditiya-A
# ==============================================================================
# TUGAS INDIVIDU: IMPLEMENTASI Natural STYLE TRANSFER (NST)
# Model ini menggabungkan konten dari 'images.jpg' dengan gaya dari 'images.png'
# Menggunakan TensorFlow dan Keras, sesuai materi perkuliahan.
# ==============================================================================

import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import time

drive.mount('/content/drive')

# --- 1. Definisi Jalur dan Parameter ---
# Ubah path ini jika Anda menjalankan di Colab dan menggunakan file upload
# PATH BARU: Mengarah ke Google Drive Anda
CONTENT_PATH = '/content/drive/MyDrive/deeplearn/kucing.jpg' # Gambar Konten (Kucing)
STYLE_PATH = '/content/drive/MyDrive/deeplearn/batik1.png'   # Gambar Gaya (Batik Mega Mendung)
OUTPUT_FILENAME = '/content/hasil_style_transfer.png' # Diubah ke jalur absolut
IMG_SIZE = 512 # Ukuran gambar output (dapat diubah)

# Bobot untuk fungsi loss
CONTENT_WEIGHT = 1e-2
STYLE_WEIGHT = 1e4
TOTAL_VARIATION_WEIGHT = 30 # Untuk menghaluskan hasil

# Jumlah iterasi untuk optimasi (L-BFGS seringkali lebih cepat,
# tetapi kita pakai Gradient Descent untuk kompatibilitas Keras standar)
ITERATIONS = 100

# --- 2. Fungsi Bantuan untuk Pra-pemrosesan dan Pemuatan Gambar ---

def preprocess_image(image_path, target_size):
    """Memuat, mengubah ukuran, dan pra-memproses gambar untuk VGG19."""
    # Memuat gambar dalam mode RGB
    img = load_img(image_path, target_size=(target_size, target_size))
    # Mengubah ke array numpy
    img = img_to_array(img)
    # Menambahkan dimensi batch (1, H, W, C)
    img = np.expand_dims(img, axis=0)
    # Pra-pemrosesan VGG19 (mengubah BGR, menormalisasi)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return tf.convert_to_tensor(img)

def deprocess_image(x):
    """Mengubah tensor hasil kembali menjadi gambar yang dapat dilihat."""
    # Menghapus dimensi batch
    x = x.numpy()[0]
    # Membalikkan pra-pemrosesan VGG19
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # Konversi BGR ke RGB
    x = x[:, :, ::-1]
    # Klip nilai piksel
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# --- 3. Implementasi Fungsi Loss ---

# Layers yang akan digunakan untuk Content Loss
CONTENT_LAYERS = ['block5_conv2']

# Layers yang akan digunakan untuk Style Loss
STYLE_LAYERS = [
    'block1_conv1',
    'block2_conv1',
    'block3_conv1',
    'block4_conv1',
    'block5_conv1'
]

def gram_matrix(input_tensor):
    """Menghitung Gram Matrix (korelasi antar fitur)."""
    # tf.einsum digunakan untuk perkalian matriks yang lebih efisien:
    # 'b' batch, 'c' channels, 'x' dan 'y' dimensi spasial
    result = tf.einsum('bxyc,bxyk->bck', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    # Normalisasi dengan membagi dengan jumlah elemen
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations

def get_style_loss(base_style_feature_map, target_gram_matrix_precomputed):
    """Menghitung Style Loss (MSE dari Gram Matrix)."""
    # base_style_feature_map adalah feature map 4D dari gambar yang sedang dioptimasi
    base_gram = gram_matrix(base_style_feature_map)
    # target_gram_matrix_precomputed sudah merupakan Gram matrix (3D)
    return tf.reduce_mean(tf.square(target_gram_matrix_precomputed - base_gram))

def get_content_loss(base_content, target_content):
    """Menghitung Content Loss (MSE dari aktivasi layer)."""
    return tf.reduce_mean(tf.square(base_content - target_content))

def total_variation_loss(x):
    """Menghitung Total Variation Loss untuk menghaluskan hasil."""
    a = tf.square(
        x[:, :IMG_SIZE - 1, :IMG_SIZE - 1, :] - x[:, 1:, :IMG_SIZE - 1, :]
    )
    b = tf.square(
        x[:, :IMG_SIZE - 1, :IMG_SIZE - 1, :] - x[:, :IMG_SIZE - 1, 1:, :]
    )
    return tf.reduce_sum(a + b)

# --- 4. Model dan Ekstraksi Fitur ---

def get_model():
    """Memuat VGG19 dan membuat model ekstraktor fitur."""
    # Memuat VGG19 dengan bobot ImageNet, tidak menyertakan layer fully connected
    vgg = VGG19(weights='imagenet', include_top=False)
    vgg.trainable = False # Model tidak perlu dilatih ulang

    # Ekstraksi output dari layers yang dipilih
    style_outputs = [vgg.get_layer(name).output for name in STYLE_LAYERS]
    content_outputs = [vgg.get_layer(name).output for name in CONTENT_LAYERS]

    # Gabungkan output menjadi satu model
    model_outputs = style_outputs + content_outputs
    return tf.keras.Model(vgg.input, model_outputs)

def get_feature_representations(model, content_path, style_path):
    """Mendapatkan fitur konten dan gaya dari gambar input."""
    # Muat dan pra-proses gambar
    content_image = preprocess_image(content_path, IMG_SIZE)
    style_image = preprocess_image(style_path, IMG_SIZE)

    # Ekstraksi fitur
    style_features = model(style_image)
    content_features = model(content_image)

    # Pisahkan output menjadi fitur gaya dan konten
    style_dict = {
        name: value for name, value in zip(STYLE_LAYERS, style_features[:len(STYLE_LAYERS)])
    }
    content_dict = {
        name: value for name, value in zip(CONTENT_LAYERS, content_features[len(STYLE_LAYERS):])
    }
    return style_dict, content_dict

# --- 5. Fungsi Total Loss dan Optimasi ---

def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):
    """Menghitung total loss (gaya, konten, dan total variasi)."""
    style_weight, content_weight, total_variation_weight = loss_weights

    # Ekstraksi fitur dari gambar yang sedang dioptimasi (init_image)
    model_outputs = model(init_image)
    style_output_features = model_outputs[:len(STYLE_LAYERS)]
    content_output_features = model_outputs[len(STYLE_LAYERS):]

    style_score = 0
    content_score = 0

    # 1. Style Loss
    # target_gram_matrix adalah Gram matrix yang sudah dihitung sebelumnya dari gambar gaya target
    # comb_feature_map adalah feature map 4D dari gambar yang sedang dioptimasi
    for target_gram_matrix, comb_feature_map in zip(gram_style_features, style_output_features):
        style_score += get_style_loss(comb_feature_map, target_gram_matrix)
    style_score *= style_weight

    # 2. Content Loss
    for target_content, comb_content in zip(content_features, content_output_features):
        content_score += get_content_loss(comb_content, target_content)
    content_score *= content_weight

    # 3. Total Variation Loss
    tv_loss = total_variation_loss(init_image) * total_variation_weight

    # Total Loss
    loss = style_score + content_score + tv_loss
    return loss, style_score, content_score, tv_loss

# Menggunakan tf.function untuk komputasi gradient yang lebih cepat
@tf.function
def compute_grads(cfg):
    """Menghitung gradient dari total loss."""
    with tf.GradientTape() as tape:
        all_loss = compute_loss(**cfg)
    # Gradient dihitung terhadap tensor gambar hasil (init_image)
    total_loss = all_loss[0]
    return tape.gradient(total_loss, cfg['init_image']), all_loss

# --- 6. Fungsi Utama Implementasi NST ---

def run_style_transfer(content_path, style_path, num_iterations=ITERATIONS):
    """Fungsi utama untuk menjalankan proses Style Transfer."""
    print("Memulai proses Neural Style Transfer...")

    # Inisialisasi model VGG19
    model = get_model()

    # Mendapatkan fitur target (gaya dan konten)
    style_features, content_features = get_feature_representations(
        model, content_path, style_path
    )

    # Pre-calculate Gram Matrices untuk fitur gaya (target)
    gram_style_features = [gram_matrix(style_feature) for style_feature in style_features.values()]

    # Gambar awal (dimulai dari gambar konten)
    init_image = preprocess_image(content_path, IMG_SIZE)
    # Membuat tensor variabel agar dapat dioptimasi
    init_image = tf.Variable(init_image, dtype=tf.float32)

    # Optimizer (Adam, bukan L-BFGS, karena lebih umum di Keras)
    optimizer = tf.optimizers.Adam(learning_rate=5.0, beta_1=0.99, epsilon=1e-1)

    # Konfigurasi untuk fungsi komputasi
    cfg = {
        'model': model,
        'loss_weights': (STYLE_WEIGHT, CONTENT_WEIGHT, TOTAL_VARIATION_WEIGHT),
        'init_image': init_image,
        'gram_style_features': gram_style_features,
        'content_features': list(content_features.values()) # <--- FIX: Pass the values directly
    }

    best_loss, best_img = float('inf'), None
    start_time = time.time()

    print(f"Mengoptimasi selama {num_iterations} iterasi...")

    for i in range(1, num_iterations + 1):
        # Hitung gradien dan loss
        grads, all_loss = compute_grads(cfg)
        loss, style_score, content_score, tv_loss = all_loss

        # Terapkan gradien
        optimizer.apply_gradients([(grads, init_image)])

        # Klip nilai agar tetap dalam rentang yang valid (0-255)
        # Langkah ini penting setelah de-preprocessing untuk menghindari artefak
        clipped = tf.clip_by_value(init_image, -200.0, 600.0)
        init_image.assign(clipped)

        if loss < best_loss:
            # Simpan gambar terbaik sejauh ini
            best_loss = loss
            best_img = deprocess_image(init_image)

        if i % 10 == 0:
            print(f"Iterasi {i}/{num_iterations} | Total Loss: {loss.numpy():.2f} | Content: {content_score.numpy():.2f} | Style: {style_score.numpy():.2f}")

    end_time = time.time()
    print(f"\nOptimasi Selesai dalam {end_time - start_time:.2f} detik.")

    # Simpan hasil terbaik
    if best_img is not None:
        final_img = tf.keras.preprocessing.image.array_to_img(best_img)
        final_img.save(OUTPUT_FILENAME)
        print(f"Gambar hasil style transfer telah disimpan sebagai: {OUTPUT_FILENAME}")
    from google.colab import drive
    
    return final_img
    

# Jalankan fungsi utama Neural Style Transfer
# Ini akan menyimpan gambar hasil ke OUTPUT_FILENAME ('hasil_style_transfer.png')
run_style_transfer(CONTENT_PATH, STYLE_PATH)

    

