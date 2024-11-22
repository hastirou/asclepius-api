const tf = require("@tensorflow/tfjs-node");
const InputError = require("../exceptions/InputError");

async function predictClassification(model, image) {
  try {
    // console.log("Image buffer length:", image.length);

    // Decode and preprocess the image
    const tensor = tf.node
      .decodeImage(image)
      .resizeNearestNeighbor([224, 224])
      .expandDims()
      .toFloat();

    if (!tensor) {
      console.error("Failed to decode image.");
      throw new InputError("Invalid image format.");
    }

    // console.log("Tensor shape:", tensor.shape);

    if (!model) {
      console.error("Model not loaded properly.");
      throw new InputError("Model is not available.");
    }

    // Predict using the resized tensor
    const prediction = model.predict(tensor);
    const scores = await prediction.data();
    // console.log("Prediction scores:", scores);

    const confidenceScore = Math.max(...scores) * 100;
    // console.log("Confidence score:", confidenceScore);

    const label = confidenceScore > 50 ? "Cancer" : "Non-cancer";
    let suggestion;
    if (label === "Cancer") {
      suggestion = "Segera periksa ke dokter!";
    } else {
      suggestion = "Penyakit kanker tidak terdeteksi.";
    }

    return { confidenceScore, label, suggestion };

  } catch (error) {
    console.error("Prediction error:", error);
    throw new InputError("Terjadi kesalahan dalam melakukan prediksi");
  }
}

module.exports = predictClassification;
