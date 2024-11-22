const predictClassification = require('../services/inferenceService');
const storeData = require('../services/storeData');
const crypto = require('crypto');

async function postPredictHandler(request, h) {
//   console.log(request.payload);
  const { image } = request.payload;
  const { model } = request.server.app;

  if (!image) {
    return h.response({
      status: 'fail',
      message: 'Image not provided',
    }).code(400);
  }

  try {
    const { confidenceScore, label, suggestion } = await predictClassification(model, image);
    const id = crypto.randomUUID();
    const createdAt = new Date().toISOString();

    const data = {
      id,
      result: label,
      suggestion,
      confidenceScore,
      createdAt,
    };

    // console.log("Saving to Firestore:", data);
    await storeData(id, data);

    const response = h.response({
      status: 'success',
      message: 'Model is predicted successfully',
      data,
    });
    response.code(201);
    return response;
  } catch (error) {
    // console.error("Error in postPredictHandler:", error);
    return h.response({
      status: 'fail',
      message: 'Terjadi kesalahan dalam melakukan prediksi',
    }).code(400);
  }
}

module.exports = postPredictHandler;
