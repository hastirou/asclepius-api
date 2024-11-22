const tf = require('@tensorflow/tfjs-node');

const MODEL_URL = 'https://storage.googleapis.com/ml-model-storage-assesment/submissions-model/model.json';

async function loadModel() {
    try {
        const model = await tf.loadGraphModel(MODEL_URL);
        console.log("Model loaded successfully.");
        return model;
    } catch (error) {
        console.error("Error loading the model:", error);
        throw new Error("Model loading failed.");
    }
}

module.exports = loadModel;
