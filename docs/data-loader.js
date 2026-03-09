/********************************************************************
 DATA LOADER FOR BROWSER-ONLY MNIST CSV FILES

 Responsibilities:
 - Parse uploaded CSV files
 - Convert to TensorFlow tensors
 - Normalize pixel values
 - Provide helper utilities for:
      * validation split
      * random batch selection
      * adding noise
      * drawing images to canvas
      * calculating reconstruction error

 CSV Format:
 label, pixel0, pixel1, ... pixel783
********************************************************************/

const IMAGE_SIZE = 28 * 28;
const NUM_CLASSES = 10;

/********************************************************************
 CSV PARSER
 Reads text file and converts into numeric arrays.
********************************************************************/
async function parseCSV(file){

  const text = await file.text();

  const lines = text.split(/\r?\n/);

  const images = [];
  const labels = [];

  for(let line of lines){

    if(!line.trim()) continue;

    const values = line.split(",");

    if(values.length !== 785) continue;

    const label = parseInt(values[0]);

    const pixels = values
      .slice(1)
      .map(v => Number(v)/255); // normalize to [0,1]

    labels.push(label);
    images.push(pixels);

  }

  const xs = tf.tensor2d(images)
        .reshape([images.length,28,28,1]);

  const ys = tf.oneHot(
      tf.tensor1d(labels,"int32"),
      NUM_CLASSES
  );

  return {xs, ys};
}

/********************************************************************
 LOAD TRAIN DATA
********************************************************************/
async function loadTrainFromFiles(file){

  const data = await parseCSV(file);
  return data;

}

/********************************************************************
 LOAD TEST DATA
********************************************************************/
async function loadTestFromFiles(file){

  const data = await parseCSV(file);
  return data;

}

/********************************************************************
 SPLIT TRAIN / VALIDATION
********************************************************************/
function splitTrainVal(xs, ys, valRatio = 0.1){

  const num = xs.shape[0];
  const valSize = Math.floor(num * valRatio);

  const trainXs = xs.slice([0,0,0,0],[num-valSize,28,28,1]);
  const valXs = xs.slice([num-valSize,0,0,0],[valSize,28,28,1]);

  const trainYs = ys.slice([0,0],[num-valSize,10]);
  const valYs = ys.slice([num-valSize,0],[valSize,10]);

  return {trainXs,trainYs,valXs,valYs};

}

/********************************************************************
 GET RANDOM TEST BATCH
 Used for preview/testing UI
********************************************************************/
function getRandomTestBatch(xs, ys, k = 5){

  const total = xs.shape[0];

  const indices = [];

  for(let i=0;i<k;i++){
    indices.push(Math.floor(Math.random()*total));
  }

  const xsBatch = tf.stack(
    indices.map(i=>xs.slice([i,0,0,0],[1,28,28,1]).squeeze())
  );

  const ysBatch = tf.stack(
    indices.map(i=>ys.slice([i,0],[1,10]).squeeze())
  );

  return {xsBatch, ysBatch};

}

/********************************************************************
 ADD RANDOM GAUSSIAN NOISE
 Used for denoising autoencoder training/testing.

 noiseFactor controls noise strength.
********************************************************************/
function addRandomNoise(tensor, noiseFactor = 0.3){

  return tf.tidy(()=>{

    const noise = tf.randomNormal(tensor.shape);

    const noisy = tensor.add(noise.mul(noiseFactor));

    // clamp values between 0 and 1
    return noisy.clipByValue(0,1);

  });

}

/********************************************************************
 DRAW 28x28 TENSOR TO CANVAS
********************************************************************/
function draw28x28ToCanvas(tensor, canvas, scale = 4){

  const [h,w] = [28,28];

  canvas.width = w*scale;
  canvas.height = h*scale;

  const ctx = canvas.getContext("2d");

  const data = tensor.dataSync();

  const img = ctx.createImageData(w,h);

  for(let i=0;i<data.length;i++){

    const val = data[i]*255;

    img.data[i*4+0]=val;
    img.data[i*4+1]=val;
    img.data[i*4+2]=val;
    img.data[i*4+3]=255;

  }

  const tmp = document.createElement("canvas");
  tmp.width=w;
  tmp.height=h;
  tmp.getContext("2d").putImageData(img,0,0);

  ctx.imageSmoothingEnabled=false;
  ctx.drawImage(tmp,0,0,w*scale,h*scale);

}

/********************************************************************
 CALCULATE MSE
 Used for evaluating denoising quality.
********************************************************************/
function calculateMSE(original, reconstructed){

  return tf.tidy(()=>{

    const mse = tf.losses.meanSquaredError(original,reconstructed);

    const v = mse.mean().dataSync()[0];

    return v;
   /********************************************************************
 EXPORT FUNCTIONS TO GLOBAL WINDOW
 This allows app.js to access the functions defined in this file.
********************************************************************/

window.loadTrainFromFiles = loadTrainFromFiles;
window.loadTestFromFiles = loadTestFromFiles;
window.splitTrainVal = splitTrainVal;
window.getRandomTestBatch = getRandomTestBatch;
window.addRandomNoise = addRandomNoise;
window.draw28x28ToCanvas = draw28x28ToCanvas;
window.calculateMSE = calculateMSE;

  });

}
