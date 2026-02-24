/**
 * Neural Network Design: The Gradient Puzzle
 *
 * Objective:
 * Modify the Student Model architecture and loss function to transform
 * random noise input into a smooth, directional gradient output.
 */

// ==========================================
// 1. Global State & Config
// ==========================================
const CONFIG = {
  // Model definition shape (no batch dim) - used for layer creation
  inputShapeModel: [16, 16, 1],
  // Data tensor shape (includes batch dim) - used for input tensor creation
  inputShapeData: [1, 16, 16, 1],
  learningRate: 0.05,
  autoTrainSpeed: 50, // ms delay between steps (lower is faster)
};

let state = {
  step: 0,
  isAutoTraining: false,
  autoTrainInterval: null,
  xInput: null, 
  baselineModel: null,
  studentModel: null,
  baselineOptimizer: null,
  studentOptimizer: null,
};

// Standard MSE: Mean Squared Error
function mse(yTrue, yPred) {
  return tf.losses.meanSquaredError(yTrue, yPred);
}

// TODO: Helper - Smoothness (Total Variation)
function smoothness(yPred) {
  return tf.tidy(() => {
    // Просто считаем разницу между соседними пикселями
    const diffX = yPred.slice([0, 0, 0, 0], [1, 16, 15, 1])
      .sub(yPred.slice([0, 0, 1, 0], [1, 16, 15, 1]));
    
    const diffY = yPred.slice([0, 0, 0, 0], [1, 15, 16, 1])
      .sub(yPred.slice([0, 1, 0, 0], [1, 15, 16, 1]));
    
    // L1 норма работает лучше для гладкости
    return tf.mean(tf.abs(diffX)).add(tf.mean(tf.abs(diffY))).mul(20);
  });
}

// TODO: Helper - Directionality (Gradient)
function directionX(yPred) {
  return tf.tidy(() => {
    const height = 16;
    const width = 16;
    
    // Целевой градиент от темного к светлому
    const target = tf.linspace(0.1, 0.9, width)
      .reshape([1, 1, width, 1])
      .tile([1, height, 1, 1]);
    
    // Просто MSE с целевым градиентом
    return tf.losses.meanSquaredError(target, yPred).mul(100);
  });
}

// ==========================================
// 3. Model Architecture
// ==========================================

// Baseline Model: Fixed Compression (Undercomplete AE)
// 16x16 -> 64 -> 16x16
function createBaselineModel() {
  const model = tf.sequential();
  model.add(tf.layers.flatten({ inputShape: CONFIG.inputShapeModel }));
  model.add(tf.layers.dense({ units: 64, activation: "relu" })); // Bottleneck
  model.add(tf.layers.dense({ units: 256, activation: "sigmoid" })); // Output 0-1
  // Reshape back to [16, 16, 1] (batch dim is handled automatically)
  model.add(tf.layers.reshape({ targetShape: [16, 16, 1] }));
  return model;
}

/// ------------------------------------------------------------------
// [TODO-A]: STUDENT ARCHITECTURE DESIGN 
// Modify this function to implement 'transformation' and 'expansion'.
// ------------------------------------------------------------------
function createStudentModel(archType) {
  const model = tf.sequential();
  model.add(tf.layers.flatten({ inputShape: CONFIG.inputShapeModel }));

  if (archType === "compression") {
    // [Implemented] Bottleneck: Compress information
    model.add(tf.layers.dense({ units: 64, activation: "relu" }));
    model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));
  } else if (archType === "transformation") {
    // Implement Transformation (1:1 mapping)
    // Сохраняем размерность 256 (равную входной 16x16=256)
    model.add(tf.layers.dense({ 
      units: 256, 
      activation: "relu",
      name: "transform_hidden"
    }));
    
    // Выходной слой с сигмоидой для нормализации в [0,1]
    model.add(tf.layers.dense({ 
      units: 256, 
      activation: "sigmoid",
      name: "transform_output"
    }));

  } else if (archType === "expansion") {
    // Implement Expansion (Overcomplete)
    // Увеличиваем размерность для более сложных трансформаций
    model.add(tf.layers.dense({ 
      units: 512, 
      activation: "relu",
      name: "expand_hidden_1"
    }));
    
    // Промежуточный слой для обработки
    model.add(tf.layers.dense({ 
      units: 384, 
      activation: "relu",
      name: "expand_hidden_2"
    }));
    
    // Возвращаемся к исходной размерности
    model.add(tf.layers.dense({ 
      units: 256, 
      activation: "sigmoid",
      name: "expand_output"
    }));
    
  } else {
    // Safety check for unknown architectures
    throw new Error(`Unknown architecture type: ${archType}`);
  }

  model.add(tf.layers.reshape({ targetShape: [16, 16, 1] }));
  return model;
}

// ==========================================
// 4. Custom Loss Function
// ==========================================

// ------------------------------------------------------------------
// [TODO-B]: STUDENT LOSS DESIGN 
// Modify this function to create a smooth gradient.
// ------------------------------------------------------------------
function studentLoss(yTrue, yPred) {
  return tf.tidy(() => {
    // 1. Basic Reconstruction (MSE) - минимальное влияние, чтобы сохранить цвета
    // Уменьшаем вес, чтобы модель не копировала шум
    const lossMSE = mse(yTrue, yPred).mul(0.01);

    // 2. Smoothness - "Be smooth locally"
    // Заставляем соседние пиксели быть похожими (убираем шум)
    const lossSmooth = smoothness(yPred).mul(0.5); // Вес 0.5

    // 3. Direction - "Be bright on the right"
    // Создаем градиент слева направо (главная цель)
    const lossDir = directionX(yPred).mul(1.0); // Вес 1.0

    // Total Loss - комбинация всех компонентов
    return lossMSE.add(lossSmooth).add(lossDir);
  });
}

// ==========================================
// 5. Training Loop 
// ==========================================

async function trainStep() {
  state.step++;

  // Safety check: Ensure models are initialized
  if (!state.studentModel || !state.studentModel.getWeights) {
    log("Error: Student model not initialized properly.", true);
    stopAutoTrain();
    return;
  }

  // Train Baseline (MSE Only) - используем baselineOptimizer
  const baselineLossVal = tf.tidy(() => {
    const { value, grads } = tf.variableGrads(() => {
      const yPred = state.baselineModel.predict(state.xInput);
      return mse(state.xInput, yPred); // Baseline always uses MSE
    }, state.baselineModel.getWeights());

    state.baselineOptimizer.applyGradients(grads); // [ИСПРАВЛЕНО] Свой оптимизатор
    return value.dataSync()[0];
  });

  // Train Student (Custom Loss) - используем studentOptimizer
  let studentLossVal = 0;
  try {
    studentLossVal = tf.tidy(() => {
      const { value, grads } = tf.variableGrads(() => {
        const yPred = state.studentModel.predict(state.xInput);
        return studentLoss(state.xInput, yPred); // Uses student's custom loss
      }, state.studentModel.getWeights());

      state.studentOptimizer.applyGradients(grads); // [ИСПРАВЛЕНО] Свой оптимизатор
      return value.dataSync()[0];
    });
    log(
      `Step ${state.step}: Base Loss=${baselineLossVal.toFixed(4)} | Student Loss=${studentLossVal.toFixed(4)}`,
    );
  } catch (e) {
    log(`Error in Student Training: ${e.message}`, true);
    stopAutoTrain();
    return;
  }

  // Visualize
  if (state.step % 5 === 0 || !state.isAutoTraining) {
    await render();
    updateLossDisplay(baselineLossVal, studentLossVal);
  }
}

// ==========================================
// 6. UI & Initialization logic
// ==========================================

function init() {
  // 1. Generate fixed noise (Batch size included: [1, 16, 16, 1])
  state.xInput = tf.randomUniform(CONFIG.inputShapeData);

  // 2. Initialize Models
  resetModels();

  // 3. Render Initial Input
  tf.browser.toPixels(
    state.xInput.squeeze(),
    document.getElementById("canvas-input"),
  );

  // 4. Bind Events
  document
    .getElementById("btn-train")
    .addEventListener("click", () => trainStep());
  document
    .getElementById("btn-auto")
    .addEventListener("click", toggleAutoTrain);
  document.getElementById("btn-reset").addEventListener("click", resetModels);

  document.querySelectorAll('input[name="arch"]').forEach((radio) => {
    radio.addEventListener("change", (e) => {
      resetModels(e.target.value);
      document.getElementById("student-arch-label").innerText =
        e.target.value.charAt(0).toUpperCase() + e.target.value.slice(1);
    });
  });

  log("Initialized. Ready to train.");
}

function resetModels(archType = null) {
  // [Fix]: When called via event listener, archType is an Event object.
  // We must ensure it's either a string or null.
  if (typeof archType !== "string") {
    archType = null;
  }

  // Safety: Stop auto-training to prevent race conditions during reset
  if (state.isAutoTraining) {
    stopAutoTrain();
  }

  if (!archType) {
    const checked = document.querySelector('input[name="arch"]:checked');
    archType = checked ? checked.value : "compression";
  }

  // Dispose old resources to avoid memory leaks
  if (state.baselineModel) {
    state.baselineModel.dispose();
    state.baselineModel = null;
  }
  if (state.studentModel) {
    state.studentModel.dispose();
    state.studentModel = null;
  }
  // Dispose обоих оптимизаторов
  if (state.baselineOptimizer) {
    state.baselineOptimizer.dispose();
    state.baselineOptimizer = null;
  }
  if (state.studentOptimizer) {
    state.studentOptimizer.dispose();
    state.studentOptimizer = null;
  }

  // Create New Models
  state.baselineModel = createBaselineModel();
  try {
    state.studentModel = createStudentModel(archType);
  } catch (e) {
    log(`Error creating model: ${e.message}`, true);
    state.studentModel = createBaselineModel(); // Fallback to avoid crash
  }

  //  Создаем два отдельных оптимизатора
  state.baselineOptimizer = tf.train.adam(CONFIG.learningRate);
  state.studentOptimizer = tf.train.adam(CONFIG.learningRate);
  state.step = 0;

  log(`Models reset. Student Arch: ${archType}`);
  render();
}

async function render() {
  // Tensor memory management with tidy not possible here due to async toPixels,
  // so we manually dispose predictions.
  const basePred = state.baselineModel.predict(state.xInput);
  const studPred = state.studentModel.predict(state.xInput);

  await tf.browser.toPixels(
    basePred.squeeze(),
    document.getElementById("canvas-baseline"),
  );
  await tf.browser.toPixels(
    studPred.squeeze(),
    document.getElementById("canvas-student"),
  );

  basePred.dispose();
  studPred.dispose();
}

// UI Helpers
function updateLossDisplay(base, stud) {
  document.getElementById("loss-baseline").innerText =
    `Loss: ${base.toFixed(5)}`;
  document.getElementById("loss-student").innerText =
    `Loss: ${stud.toFixed(5)}`;
}

function log(msg, isError = false) {
  const el = document.getElementById("log-area");
  const span = document.createElement("div");
  span.innerText = `> ${msg}`;
  if (isError) span.classList.add("error");
  el.prepend(span);
}

// Auto Train Logic
function toggleAutoTrain() {
  const btn = document.getElementById("btn-auto");
  if (state.isAutoTraining) {
    stopAutoTrain();
  } else {
    state.isAutoTraining = true;
    btn.innerText = "Auto Train (Stop)";
    btn.classList.add("btn-stop");
    btn.classList.remove("btn-auto");
    loop();
  }
}

function stopAutoTrain() {
  state.isAutoTraining = false;
  const btn = document.getElementById("btn-auto");
  btn.innerText = "Auto Train (Start)";
  btn.classList.add("btn-auto");
  btn.classList.remove("btn-stop");
}

function loop() {
  if (state.isAutoTraining) {
    trainStep();
    setTimeout(loop, CONFIG.autoTrainSpeed);
  }
}

// Start
init();
