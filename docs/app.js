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
  xInput: null, // The fixed noise input
  baselineModel: null,
  studentModel: null,
  optimizer: null,
};

// ==========================================
// 2. Helper Functions (Loss Components)
// ==========================================

// Standard MSE: Mean Squared Error
function mse(yTrue, yPred) {
  return tf.losses.meanSquaredError(yTrue, yPred);
}

// Smoothness (Total Variation) - Penalize differences between adjacent pixels
function smoothness(yPred) {
  // Difference in X direction: pixel[i, j] - pixel[i, j+1]
  const diffX = yPred
    .slice([0, 0, 0, 0], [-1, -1, 15, -1])
    .sub(yPred.slice([0, 0, 1, 0], [-1, -1, 15, -1]));

  // Difference in Y direction: pixel[i, j] - pixel[i+1, j]
  const diffY = yPred
    .slice([0, 0, 0, 0], [-1, 15, -1, -1])
    .sub(yPred.slice([0, 1, 0, 0], [-1, 15, -1, -1]));

  // Return mean of absolute differences (L1 norm - лучше для резких границ)
  return tf.mean(tf.abs(diffX)).add(tf.mean(tf.abs(diffY)));
}

// Directionality (Gradient) - Encourage pixels on the right to be brighter
function directionX(yPred) {
  // Create a weight mask that increases from left (-1) to right (+1)
  const width = 16;
  const mask = tf.linspace(-1, 1, width).reshape([1, 1, width, 1]); // [1, 1, 16, 1]
  
  // We want yPred to correlate with mask (bright on right)
  // Maximize correlation => Minimize negative correlation
  return tf.mean(yPred.mul(mask)).mul(-1);
}

// Directionality (Vertical) - Encourage pixels on the bottom to be brighter
function directionY(yPred) {
  // Create a weight mask that increases from top (-1) to bottom (+1)
  const height = 16;
  const mask = tf.linspace(-1, 1, height).reshape([1, height, 1, 1]); // [1, 16, 1, 1]
  
  return tf.mean(yPred.mul(mask)).mul(-1);
}

// [НОВАЯ ФУНКЦИЯ] Сохранение гистограммы цветов
// Штрафует за изменение распределения яркости пикселей
function preserveHistogram(yTrue, yPred) {
  return tf.tidy(() => {
    // Сортируем пиксели в обоих изображениях
    const sortedTrue = yTrue.reshape([-1]).sort();
    const sortedPred = yPred.reshape([-1]).sort();
    
    // Штрафуем за разницу в распределении (MSE между отсортированными значениями)
    return tf.losses.meanSquaredError(sortedTrue, sortedPred);
  });
}

// [НОВАЯ ФУНКЦИЯ] Проверка на создание новых цветов
// Строгий штраф за выход за пределы исходной гистограммы
function strictHistogramPreservation(yTrue, yPred) {
  return tf.tidy(() => {
    // Сортируем исходные пиксели
    const sortedTrue = yTrue.reshape([-1]).sort();
    
    // Сортируем предсказанные пиксели
    const sortedPred = yPred.reshape([-1]).sort();
    
    // Вычисляем разницу между соответствующими квантилями
    const diff = sortedTrue.sub(sortedPred);
    
    // Используем L1 норму для более строгого штрафа
    return tf.abs(diff).mean().mul(2.0);
  });
}

// ==========================================
// 3. Model Architecture
// ==========================================

// Baseline Model: Fixed Compression (Undercomplete AE)
function createBaselineModel() {
  const model = tf.sequential();
  model.add(tf.layers.flatten({ inputShape: CONFIG.inputShapeModel }));
  model.add(tf.layers.dense({ units: 64, activation: "relu" })); // Bottleneck
  model.add(tf.layers.dense({ units: 256, activation: "sigmoid" })); // Output 0-1
  model.add(tf.layers.reshape({ targetShape: [16, 16, 1] }));
  return model;
}

// ------------------------------------------------------------------
// [FIXED-A]: Студенческие архитектуры для всех трех типов
// ------------------------------------------------------------------
function createStudentModel(archType) {
  const model = tf.sequential();
  model.add(tf.layers.flatten({ inputShape: CONFIG.inputShapeModel }));

  if (archType === "compression") {
    // Compression: Сжимаем информацию, но стараемся сохранить распределение
    // Добавляем больше нейронов в скрытом слое для сохранения информации о цветах
    model.add(tf.layers.dense({ 
      units: 128,                    // Увеличили с 64 до 128 для лучшего сохранения цветов
      activation: "relu",
      kernelInitializer: 'heNormal'  // Лучшая инициализация для ReLU
    }));
    model.add(tf.layers.dense({ 
      units: 256, 
      activation: "sigmoid" 
    }));
  } else if (archType === "transformation") {
    // [FIXED]: Transformation - сохраняем размерность для точной перестановки пикселей
    model.add(tf.layers.dense({ 
      units: 256,                     // Полное сохранение размерности
      activation: "relu",
      kernelInitializer: 'heNormal'
    }));
    model.add(tf.layers.dense({ 
      units: 256, 
      activation: "sigmoid" 
    }));
  } else if (archType === "expansion") {
    // [FIXED]: Expansion - избыточная размерность для сложных преобразований
    model.add(tf.layers.dense({ 
      units: 512,                     // Увеличиваем размерность
      activation: "relu",
      kernelInitializer: 'heNormal'
    }));
    model.add(tf.layers.dense({ 
      units: 512,                     // Дополнительный слой для нелинейности
      activation: "relu"
    }));
    model.add(tf.layers.dense({ 
      units: 256, 
      activation: "sigmoid" 
    }));
  }

  model.add(tf.layers.reshape({ targetShape: [16, 16, 1] }));
  return model;
}

// ==========================================
// 4. Custom Loss Function
// ==========================================

// ------------------------------------------------------------------
// [FIXED-B]: Функция потерь для создания градиента с сохранением цветов
// ------------------------------------------------------------------
function studentLoss(yTrue, yPred) {
  return tf.tidy(() => {
    // 1. [ОСНОВНОЕ ОГРАНИЧЕНИЕ] Сохранение гистограммы цветов
    // Используем комбинацию мягкого и строгого сохранения
    const lossHistSoft = preserveHistogram(yTrue, yPred).mul(50.0);
    const lossHistStrict = strictHistogramPreservation(yTrue, yPred).mul(100.0);
    
    // 2. Гладкость - штраф за резкие переходы
    const lossSmooth = smoothness(yPred).mul(5.0);
    
    // 3. Направление градиента (горизонтальное и вертикальное)
    const lossDirX = directionX(yPred).mul(10.0);
    const lossDirY = directionY(yPred).mul(5.0); // Добавили вертикальный градиент
    
    // 4. [АДАПТИВНЫЙ MSE] Разный для разных архитектур
    const archType = document.querySelector('input[name="arch"]:checked').value;
    let lossMSE;
    
    if (archType === "compression") {
      // Compression нужно больше MSE для сохранения структуры
      lossMSE = mse(yTrue, yPred).mul(1.0);
    } else if (archType === "transformation") {
      // Transformation может обойтись малым MSE
      lossMSE = mse(yTrue, yPred).mul(0.2);
    } else {
      // Expansion нужен средний MSE
      lossMSE = mse(yTrue, yPred).mul(0.5);
    }
    
    // Суммируем все компоненты потерь
    return lossMSE
      .add(lossSmooth)
      .add(lossDirX)
      .add(lossDirY)
      .add(lossHistSoft)
      .add(lossHistStrict);
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

  // Train Baseline (MSE Only)
  const baselineLossVal = tf.tidy(() => {
    const { value, grads } = tf.variableGrads(() => {
      const yPred = state.baselineModel.predict(state.xInput);
      return mse(state.xInput, yPred);
    }, state.baselineModel.getWeights());

    state.optimizer.applyGradients(grads);
    return value.dataSync()[0];
  });

  // Train Student (Custom Loss)
  let studentLossVal = 0;
  try {
    studentLossVal = tf.tidy(() => {
      const { value, grads } = tf.variableGrads(() => {
        const yPred = state.studentModel.predict(state.xInput);
        return studentLoss(state.xInput, yPred);
      }, state.studentModel.getWeights());

      state.optimizer.applyGradients(grads);
      return value.dataSync()[0];
    });
    log(
      `Step ${state.step}: Base Loss=${baselineLossVal.toFixed(4)} | Student Loss=${studentLossVal.toFixed(4)}`
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
    document.getElementById("canvas-input")
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
  if (typeof archType !== "string") {
    archType = null;
  }

  if (state.isAutoTraining) {
    stopAutoTrain();
  }

  if (!archType) {
    const checked = document.querySelector('input[name="arch"]:checked');
    archType = checked ? checked.value : "compression";
  }

  if (state.baselineModel) {
    state.baselineModel.dispose();
    state.baselineModel = null;
  }
  if (state.studentModel) {
    state.studentModel.dispose();
    state.studentModel = null;
  }
  if (state.optimizer) {
    state.optimizer.dispose();
    state.optimizer = null;
  }

  state.baselineModel = createBaselineModel();
  
  // [FIXED]: Теперь все архитектуры реализованы
  try {
    state.studentModel = createStudentModel(archType);
  } catch (e) {
    log(`Error creating model: ${e.message}`, true);
    state.studentModel = createBaselineModel();
  }

  state.optimizer = tf.train.adam(CONFIG.learningRate);
  state.step = 0;

  log(`Models reset. Student Arch: ${archType}`);
  render();
}

async function render() {
  const basePred = state.baselineModel.predict(state.xInput);
  const studPred = state.studentModel.predict(state.xInput);

  await tf.browser.toPixels(
    basePred.squeeze(),
    document.getElementById("canvas-baseline")
  );
  await tf.browser.toPixels(
    studPred.squeeze(),
    document.getElementById("canvas-student")
  );

  basePred.dispose();
  studPred.dispose();
}

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
