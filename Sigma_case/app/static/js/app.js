// Добавляем обработку ошибок в начале
console.log("JavaScript загружен");

document.addEventListener("DOMContentLoaded", () => {
  console.log("DOM загружен");
  
  const fileInput = document.getElementById("fileInput");
  const uploadArea = document.getElementById("uploadArea");
  const fileName = document.getElementById("fileName");
  const fileSettings = document.getElementById("fileSettings");
  const maxRowsInput = document.getElementById("maxRows");
  const fileInfo = document.getElementById("fileInfo");
  const submitBtn = document.getElementById("submitBtn");
  const statusCard = document.getElementById("statusCard");
  const resultCard = document.getElementById("resultCard");
  const progressFill = document.getElementById("progressFill");
  const progressText = document.getElementById("progressText");
  const statusMessage = document.getElementById("statusMessage");
  const downloadFullBtn = document.getElementById("downloadFullBtn");
  const downloadSimpleBtn = document.getElementById("downloadSimpleBtn");

  if (!fileInput || !uploadArea || !submitBtn) {
    console.error("Не найдены необходимые элементы DOM");
    return;
  }

  let currentFile = null;
  let downloadUrl = null;
  let fullResultName = null;
  let simpleResultName = null;

  // Обработка клика по области загрузки
  uploadArea.addEventListener("click", () => {
    fileInput.click();
  });

  // Обработка выбора файла
  fileInput.addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (file) {
      currentFile = file;
      fileName.textContent = `📄 ${file.name} (${(file.size / 1024).toFixed(2)} KB)`;
      uploadArea.classList.add("file-selected");
      
      // Получаем информацию о CSV
      await loadCSVInfo(file);
    }
  });

  // Drag and Drop
  uploadArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadArea.classList.add("dragover");
  });

  uploadArea.addEventListener("dragleave", () => {
    uploadArea.classList.remove("dragover");
  });

  uploadArea.addEventListener("drop", async (e) => {
    e.preventDefault();
    uploadArea.classList.remove("dragover");
    
    const file = e.dataTransfer.files[0];
    if (file && file.name.endsWith('.csv')) {
      currentFile = file;
      fileInput.files = e.dataTransfer.files;
      fileName.textContent = `📄 ${file.name} (${(file.size / 1024).toFixed(2)} KB)`;
      uploadArea.classList.add("file-selected");
      
      await loadCSVInfo(file);
    } else {
      showError("Пожалуйста, выберите CSV файл");
    }
  });

  // Загрузка информации о CSV
  async function loadCSVInfo(file) {
    try {
      showStatus("Анализ файла...", 10);
      
      const formData = new FormData();
      formData.append("file", file);
      
      const response = await fetch("/api/csv-info", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || "Ошибка при анализе файла");
      }

      const info = await response.json();
      
      // Проверяем наличие необходимых колонок
      const requiredColumns = ['№ вопроса', 'Транскрибация ответа', 'Оценка экзаменатора'];
      const missingColumns = requiredColumns.filter(col => !info.columns.includes(col));
      
      if (missingColumns.length > 0) {
        showError(`Отсутствуют необходимые колонки: ${missingColumns.join(', ')}`);
        submitBtn.disabled = true;
        return;
      }
      
      // Показываем информацию о файле
      fileInfo.textContent = `Найдено строк: ${info.row_count} | Колонок: ${info.columns.length}`;
      fileSettings.style.display = "block";
      submitBtn.disabled = false;
      hideStatus();
      
    } catch (error) {
      console.error("Ошибка при анализе файла:", error);
      showError(`Ошибка при анализе файла: ${error.message}`);
      submitBtn.disabled = true;
    }
  }

  // Обработка отправки
  submitBtn.addEventListener("click", async () => {
    if (!currentFile) {
      showError("Выберите файл");
      return;
    }

    let progressInterval = null;
    let progressPollInterval = null;

    try {
      // Показываем карточку статуса сразу
      statusCard.style.display = "block";
      resultCard.style.display = "none";
      submitBtn.disabled = true;

      const formData = new FormData();
      formData.append("file", currentFile);
      
      // Добавляем количество строк для обработки
      const maxRows = maxRowsInput.value ? parseInt(maxRowsInput.value) : null;
      if (maxRows && maxRows > 0) {
        formData.append("max_rows", maxRows);
      }

      // Начальный статус
      showStatus("Подготовка к обработке...", 5);
      
      let taskId = null;

      const response = await fetch("/api/evaluate-csv", {
        method: "POST",
        body: formData,
      });

      // Получаем task_id из заголовков
      taskId = response.headers.get("X-Task-Id");
      
      // Если есть task_id, начинаем опрос прогресса
      if (taskId) {
        progressPollInterval = setInterval(async () => {
          try {
            const progressResponse = await fetch(`/api/progress/${taskId}`);
            if (progressResponse.ok) {
              const progress = await progressResponse.json();
              
              const processed = progress.processed || 0;
              const total = progress.total || 0;
              const percent = progress.progress_percent || 0;
              const remainingTime = progress.estimated_remaining_time || 0;
              
              // Форматируем оставшееся время
              let timeText = "";
              if (remainingTime > 0) {
                if (remainingTime < 60) {
                  timeText = `~${Math.ceil(remainingTime)} сек`;
                } else {
                  const minutes = Math.floor(remainingTime / 60);
                  const seconds = Math.ceil(remainingTime % 60);
                  timeText = `~${minutes} мин ${seconds} сек`;
                }
              }
              
              const message = progress.message || `Обработка... ${processed}/${total}`;
              const statusMessage = timeText ? `${message} | Осталось: ${timeText}` : message;
              
              showStatus(statusMessage, percent);
              
              // Если обработка завершена или ошибка, останавливаем опрос
              if (progress.status === "completed" || progress.status === "error") {
                clearInterval(progressPollInterval);
                progressPollInterval = null;
                
                if (progress.status === "error") {
                  throw new Error(progress.error || "Ошибка обработки");
                }
              }
            }
          } catch (error) {
            console.error("Ошибка при получении прогресса:", error);
          }
        }, 500); // Опрашиваем каждые 500мс
      } else {
        // Fallback на старый способ, если нет task_id
        let currentProgress = 5;
        const progressSteps = [
          { progress: 10, message: "Загрузка файла на сервер..." },
          { progress: 20, message: "Очистка от HTML тегов..." },
          { progress: 30, message: "Анализ структуры данных..." },
          { progress: 40, message: "Загрузка AI модели..." },
          { progress: 50, message: "Оценка ответов AI моделью..." },
          { progress: 60, message: "Нормализация оценок..." },
          { progress: 70, message: "Расчет метрик..." },
          { progress: 80, message: "Сохранение результата..." },
        ];

        let stepIndex = 0;
        progressInterval = setInterval(() => {
          if (stepIndex < progressSteps.length) {
            const step = progressSteps[stepIndex];
            showStatus(step.message, step.progress);
            currentProgress = step.progress;
            stepIndex++;
          } else if (currentProgress < 95) {
            currentProgress += 1;
            showStatus("Обработка...", currentProgress);
          }
        }, 1500);
      }

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || `Ошибка: ${response.status}`);
      }

      showStatus("Получение результата...", 95);

      const blob = await response.blob();
      downloadUrl = window.URL.createObjectURL(blob);

      // Имена файлов из заголовков (percent-encoded)
      const fullHeader = response.headers.get("X-Full-Result");
      const simpleHeader = response.headers.get("X-Simple-Result");
      fullResultName = fullHeader ? decodeURIComponent(fullHeader) : null;
      simpleResultName = simpleHeader ? decodeURIComponent(simpleHeader) : null;
      
      const contentDisposition = response.headers.get("content-disposition");
      let filename = "прогноз.csv";
      if (contentDisposition) {
        const filenameMatch = contentDisposition.match(/filename="?(.+)"?/);
        if (filenameMatch) {
          filename = filenameMatch[1];
        }
      }

      showStatus("✅ Готово!", 100);
      
      // Показываем карточку с результатом
      setTimeout(() => {
        statusCard.style.display = "none";
        resultCard.style.display = "block";
        // Кнопка полного файла: если есть ссылка на сохраненный файл на сервере, используем /api/download-result
        downloadFullBtn.onclick = () => {
          if (fullResultName) {
            const a = document.createElement("a");
            a.href = `/api/download-result?name=${encodeURIComponent(fullResultName)}`;
            a.click();
          } else {
            // Fallback: скачать тот blob, что вернулся как основной ответ
            downloadFile(filename);
          }
        };

        // Кнопка упрощенного файла
        if (simpleResultName) {
          downloadSimpleBtn.disabled = false;
          downloadSimpleBtn.onclick = () => {
            const a = document.createElement("a");
            a.href = `/api/download-result?name=${encodeURIComponent(simpleResultName)}`;
            a.click();
          };
        } else {
          downloadSimpleBtn.disabled = true;
        }
      }, 1000);

    } catch (error) {
      console.error("Ошибка при обработке:", error);
      // Останавливаем интервалы прогресса при ошибке
      if (progressInterval) {
        clearInterval(progressInterval);
        progressInterval = null;
      }
      if (progressPollInterval) {
        clearInterval(progressPollInterval);
        progressPollInterval = null;
      }
      
      showError(`Ошибка: ${error.message}`);
      submitBtn.disabled = false;
    }
  });

  // Скачивание файла
  function downloadFile(filename) {
    if (!downloadUrl) return;
    
    const a = document.createElement("a");
    a.href = downloadUrl;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    
    showSuccess("Файл успешно скачан!");
  }

  // Вспомогательные функции
  function showStatus(message, progress) {
    statusCard.style.display = "block";
    progressText.textContent = message;
    // Плавное обновление прогресса
    progressFill.style.width = `${Math.min(100, Math.max(0, progress))}%`;
    statusMessage.textContent = "";
    statusMessage.className = "status-message";
    
    // Прокручиваем карточку статуса в видимую область
    statusCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  function hideStatus() {
    statusCard.style.display = "none";
  }

  function showError(message) {
    statusCard.style.display = "block";
    statusMessage.textContent = message;
    statusMessage.className = "status-message error";
    progressFill.style.width = "0%";
    progressText.textContent = "Ошибка";
  }

  function showSuccess(message) {
    statusMessage.textContent = message;
    statusMessage.className = "status-message success";
  }
  
  console.log("Инициализация завершена");
});
