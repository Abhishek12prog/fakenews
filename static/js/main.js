document.addEventListener("DOMContentLoaded", () => {
    const predictForm = document.getElementById("predictForm");
    if (!predictForm) {
        return;
    }

    const loadingSpinner = document.getElementById("loadingSpinner");
    const resultCard = document.getElementById("resultCard");
    const resultBadge = document.getElementById("resultBadge");
    const championModel = document.getElementById("championModel");
    const selectedModelLabel = document.getElementById("selectedModelLabel");
    const confidenceScore = document.getElementById("confidenceScore");
    const modelAccuracy = document.getElementById("modelAccuracy");
    const probabilityList = document.getElementById("probabilityList");
    const newsText = document.getElementById("newsText");
    const selectedModel = document.getElementById("selectedModel");

    const progressColors = ["bg-success", "bg-danger", "bg-primary", "bg-warning", "bg-info"];

    const formatModelName = (name) =>
        name
            .split("_")
            .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
            .join(" ");

    predictForm.addEventListener("submit", async (event) => {
        event.preventDefault();

        const payload = {
            news_text: newsText.value.trim(),
            selected_model: selectedModel.value,
        };
        if (payload.news_text.length < 30) {
            window.alert("Please enter at least 30 characters of news text.");
            return;
        }

        loadingSpinner.classList.remove("d-none");
        loadingSpinner.classList.add("d-flex");

        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });

            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.error || "Prediction failed.");
            }

            const prediction = data.prediction;
            const confidence = Number(data.confidence).toFixed(2);
            const accuracy = (Number(data.model_accuracy) * 100).toFixed(2);

            resultCard.classList.remove("d-none", "real-result", "fake-result");
            resultCard.classList.add(prediction === "REAL" ? "real-result" : "fake-result");
            resultBadge.className = `badge rounded-pill fs-6 px-3 py-2 ${prediction === "REAL" ? "text-bg-success" : "text-bg-danger"}`;
            resultBadge.textContent = prediction;
            selectedModelLabel.textContent = formatModelName(data.selected_model);
            championModel.textContent = formatModelName(data.champion_model);
            confidenceScore.textContent = `${confidence}%`;
            modelAccuracy.textContent = `${accuracy}%`;

            probabilityList.innerHTML = "";
            Object.entries(data.probabilities).forEach(([modelName, value], index) => {
                const percent = Number(value).toFixed(2);
                const wrapper = document.createElement("div");
                wrapper.className = "mb-3";
                wrapper.innerHTML = `
                    <div class="d-flex justify-content-between">
                        <span>${formatModelName(modelName)}</span>
                        <strong>${percent}%</strong>
                    </div>
                    <div class="progress graph-progress">
                        <div class="progress-bar ${progressColors[index % progressColors.length]}" role="progressbar" style="width: ${percent}%"></div>
                    </div>
                `;
                probabilityList.appendChild(wrapper);
            });
        } catch (error) {
            window.alert(error.message);
        } finally {
            loadingSpinner.classList.add("d-none");
            loadingSpinner.classList.remove("d-flex");
        }
    });
});
