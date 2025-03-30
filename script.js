document.addEventListener("DOMContentLoaded", function () {
    const trainForm = document.getElementById("train-form");
    const progressContainer = document.getElementById("progress-container");
    const progressBar = document.getElementById("progress-bar");
    const metricsDiv = document.getElementById("metrics");

    trainForm.addEventListener("submit", function (event) {
        event.preventDefault();

        const rounds = parseInt(document.getElementById("rounds").value, 10);
        const learningRate = parseFloat(document.getElementById("learning-rate").value);

        progressContainer.classList.remove("hidden");
        progressBar.value = 0;

        simulateTraining(rounds, learningRate);
    });

    function simulateTraining(rounds, learningRate) {
        let lossValues = [];
        let accuracyValues = [];
        let xValues = [];

        let i = 0;
        function trainStep() {
            if (i < rounds) {
                let loss = Math.exp(-0.1 * (i + 1)) + (Math.random() * 0.1);
                let accuracy = Math.tanh((i + 1) / rounds) + (Math.random() * 0.1);

                lossValues.push(loss);
                accuracyValues.push(accuracy);
                xValues.push(i + 1);

                progressBar.value = ((i + 1) / rounds) * 100;
                i++;
                setTimeout(trainStep, 500); 
            } else {
                progressContainer.classList.add("hidden");
                renderCharts(xValues, lossValues, accuracyValues);
                showSection("metrics-section");
            }
        }
        trainStep();
    }

    function renderCharts(xValues, lossValues, accuracyValues) {
        metricsDiv.innerHTML = `<canvas id="lossChart"></canvas><canvas id="accuracyChart"></canvas>`;

        setTimeout(() => {
            const ctx1 = document.getElementById("lossChart").getContext("2d");
            new Chart(ctx1, {
                type: "line",
                data: {
                    labels: xValues,
                    datasets: [{ label: "Loss", data: lossValues, borderColor: "red", fill: false }]
                },
                options: { responsive: true }
            });

            const ctx2 = document.getElementById("accuracyChart").getContext("2d");
            new Chart(ctx2, {
                type: "line",
                data: {
                    labels: xValues,
                    datasets: [{ label: "Accuracy", data: accuracyValues, borderColor: "green", fill: false }]
                },
                options: { responsive: true }
            });
        }, 100);
    }

    function showSection(sectionId) {
        document.querySelectorAll("main section").forEach(section => {
            section.classList.add("hidden");
        });
        document.getElementById(sectionId).classList.remove("hidden");
    }

    function showHome() {
        document.querySelectorAll("main section").forEach(section => {
            section.classList.add("hidden");
        });
    }

    window.showSection = showSection;
    window.showHome = showHome;
});
