/**
 * Main Radar Dashboard JavaScript Application
 * Handles API calls, Chart.js time-domain waveform rendering, UI updates,
 * file upload, and real-time classification logs.
 */

document.addEventListener('DOMContentLoaded', () => {
    // ----------------------------------------------------------------------
    // INITIALIZATION & DOM ELEMENTS
    // ----------------------------------------------------------------------
    const systemClockEl = document.getElementById('systemClock');
    const snrSlider = document.getElementById('snrSlider');
    const snrValueDisplay = document.getElementById('snrValueDisplay');
    
    // Buttons
    const btnGenDrone = document.getElementById('btnGenDrone');
    const btnGenBird = document.getElementById('btnGenBird');
    const btnGenNoise = document.getElementById('btnGenNoise');
    const btnRefreshHistory = document.getElementById('btnRefreshHistory');
    const btnClearHistory = document.getElementById('btnClearHistory');

    // Drag and drop
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');

    // Tabs
    const tabImgSpectrogram = document.getElementById('tabImgSpectrogram');
    const tabCanvasMatrix = document.getElementById('tabCanvasMatrix');
    const spectrogramImgWrapper = document.getElementById('spectrogramImgWrapper');
    const spectrogramCanvasWrapper = document.getElementById('spectrogramCanvasWrapper');
    const spectrogramImg = document.getElementById('spectrogramImg');
    const imgSpinner = document.getElementById('imgSpinner');

    // Visualizers
    let waveformChart = null;
    const canvasRenderer = new SpectrogramCanvasRenderer('matrixCanvas');
    let currentSpectrogramMatrix = null;

    // Start System Clock
    updateSystemClock();
    setInterval(updateSystemClock, 1000);

    // Initialize Chart.js Waveform
    initWaveformChart();

    // Load Initial Classification History
    loadHistory();

    // ----------------------------------------------------------------------
    // EVENT LISTENERS
    // ----------------------------------------------------------------------
    snrSlider.addEventListener('input', (e) => {
        snrValueDisplay.textContent = `${e.target.value} dB`;
    });

    btnGenDrone.addEventListener('click', () => fetchSampleTarget('drone'));
    btnGenBird.addEventListener('click', () => fetchSampleTarget('bird'));
    btnGenNoise.addEventListener('click', () => fetchSampleTarget('noise'));

    btnRefreshHistory.addEventListener('click', loadHistory);
    btnClearHistory.addEventListener('click', clearHistory);

    // Tab Switching
    tabImgSpectrogram.addEventListener('click', () => {
        tabImgSpectrogram.classList.add('active');
        tabCanvasMatrix.classList.remove('active');
        spectrogramImgWrapper.classList.add('active');
        spectrogramCanvasWrapper.classList.remove('active');
    });

    tabCanvasMatrix.addEventListener('click', () => {
        tabCanvasMatrix.classList.add('active');
        tabImgSpectrogram.classList.remove('active');
        spectrogramCanvasWrapper.classList.add('active');
        spectrogramImgWrapper.classList.remove('active');
        
        if (currentSpectrogramMatrix) {
            canvasRenderer.render(
                currentSpectrogramMatrix.freq,
                currentSpectrogramMatrix.time,
                currentSpectrogramMatrix.values
            );
        }
    });

    // File Drop & Upload
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('drag-over');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) {
            uploadFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            uploadFile(e.target.files[0]);
        }
    });

    // ----------------------------------------------------------------------
    // SYSTEM CLOCK
    // ----------------------------------------------------------------------
    function updateSystemClock() {
        const now = new Date();
        systemClockEl.textContent = now.toUTCString().split(' ')[4] + ' UTC';
    }

    // ----------------------------------------------------------------------
    // CHART.JS WAVEFORM INITIALIZATION
    // ----------------------------------------------------------------------
    function initWaveformChart() {
        const ctx = document.getElementById('waveformChart').getContext('2d');
        waveformChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'FMCW Beat Signal Amplitude',
                    data: [],
                    borderColor: '#00ff33',
                    borderWidth: 1.5,
                    borderCapStyle: 'square',
                    borderJoinStyle: 'miter',
                    pointRadius: 0,
                    pointHoverRadius: 3,
                    tension: 0,
                    fill: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                devicePixelRatio: Math.max(2, window.devicePixelRatio || 1),
                animation: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        callbacks: {
                            label: (ctx) => `Amp: ${ctx.parsed.y.toFixed(4)}`
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(0, 255, 51, 0.08)' },
                        ticks: { color: '#00ff33', maxTicksLimit: 10, font: { family: "'Fira Code', monospace", size: 10, weight: 'bold' } },
                        title: { display: true, text: 'TIME (S)', color: '#00ff33', font: { family: "'Fira Code', monospace", size: 10, weight: 'bold' } }
                    },
                    y: {
                        grid: { color: 'rgba(0, 255, 51, 0.08)' },
                        ticks: { color: '#00ff33', font: { family: "'Fira Code', monospace", size: 10, weight: 'bold' } },
                        title: { display: true, text: 'AMPLITUDE', color: '#00ff33', font: { family: "'Fira Code', monospace", size: 10, weight: 'bold' } }
                    }
                }
            }
        });
    }

    function updateWaveformChart(timeArray, ampArray) {
        if (!waveformChart) return;
        waveformChart.data.labels = timeArray.map(t => t.toFixed(3));
        waveformChart.data.datasets[0].data = ampArray;
        waveformChart.update();
        document.getElementById('waveformStats').textContent = `Samples: ${ampArray.length}`;
    }

    // ----------------------------------------------------------------------
    // ----------------------------------------------------------------------
    // PROGRAMMATIC PHYSICAL SIGNAL FILE GENERATOR FOR BENCHMARK TARGETS
    // ----------------------------------------------------------------------
    function generateSampleFile(targetType, snrDb) {
        const fs = 8000;
        const duration = 1.0;
        const N = Math.floor(fs * duration);
        let csvContent = "Time,Amplitude_dB\n";
        
        for (let i = 0; i < N; i++) {
            const t = i / fs;
            let signal = 0;
            if (targetType === 'drone') {
                const f_bulk = 180.0 + Math.random() * 80.0;
                const f_rot = 45.0 + Math.random() * 25.0;
                signal = 0.8 * Math.cos(2 * Math.PI * f_bulk * t) + 0.5 * Math.sin(2 * Math.PI * f_rot * t * 4);
            } else if (targetType === 'bird') {
                const f_bulk = 45.0 + Math.random() * 35.0;
                const wing_phase = 25.0 * Math.sin(2 * Math.PI * 5.0 * t);
                signal = 0.9 * Math.cos(2 * Math.PI * f_bulk * t + wing_phase);
            } else {
                signal = (Math.random() - 0.5) * 0.4;
            }
            
            const noiseAmp = Math.pow(10, -snrDb / 20.0);
            signal += (Math.random() - 0.5) * noiseAmp;
            csvContent += `${t.toFixed(4)},${signal.toFixed(6)}\n`;
        }

        const filename = `Synthetic_${targetType.toUpperCase()}_Telemetry.csv`;
        const blob = new Blob([csvContent], { type: 'text/csv' });
        return new File([blob], filename, { type: 'text/csv' });
    }

    async function fetchSampleTarget(targetType) {
        const snr = parseFloat(snrSlider.value);
        const physicalFile = generateSampleFile(targetType, snr);
        console.log(`[Frontend] Generated ${physicalFile.size} physical file bytes for ${targetType.toUpperCase()}`);
        await uploadFile(physicalFile);
    }

    // ----------------------------------------------------------------------
    // API CALLS: STREAM RADAR SIGNAL FILE BYTES TO SERVER
    // ----------------------------------------------------------------------
    async function uploadFile(file) {
        if (!file) return;
        showLoadingState(true);
        const formData = new FormData();
        formData.append('file', file);

        console.log(`[Frontend] Streaming ${file.size} physical bytes to /api/analyze-target:`, file);

        try {
            const response = await fetch('/api/analyze-target', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            if (!response.ok || !data.success) {
                const errMsg = data.message || data.error || 'Failed to process signal file stream';
                renderErrorState(errMsg);
                alert(`Backend Error: ${errMsg}`);
                return;
            }

            renderClassificationData(data);
        } catch (err) {
            console.error('File Stream Upload Error:', err);
            renderErrorState('Network Error: Failed to stream signal file bytes to backend server.');
            alert('Failed to stream signal file bytes to server.');
        } finally {
            showLoadingState(false);
        }
    }

    // ----------------------------------------------------------------------
    // RENDER CLASSIFICATION DATA & UI UPDATES
    // ----------------------------------------------------------------------
    function renderClassificationData(data) {
        // 1. Update Waveform
        if (data.waveform) {
            updateWaveformChart(data.waveform.time, data.waveform.amplitude);
        }

        // 2. Update Spectrogram Image & Canvas Matrix
        if (data.spectrogram_img) {
            spectrogramImg.src = data.spectrogram_img;
            spectrogramImg.style.display = 'block';
            const placeholder = document.getElementById('spectrogramPlaceholder');
            if (placeholder) placeholder.style.display = 'none';
        }

        if (data.spectrogram_matrix) {
            currentSpectrogramMatrix = data.spectrogram_matrix;
            canvasRenderer.render(
                currentSpectrogramMatrix.freq,
                currentSpectrogramMatrix.time,
                currentSpectrogramMatrix.values
            );
        }

        // 3. Verdict Card & Badge
        const verdictCard = document.getElementById('verdictCard');
        const verdictBadge = document.getElementById('verdictBadge');
        const verdictIcon = document.getElementById('verdictIcon');
        const verdictText = document.getElementById('verdictText');
        const verdictSubtext = document.getElementById('verdictSubtext');

        verdictCard.className = 'verdict-card';
        if (data.target_class === 'Drone') {
            verdictCard.classList.add('drone-threat');
            verdictIcon.className = 'fa-solid fa-triangle-exclamation';
            verdictText.textContent = 'DRONE THREAT DETECTED';
            verdictSubtext.textContent = 'High-frequency narrow-band propeller micro-Doppler modulations confirmed.';
        } else if (data.target_class === 'Bird') {
            verdictCard.classList.add('bird-bio');
            verdictIcon.className = 'fa-solid fa-crow';
            verdictText.textContent = 'BIRD / BIOLOGICAL TARGET';
            verdictSubtext.textContent = 'Wide-amplitude periodic wing-beat frequency sweep detected.';
        } else {
            verdictCard.classList.add('neutral');
            verdictIcon.className = 'fa-solid fa-wave-square';
            verdictText.textContent = 'AMBIENT NOISE / CLUTTER';
            verdictSubtext.textContent = 'No dominant micro-Doppler target features present.';
        }

        // 4. Confidence Progress Fill & Probabilities
        document.getElementById('confidenceScoreText').textContent = `${data.confidence.toFixed(1)}%`;
        document.getElementById('confidenceProgressFill').style.width = `${data.confidence}%`;

        const probs = data.probabilities || {};
        const pDrone = probs.Drone || 0;
        const pBird = probs.Bird || 0;
        const pNoise = probs.Noise || 0;

        document.getElementById('probDrone').textContent = `${pDrone.toFixed(1)}%`;
        document.getElementById('probBird').textContent = `${pBird.toFixed(1)}%`;
        document.getElementById('probNoise').textContent = `${pNoise.toFixed(1)}%`;

        document.getElementById('probDroneFill').style.width = `${pDrone}%`;
        document.getElementById('probBirdFill').style.width = `${pBird}%`;
        document.getElementById('probNoiseFill').style.width = `${pNoise}%`;

        // 5. Extracted Feature Metrics
        const feat = data.features || {};
        document.getElementById('metricDoppler').textContent = `${feat.doppler_shift_hz || 0} Hz`;
        document.getElementById('metricVelocity').textContent = `${feat.doppler_velocity_mps || 0} m/s`;
        document.getElementById('metricBandwidth').textContent = `${feat.mod_bandwidth_hz || 0} Hz`;
        document.getElementById('metricModRate').textContent = `${feat.mod_rate_hz || 0} Hz`;
        document.getElementById('metricCentroid').textContent = `${feat.spectral_centroid_hz || 0} Hz`;
        document.getElementById('metricHarmonic').textContent = `${feat.harmonic_ratio || 0}`;

        // 6. Summary Stats Header
        if (data.stats) {
            updateSummaryStats(data.stats);
        }

        // 7. Refresh Logs Table
        loadHistory();
    }

    // ----------------------------------------------------------------------
    // SUMMARY STATS & HISTORY LOG TABLE
    // ----------------------------------------------------------------------
    function updateSummaryStats(stats) {
        document.getElementById('statTotalScans').textContent = stats.total_scans || 0;
        document.getElementById('statDroneThreats').textContent = stats.drone_threats || 0;
        document.getElementById('statBirdTargets').textContent = stats.bird_targets || 0;
        document.getElementById('statAvgConfidence').textContent = `${(stats.avg_confidence || 0).toFixed(1)}%`;
    }

    async function loadHistory() {
        try {
            const response = await fetch('/api/history');
            const data = await response.json();
            if (!response.ok || !data.success) return;

            if (data.stats) updateSummaryStats(data.stats);

            const tbody = document.getElementById('historyTableBody');
            tbody.innerHTML = '';

            const logs = data.logs || [];
            if (logs.length === 0) {
                tbody.innerHTML = '<tr><td colspan="8" class="empty-table">No radar classification logs recorded yet.</td></tr>';
                return;
            }

            logs.forEach(log => {
                const tr = document.createElement('tr');
                let badgeClass = 'badge-noise';
                let iconClass = 'fa-wave-square';
                if (log.target_class === 'Drone') {
                    badgeClass = 'badge-drone';
                    iconClass = 'fa-triangle-exclamation';
                } else if (log.target_class === 'Bird') {
                    badgeClass = 'badge-bird';
                    iconClass = 'fa-crow';
                }

                tr.innerHTML = `
                    <td>#${log.id}</td>
                    <td><small class="text-muted">${log.timestamp}</small></td>
                    <td><strong>${log.filename}</strong></td>
                    <td>
                        <span class="badge-tag ${badgeClass}">
                            <i class="fa-solid ${iconClass}"></i> ${log.target_class}
                        </span>
                    </td>
                    <td><strong>${log.confidence.toFixed(1)}%</strong></td>
                    <td>${log.doppler_shift_hz ? log.doppler_shift_hz.toFixed(1) : '--'}</td>
                    <td>${log.mod_bandwidth_hz ? log.mod_bandwidth_hz.toFixed(1) : '--'}</td>
                    <td>${log.harmonic_ratio ? log.harmonic_ratio.toFixed(3) : '--'}</td>
                `;
                tbody.appendChild(tr);
            });

        } catch (err) {
            console.error('Failed to load history logs:', err);
        }
    }

    async function clearHistory() {
        if (!confirm('Are you sure you want to clear all radar classification logs?')) return;
        try {
            const response = await fetch('/api/history', { method: 'DELETE' });
            const data = await response.json();
            if (data.success) {
                // Clear table rows instantly
                const tbody = document.getElementById('historyTableBody');
                tbody.innerHTML = '<tr><td colspan="8" class="empty-table">NO RADAR CLASSIFICATION LOGS RECORDED YET.</td></tr>';
                // Reset metric summary counters
                document.getElementById('statTotalScans').textContent = '0';
                document.getElementById('statDroneThreats').textContent = '0';
                document.getElementById('statBirdTargets').textContent = '0';
                document.getElementById('statAvgConfidence').textContent = '0.0%';
                loadHistory();
            }
        } catch (err) {
            console.error('Failed to clear history:', err);
            alert('Failed to clear classification logs from server.');
        }
    }

    function showLoadingState(isLoading) {
        imgSpinner.style.display = isLoading ? 'flex' : 'none';
        
        // Lock UI buttons during backend STFT & PyTorch processing
        btnGenDrone.disabled = isLoading;
        btnGenBird.disabled = isLoading;
        btnGenNoise.disabled = isLoading;
        btnRefreshHistory.disabled = isLoading;
        btnClearHistory.disabled = isLoading;

        if (isLoading) {
            btnGenDrone.style.opacity = '0.5';
            btnGenBird.style.opacity = '0.5';
            btnGenNoise.style.opacity = '0.5';
            document.body.style.cursor = 'wait';
        } else {
            btnGenDrone.style.opacity = '1';
            btnGenBird.style.opacity = '1';
            btnGenNoise.style.opacity = '1';
            document.body.style.cursor = 'default';
        }
    }

    function renderErrorState(errorMessage) {
        const verdictCard = document.getElementById('verdictCard');
        const verdictIcon = document.getElementById('verdictIcon');
        const verdictText = document.getElementById('verdictText');
        const verdictSubtext = document.getElementById('verdictSubtext');

        verdictCard.className = 'verdict-card drone-threat';
        verdictIcon.className = 'fa-solid fa-triangle-exclamation';
        verdictText.textContent = 'TELEMETRY PROCESSING ERROR';
        verdictSubtext.textContent = errorMessage || 'Server failed to process file stream.';

        document.getElementById('confidenceScoreText').textContent = '0.0%';
        document.getElementById('confidenceProgressFill').style.width = '0%';
    }
});
