/**
 * Spectrogram Heatmap Canvas Renderer
 * Renders 2D Joint Time-Frequency STFT matrix on HTML5 canvas with custom Turbo/Viridis colormap.
 */

class SpectrogramCanvasRenderer {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) return;
        this.ctx = this.canvas.getContext('2d');
        
        // Setup Turbo Colormap Lookup Table (256 RGB values)
        this.colorLUT = this._generateTurboLUT();
    }

    render(freqArray, timeArray, values2D) {
        if (!this.canvas || !values2D || values2D.length === 0) return;

        const numFreq = values2D.length;       // Rows (Frequency)
        const numTime = values2D[0].length;    // Cols (Time)

        const width = this.canvas.width;
        const height = this.canvas.height;
        const padding = { top: 20, right: 60, bottom: 35, left: 55 };

        const plotWidth = width - padding.left - padding.right;
        const plotHeight = height - padding.top - padding.bottom;

        // Clear canvas
        this.ctx.fillStyle = '#070a12';
        this.ctx.fillRect(0, 0, width, height);

        // Find Min & Max dB values
        let minVal = Infinity;
        let maxVal = -Infinity;
        for (let r = 0; r < numFreq; r++) {
            for (let c = 0; c < numTime; c++) {
                const val = values2D[r][c];
                if (val < minVal) minVal = val;
                if (val > maxVal) maxVal = val;
            }
        }
        if (maxVal === minVal) maxVal = minVal + 1;

        // Create ImageData for high-speed pixel rendering
        const imgData = this.ctx.createImageData(plotWidth, plotHeight);
        const data = imgData.data;

        for (let py = 0; py < plotHeight; py++) {
            // Y-axis inverted (0 frequency at bottom)
            const rNorm = 1.0 - (py / plotHeight);
            const r = Math.min(numFreq - 1, Math.floor(rNorm * numFreq));

            for (let px = 0; px < plotWidth; px++) {
                const cNorm = px / plotWidth;
                const c = Math.min(numTime - 1, Math.floor(cNorm * numTime));

                const val = values2D[r][c];
                const normVal = Math.max(0, Math.min(1, (val - minVal) / (maxVal - minVal)));
                
                const lutIdx = Math.floor(normVal * 255);
                const rgb = this.colorLUT[lutIdx];

                const pixelIdx = (py * plotWidth + px) * 4;
                data[pixelIdx]     = rgb[0]; // R
                data[pixelIdx + 1] = rgb[1]; // G
                data[pixelIdx + 2] = rgb[2]; // B
                data[pixelIdx + 3] = 255;    // A
            }
        }

        // Draw heatmap image onto plot area
        this.ctx.putImageData(imgData, padding.left, padding.top);

        // Draw Grid Spines & Axes
        this.ctx.strokeStyle = '#1e293b';
        this.ctx.lineWidth = 1;
        this.ctx.strokeRect(padding.left, padding.top, plotWidth, plotHeight);

        // Draw Frequency Axis Labels (Y-axis)
        this.ctx.fillStyle = '#94a3b8';
        this.ctx.font = '10px "JetBrains Mono", monospace';
        this.ctx.textAlign = 'right';
        this.ctx.textBaseline = 'middle';

        const yTicks = 5;
        const maxFreq = freqArray ? freqArray[freqArray.length - 1] : 4000;
        for (let i = 0; i <= yTicks; i++) {
            const yPos = padding.top + plotHeight - (i / yTicks) * plotHeight;
            const freqVal = Math.round((i / yTicks) * maxFreq);
            this.ctx.fillText(`${freqVal} Hz`, padding.left - 8, yPos);
            
            // Grid line
            this.ctx.beginPath();
            this.ctx.strokeStyle = 'rgba(255,255,255,0.05)';
            this.ctx.moveTo(padding.left, yPos);
            this.ctx.lineTo(padding.left + plotWidth, yPos);
            this.ctx.stroke();
        }

        // Draw Time Axis Labels (X-axis)
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'top';
        const xTicks = 5;
        const maxTime = timeArray ? timeArray[timeArray.length - 1] : 1.0;
        for (let i = 0; i <= xTicks; i++) {
            const xPos = padding.left + (i / xTicks) * plotWidth;
            const timeVal = ((i / xTicks) * maxTime).toFixed(2);
            this.ctx.fillText(`${timeVal}s`, xPos, padding.top + plotHeight + 6);
        }

        // Colorbar Legend (Right side)
        const cbLeft = padding.left + plotWidth + 12;
        const cbWidth = 14;
        const cbHeight = plotHeight;

        const cbImgData = this.ctx.createImageData(cbWidth, cbHeight);
        const cbData = cbImgData.data;

        for (let py = 0; py < cbHeight; py++) {
            const normVal = 1.0 - (py / cbHeight);
            const lutIdx = Math.floor(normVal * 255);
            const rgb = this.colorLUT[lutIdx];

            for (let px = 0; px < cbWidth; px++) {
                const idx = (py * cbWidth + px) * 4;
                cbData[idx]     = rgb[0];
                cbData[idx + 1] = rgb[1];
                cbData[idx + 2] = rgb[2];
                cbData[idx + 3] = 255;
            }
        }
        this.ctx.putImageData(cbImgData, cbLeft, padding.top);
        this.ctx.strokeStyle = '#1e293b';
        this.ctx.strokeRect(cbLeft, padding.top, cbWidth, cbHeight);
    }

    _generateTurboLUT() {
        // High-contrast Turbo colormap interpolation
        const lut = [];
        for (let i = 0; i < 256; i++) {
            const x = i / 255.0;
            // Polynomial approximation for Turbo palette
            const r = Math.max(0, Math.min(255, Math.floor(255 * (0.1357 + x * (4.5155 + x * (-12.05 + x * (17.38 - x * 8.98)))))));
            const g = Math.max(0, Math.min(255, Math.floor(255 * (0.0914 + x * (2.0791 + x * (2.854 - x * (11.05 - x * 6.06)))))));
            const b = Math.max(0, Math.min(255, Math.floor(255 * (0.1067 + x * (12.585 - x * (40.45 - x * (48.33 - x * 19.34)))))));
            lut.push([r, g, b]);
        }
        return lut;
    }
}
