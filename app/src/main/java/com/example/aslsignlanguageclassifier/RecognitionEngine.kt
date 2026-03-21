package com.example.aslsignlanguageclassifier

import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import android.os.Handler
import android.os.Looper
import android.util.Log
import androidx.annotation.OptIn
import androidx.camera.core.CameraSelector
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.lifecycle.LifecycleOwner
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import org.tensorflow.lite.Interpreter
import java.io.ByteArrayOutputStream
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.round
import kotlin.math.sqrt

private const val TAG = "ASL_APP"

class RecognitionEngine(
    private val context: Context
) {
    private val _uiState = MutableStateFlow(RecognitionUiState())
    val uiState: StateFlow<RecognitionUiState> = _uiState.asStateFlow()

    private val mainHandler = Handler(Looper.getMainLooper())

    private val cameraExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private val preview = Preview.Builder().build()
    private val imageAnalyzer = ImageAnalysis.Builder()
        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
        .build()

    private val labels: List<String> by lazy {
        loadLabels(context, "letter_labels.txt")
    }

    private val wordLabels: List<String> by lazy {
        loadLabels(context, "class_names_noface50words.txt")
    }

    private val interpreter: Interpreter by lazy {
        Interpreter(loadModelFile(context, "asl_letters_augmented.tflite"))
    }

    private val wordInterpreter: Interpreter by lazy {
        Interpreter(loadModelFile(context, "asl_word_bilstm_normvel50words_compatible.tflite"))
    }

    private val handLandmarker: HandLandmarker by lazy {
        val baseOptions = com.google.mediapipe.tasks.core.BaseOptions.builder()
            .setModelAssetPath("hand_landmarker.task")
            .build()

        val options = HandLandmarker.HandLandmarkerOptions.builder()
            .setBaseOptions(baseOptions)
            .setRunningMode(RunningMode.IMAGE)
            .setNumHands(2)
            .build()

        HandLandmarker.createFromOptions(context, options)
    }

    private val poseLandmarker: PoseLandmarker by lazy {
        val baseOptions = com.google.mediapipe.tasks.core.BaseOptions.builder()
            .setModelAssetPath("pose_landmarker_lite.task")
            .build()

        val options = PoseLandmarker.PoseLandmarkerOptions.builder()
            .setBaseOptions(baseOptions)
            .setRunningMode(RunningMode.IMAGE)
            .setNumPoses(1)
            .build()

        PoseLandmarker.createFromOptions(context, options)
    }

    private val wordCaptureFrames = 16
    private val minCaptureFrames = 8
    private val endOnMissingHandFrames = 2
    private val predictionHoldMs = 2500L
    private val analysisIntervalMs = 80L

    private var freezePredictionUntil = 0L
    private var captureActive = false
    private var missingHandFrames = 0
    private var lastAnalysisTime = 0L
    private val wordCaptureBuffer = ArrayList<FloatArray>()

    private val letterOutput by lazy {
        Array(1) { FloatArray(labels.size.coerceAtLeast(1)) }
    }

    @Volatile
    private var isStarted = false

    fun start(previewView: PreviewView, lifecycleOwner: LifecycleOwner) {
        if (isStarted) return
        isStarted = true

        updateUi {
            it.copy(
                displayStatus = "Starting engine...",
                predictionText = "Loading models...",
                isEngineReady = false,
                errorMessage = null,
                showClearButton = it.isWordMode
            )
        }

        try {
            preview.setSurfaceProvider(previewView.surfaceProvider)

            // Force initialization here so any asset/model problem is caught early.
            labels
            wordLabels
            interpreter
            wordInterpreter
            handLandmarker
            poseLandmarker

            imageAnalyzer.setAnalyzer(cameraExecutor) { imageProxy ->
                try {
                    analyzeFrame(imageProxy)
                } catch (e: Exception) {
                    Log.e(TAG, "Analyzer error", e)
                    updateUi {
                        it.copy(
                            displayStatus = "Error",
                            predictionText = "Analyzer failed",
                            isRecording = false,
                            errorMessage = e.message ?: e.javaClass.simpleName
                        )
                    }
                } finally {
                    imageProxy.close()
                }
            }

            val cameraProvider = ProcessCameraProvider.getInstance(context).get()
            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                lifecycleOwner,
                CameraSelector.DEFAULT_FRONT_CAMERA,
                preview,
                imageAnalyzer
            )

            updateUi {
                it.copy(
                    displayStatus = if (it.isWordMode) "Ready for word" else "Ready for letter",
                    predictionText = "Show hand to sign",
                    isEngineReady = true,
                    isRecording = false,
                    showClearButton = it.isWordMode,
                    errorMessage = null
                )
            }
        } catch (e: Exception) {
            Log.e(TAG, "Engine start failed", e)
            updateUi {
                it.copy(
                    displayStatus = "Initialization failed",
                    predictionText = "Check models/assets",
                    isEngineReady = false,
                    isRecording = false,
                    errorMessage = e.message ?: e.javaClass.simpleName
                )
            }
        }
    }

    fun toggleMode() {
        val newWordMode = !_uiState.value.isWordMode
        clearInternalState()

        updateUi {
            it.copy(
                isWordMode = newWordMode,
                displayStatus = if (newWordMode) "Ready for word" else "Ready for letter",
                predictionText = if (newWordMode) {
                    "Switched to WORD mode"
                } else {
                    "Switched to LETTER mode"
                },
                lastPredictedWord = if (newWordMode) it.lastPredictedWord else it.lastPredictedWord,
                isRecording = false,
                showClearButton = newWordMode,
                errorMessage = null
            )
        }
    }

    fun clearWordState() {
        clearInternalState()
        updateUi {
            it.copy(
                displayStatus = "Ready for word",
                predictionText = "Cleared",
                isRecording = false,
                showClearButton = true,
                errorMessage = null
            )
        }
    }

    fun release() {
        try {
            imageAnalyzer.clearAnalyzer()
        } catch (_: Exception) {
        }

        try {
            handLandmarker.close()
        } catch (_: Exception) {
        }

        try {
            poseLandmarker.close()
        } catch (_: Exception) {
        }

        try {
            interpreter.close()
        } catch (_: Exception) {
        }

        try {
            wordInterpreter.close()
        } catch (_: Exception) {
        }

        try {
            cameraExecutor.shutdown()
        } catch (_: Exception) {
        }

        isStarted = false
    }

    private fun updateUi(block: (RecognitionUiState) -> RecognitionUiState) {
        if (Looper.myLooper() == Looper.getMainLooper()) {
            _uiState.value = block(_uiState.value)
        } else {
            mainHandler.post {
                _uiState.value = block(_uiState.value)
            }
        }
    }

    private fun clearInternalState() {
        captureActive = false
        missingHandFrames = 0
        freezePredictionUntil = 0L
        wordCaptureBuffer.clear()
    }

    private fun analyzeFrame(imageProxy: ImageProxy) {
        val now = System.currentTimeMillis()
        if (now - lastAnalysisTime < analysisIntervalMs) return
        lastAnalysisTime = now

        val bitmap = imageProxyToBitmap(imageProxy) ?: return
        val rotatedBitmap = safeRotateBitmap(
            bitmap = bitmap,
            rotationDegrees = imageProxy.imageInfo.rotationDegrees.toFloat()
        ) ?: return

        val mpImage = BitmapImageBuilder(rotatedBitmap).build()
        val handResult = handLandmarker.detect(mpImage)

        if (!_uiState.value.isWordMode) {
            handleLetterMode(handResult)
        } else {
            val poseResult = poseLandmarker.detect(mpImage)
            handleWordMode(now, handResult, poseResult)
        }
    }

    private fun handleLetterMode(handResult: HandLandmarkerResult) {
        if (handResult.landmarks().isEmpty() ||
            handResult.handednesses().isEmpty() ||
            handResult.handednesses()[0].isEmpty()
        ) {
            updateUi {
                if (it.displayStatus == "No hand" && it.predictionText == "-") {
                    it
                } else {
                    it.copy(
                        displayStatus = "No hand",
                        predictionText = "-",
                        isRecording = false,
                        showClearButton = false
                    )
                }
            }
            return
        }

        val landmarks = handResult.landmarks()[0]
        val handedness = handResult.handednesses()[0][0].categoryName()

        var normalized = normalizeLandmarksLetter(landmarks)
        if (handedness.equals("Left", ignoreCase = true)) {
            normalized = mirrorNormalizedX(normalized)
        }

        val inputBuffer = normalized21x3ToInputBuffer(normalized)
        interpreter.run(inputBuffer, letterOutput)

        val scores = letterOutput[0]
        val bestIdx = argMax(scores)
        val conf = scores[bestIdx]

        val prediction = "Letter top: ${labels[bestIdx]} (${String.format("%.4f", conf)})"

        updateUi {
            it.copy(
                displayStatus = "Hand: $handedness",
                predictionText = prediction,
                isRecording = false,
                showClearButton = false,
                errorMessage = null
            )
        }
    }

    private fun handleWordMode(
        now: Long,
        handResult: HandLandmarkerResult,
        poseResult: PoseLandmarkerResult
    ) {
        val hasHand = handResult.landmarks().isNotEmpty()

        if (hasHand) {
            if (!captureActive) {
                captureActive = true
                missingHandFrames = 0
                wordCaptureBuffer.clear()

                updateUi {
                    it.copy(
                        displayStatus = "Recording...",
                        predictionText = "Started auto capture...",
                        isRecording = true,
                        showClearButton = true,
                        errorMessage = null
                    )
                }
            }

            val frame258 = buildWordRawFrameFromPoseAndHands(poseResult, handResult)
            wordCaptureBuffer.add(frame258)

            updateUi {
                it.copy(
                    displayStatus = "Recording...",
                    predictionText = "Recording word clip... ${wordCaptureBuffer.size}/$wordCaptureFrames",
                    isRecording = true,
                    showClearButton = true,
                    errorMessage = null
                )
            }

            if (wordCaptureBuffer.size >= wordCaptureFrames) {
                finishWordPrediction(now)
            }
        } else {
            if (captureActive) {
                missingHandFrames++

                if (missingHandFrames >= endOnMissingHandFrames &&
                    wordCaptureBuffer.size >= minCaptureFrames
                ) {
                    finishWordPrediction(now)
                } else {
                    updateUi {
                        it.copy(
                            displayStatus = "Recording...",
                            predictionText = "Waiting for hand... ${wordCaptureBuffer.size}/$wordCaptureFrames",
                            isRecording = true,
                            showClearButton = true,
                            errorMessage = null
                        )
                    }
                }
            } else if (now > freezePredictionUntil) {
                updateUi {
                    it.copy(
                        displayStatus = "Ready for word",
                        predictionText = "Show hand to sign",
                        isRecording = false,
                        showClearButton = true
                    )
                }
            }
        }
    }

    private fun finishWordPrediction(now: Long) {
        val resultText = runWordInferenceHolisticStyle(
            captureBuffer = wordCaptureBuffer,
            wordInterpreter = wordInterpreter,
            wordLabels = wordLabels
        )

        captureActive = false
        missingHandFrames = 0
        freezePredictionUntil = now + predictionHoldMs
        wordCaptureBuffer.clear()

        updateUi {
            it.copy(
                displayStatus = "Predicted",
                predictionText = resultText,
                lastPredictedWord = extractPredictedWord(resultText),
                isRecording = false,
                showClearButton = true,
                errorMessage = null
            )
        }
    }
}

private fun normalizeLandmarksLetter(lm: List<NormalizedLandmark>): FloatArray {
    val out = FloatArray(63)
    val wx = lm[0].x()
    val wy = lm[0].y()
    val wz = lm[0].z()

    var scale = 0f
    for (i in 0 until 21) {
        val x = lm[i].x() - wx
        val y = lm[i].y() - wy
        val z = lm[i].z() - wz
        scale += x * x + y * y + z * z
    }
    scale = sqrt(scale / 21f).coerceAtLeast(1e-6f)

    var idx = 0
    for (i in 0 until 21) {
        out[idx++] = (lm[i].x() - wx) / scale
        out[idx++] = (lm[i].y() - wy) / scale
        out[idx++] = (lm[i].z() - wz) / scale
    }
    return out
}

private fun normalized21x3ToInputBuffer(v: FloatArray): ByteBuffer {
    val b = ByteBuffer.allocateDirect(v.size * 4).order(ByteOrder.nativeOrder())
    v.forEach { b.putFloat(it) }
    b.rewind()
    return b
}

private fun mirrorNormalizedX(v: FloatArray): FloatArray {
    val out = v.copyOf()
    for (i in 0 until 21) {
        out[i * 3] *= -1f
    }
    return out
}

private fun argMax(a: FloatArray): Int {
    var idx = 0
    for (i in a.indices) {
        if (a[i] > a[idx]) idx = i
    }
    return idx
}

private fun loadLabels(c: Context, name: String): List<String> {
    return c.assets.open(name).bufferedReader().readLines()
}

private fun loadModelFile(c: Context, name: String): ByteBuffer {
    val fd = c.assets.openFd(name)
    FileInputStream(fd.fileDescriptor).use { input ->
        val ch = input.channel
        return ch.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
    }
}

private fun buildWordRawFrameFromPoseAndHands(
    poseResult: PoseLandmarkerResult,
    handResult: HandLandmarkerResult
): FloatArray {
    val raw = FloatArray(258)

    if (poseResult.landmarks().isNotEmpty()) {
        val pose = poseResult.landmarks()[0]
        var idx = 0
        for (i in 0 until minOf(33, pose.size)) {
            raw[idx++] = pose[i].x()
            raw[idx++] = pose[i].y()
            raw[idx++] = pose[i].z()
            raw[idx++] = 0f
        }
    }

    val hands = handResult.landmarks()
    val handednesses = handResult.handednesses()

    for (h in hands.indices) {
        if (h >= handednesses.size || handednesses[h].isEmpty()) continue
        val handed = handednesses[h][0].categoryName()
        val offset = if (handed.equals("Left", true)) 132 else 195
        val hand = hands[h]
        var idx = offset
        for (i in 0 until minOf(21, hand.size)) {
            raw[idx++] = hand[i].x()
            raw[idx++] = hand[i].y()
            raw[idx++] = hand[i].z()
        }
    }

    return raw
}

private fun normalizeHand(handFlat: FloatArray): FloatArray {
    if (handFlat.size != 63) return FloatArray(63)
    val out = handFlat.copyOf()

    var allZero = true
    for (v in out) {
        if (v != 0f) {
            allZero = false
            break
        }
    }
    if (allZero) return FloatArray(63)

    val x0 = out[0]
    val y0 = out[1]
    val z0 = out[2]

    var scale = 0f
    for (i in 0 until 21) {
        val base = i * 3
        val x = out[base] - x0
        val y = out[base + 1] - y0
        val z = out[base + 2] - z0
        scale += x * x + y * y + z * z
    }
    scale = sqrt(scale / 21f).coerceAtLeast(1e-6f)

    for (i in 0 until 21) {
        val base = i * 3
        out[base] = (out[base] - x0) / scale
        out[base + 1] = (out[base + 1] - y0) / scale
        out[base + 2] = (out[base + 2] - z0) / scale
    }

    return out
}

private fun normalizePose(poseFlat: FloatArray): FloatArray {
    if (poseFlat.size != 132) return FloatArray(132)
    val out = poseFlat.copyOf()

    var xyzAllZero = true
    for (i in 0 until 33) {
        val base = i * 4
        if (out[base] != 0f || out[base + 1] != 0f || out[base + 2] != 0f) {
            xyzAllZero = false
            break
        }
    }
    if (xyzAllZero) return out

    val ls = 11 * 4
    val rs = 12 * 4

    val ldx = out[ls]
    val ldy = out[ls + 1]
    val ldz = out[ls + 2]
    val rdx = out[rs]
    val rdy = out[rs + 1]
    val rdz = out[rs + 2]

    val leftOk = !(ldx == 0f && ldy == 0f && ldz == 0f)
    val rightOk = !(rdx == 0f && rdy == 0f && rdz == 0f)

    val centerX: Float
    val centerY: Float
    val centerZ: Float
    val scale: Float

    if (leftOk && rightOk) {
        centerX = (ldx + rdx) / 2f
        centerY = (ldy + rdy) / 2f
        centerZ = (ldz + rdz) / 2f
        scale = sqrt(
            (ldx - rdx) * (ldx - rdx) +
                    (ldy - rdy) * (ldy - rdy) +
                    (ldz - rdz) * (ldz - rdz)
        ).coerceAtLeast(1e-6f)
    } else {
        centerX = out[0]
        centerY = out[1]
        centerZ = out[2]
        var sum = 0f
        var count = 0
        for (i in 0 until 33) {
            val base = i * 4
            val x = out[base]
            val y = out[base + 1]
            val z = out[base + 2]
            val norm = sqrt(x * x + y * y + z * z)
            if (norm > 0f) {
                sum += norm
                count++
            }
        }
        scale = if (count > 0) (sum / count).coerceAtLeast(1e-6f) else 1f
    }

    for (i in 0 until 33) {
        val base = i * 4
        out[base] = (out[base] - centerX) / scale
        out[base + 1] = (out[base + 1] - centerY) / scale
        out[base + 2] = (out[base + 2] - centerZ) / scale
    }
    return out
}

private fun normalizeWordFrame(rawFrame: FloatArray): FloatArray {
    val pose = normalizePose(rawFrame.copyOfRange(0, 132))
    val lh = normalizeHand(rawFrame.copyOfRange(132, 195))
    val rh = normalizeHand(rawFrame.copyOfRange(195, 258))
    val out = FloatArray(258)
    System.arraycopy(pose, 0, out, 0, 132)
    System.arraycopy(lh, 0, out, 132, 63)
    System.arraycopy(rh, 0, out, 195, 63)
    return out
}

private fun resampleSequenceRaw(seq: List<FloatArray>, targetLen: Int = 30): List<FloatArray> {
    if (seq.isEmpty()) return emptyList()
    if (seq.size == targetLen) return seq
    val out = ArrayList<FloatArray>(targetLen)
    for (i in 0 until targetLen) {
        val idx = round((i.toFloat() * (seq.size - 1)) / (targetLen - 1)).toInt()
        out.add(seq[idx].copyOf())
    }
    return out
}

private fun preprocessWordSequence(rawSeq: List<FloatArray>): Array<FloatArray> {
    val wordSeqLen = 30
    val wordTrimStart = 4

    var seq = rawSeq
    if (seq.size >= wordTrimStart + wordSeqLen) {
        seq = seq.drop(wordTrimStart)
    }
    seq = resampleSequenceRaw(seq, wordSeqLen)

    val normSeq = Array(seq.size) { i -> normalizeWordFrame(seq[i]) }
    val featSeq = Array(seq.size) { FloatArray(516) }

    for (i in normSeq.indices) {
        val prev = if (i == 0) normSeq[i] else normSeq[i - 1]
        for (j in 0 until 258) {
            featSeq[i][j] = normSeq[i][j]
            featSeq[i][258 + j] = normSeq[i][j] - prev[j]
        }
    }
    return featSeq
}

private fun buildWindowBatches(rawCapture: List<FloatArray>): Array<Array<FloatArray>> {
    val windows = ArrayList<Array<FloatArray>>()
    if (rawCapture.size >= 38) {
        val slices = listOf(4 to 34, 6 to 36, 8 to 38)
        for ((start, end) in slices) {
            windows.add(preprocessWordSequence(rawCapture.subList(start, end)))
        }
    } else {
        windows.add(preprocessWordSequence(rawCapture))
    }
    return windows.toTypedArray()
}

private fun runWordInferenceHolisticStyle(
    captureBuffer: ArrayList<FloatArray>,
    wordInterpreter: Interpreter,
    wordLabels: List<String>
): String {
    if (captureBuffer.isEmpty()) return "No sequence"
    if (wordLabels.isEmpty()) return "Word: UNKNOWN (TRY AGAIN)"

    val batches = buildWindowBatches(captureBuffer)
    val avg = FloatArray(wordLabels.size)

    for (batch in batches) {
        val input = Array(1) { Array(30) { FloatArray(516) } }
        for (i in 0 until 30) {
            for (j in 0 until 516) {
                input[0][i][j] = batch[i][j]
            }
        }

        val output = Array(1) { FloatArray(wordLabels.size) }
        wordInterpreter.run(input, output)

        for (k in avg.indices) {
            avg[k] += output[0][k]
        }
    }

    for (k in avg.indices) {
        avg[k] /= batches.size
    }

    val topIdx = avg.indices.sortedByDescending { avg[it] }.take(3)
    val bestIdx = topIdx[0]
    val conf1 = avg[bestIdx]
    val conf2 = if (topIdx.size > 1) avg[topIdx[1]] else 0f
    val margin = conf1 - conf2

    val status = when {
        conf1 >= 0.60f && margin >= 0.11f -> "ACCEPTED"
        conf1 >= 0.42f && margin >= 0.06f -> "MAYBE"
        else -> "TRY AGAIN"
    }

    return if (status == "TRY AGAIN") {
        "Word: UNKNOWN ($status)"
    } else {
        "Word: ${wordLabels[bestIdx]} (${String.format("%.4f", conf1)}) [$status]"
    }
}

private fun extractPredictedWord(resultText: String): String {
    return if (resultText.startsWith("Word: ")) {
        resultText.removePrefix("Word: ")
            .substringBefore(" (")
            .substringBefore(" [")
            .trim()
    } else {
        resultText
    }
}

private fun safeRotateBitmap(bitmap: Bitmap, rotationDegrees: Float): Bitmap? {
    return try {
        if (rotationDegrees == 0f) {
            bitmap
        } else {
            Bitmap.createBitmap(
                bitmap,
                0,
                0,
                bitmap.width,
                bitmap.height,
                Matrix().apply { postRotate(rotationDegrees) },
                true
            )
        }
    } catch (e: Exception) {
        Log.e(TAG, "safeRotateBitmap failed", e)
        null
    }
}

@OptIn(ExperimentalGetImage::class)
private fun imageProxyToBitmap(image: ImageProxy): Bitmap? {
    return try {
        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = YuvImage(
            nv21,
            ImageFormat.NV21,
            image.width,
            image.height,
            null
        )

        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), 70, out)
        val bytes = out.toByteArray()
        android.graphics.BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
    } catch (e: Exception) {
        Log.e(TAG, "imageProxyToBitmap failed", e)
        null
    }
}