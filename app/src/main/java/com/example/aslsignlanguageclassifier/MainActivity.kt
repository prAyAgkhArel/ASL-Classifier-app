package com.example.aslsignlanguageclassifier

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.OptIn
import androidx.camera.core.CameraSelector
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.*
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.compose.ui.zIndex
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.util.concurrent.Executors
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row

class MainActivity : ComponentActivity() {

    private val cameraPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            setContent {
                MaterialTheme {
                    AppScreen(granted)
                }
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        val granted = ContextCompat.checkSelfPermission(
            this, Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED

        if (granted) {
            setContent { MaterialTheme { AppScreen(true) } }
        } else {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }
}

@Composable
fun AppScreen(hasCameraPermission: Boolean) {
    if (hasCameraPermission) {
        CameraPreviewScreen()
    } else {
        Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            Text("Camera permission denied")
        }
    }
}

@OptIn(ExperimentalGetImage::class)
@Composable
fun CameraPreviewScreen() {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current

    var displayStatus by remember { mutableStateOf("No hand detected") }
    var predictionText by remember { mutableStateOf("Prediction: -") }
    var isWordMode by remember { mutableStateOf(false) }

    val WORD_CAPTURE_FRAMES = 15
    val WORD_SEQUENCE_LENGTH = 30
    val MIN_CAPTURE_FRAMES = 8
    val END_ON_MISSING_HAND_FRAMES = 2
    val PREDICTION_HOLD_MS = 2500L

    var freezePredictionUntil by remember { mutableLongStateOf(0L) }

    var captureRequested by remember { mutableStateOf(false) }
    var captureActive by remember { mutableStateOf(false) }
    var captureCompleted by remember { mutableStateOf(false) }
    var missingHandFrames by remember { mutableStateOf(0) }

    val wordCaptureBuffer = remember { ArrayList<FloatArray>() }

    val previewView = remember {
        PreviewView(context).apply {
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
            scaleType = PreviewView.ScaleType.FILL_CENTER
        }
    }

    val cameraExecutor = remember { Executors.newSingleThreadExecutor() }

    val labels = remember { loadLabels(context, "letter_labels.txt") }
    val wordLabels = remember { loadLabels(context, "class_names_noface50words.txt") }

    val interpreter = remember {
        Interpreter(loadModelFile(context, "asl_letters_augmented.tflite"))
    }

    val wordInterpreter = remember {
        Interpreter(loadModelFile(context, "asl_word_bilstm_normvel50words_compatible.tflite"))
    }

    val handLandmarker = remember {
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

    val poseLandmarker = remember {
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

    val preview = remember { Preview.Builder().build() }

    val imageAnalyzer = remember {
        ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()
    }

    DisposableEffect(Unit) {
        onDispose {
            handLandmarker.close()
            poseLandmarker.close()
            interpreter.close()
            wordInterpreter.close()
            cameraExecutor.shutdown()
        }
    }

    LaunchedEffect(Unit) {
        preview.setSurfaceProvider(previewView.surfaceProvider)

        imageAnalyzer.setAnalyzer(cameraExecutor) { imageProxy ->
            try {
                val now = System.currentTimeMillis()
                val bitmap = imageProxyToBitmap(imageProxy)

                if (bitmap == null) {
                    if (now > freezePredictionUntil) {
                        displayStatus = "Bitmap null"
                        predictionText = "-"
                    }
                    return@setAnalyzer
                }

                val matrix = Matrix().apply {
                    postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())
                }

                val rotatedBitmap = Bitmap.createBitmap(
                    bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true
                )

                val mpImage = BitmapImageBuilder(rotatedBitmap).build()

                // single hand detection call per frame
                val handResult = handLandmarker.detect(mpImage)

                if (!isWordMode) {
                    if (handResult.landmarks().isNotEmpty()) {
                        val landmarks = handResult.landmarks()[0]
                        val handedness = handResult.handednesses()[0][0].categoryName()
                        displayStatus = "Hand: $handedness"

                        var normalized = normalizeLandmarksLetter(landmarks)

                        if (handedness.equals("Left", true)) {
                            normalized = mirrorNormalizedX(normalized)
                        }

                        val buffer = normalized21x3ToInputBuffer(normalized)
                        val output = Array(1) { FloatArray(labels.size) }
                        interpreter.run(buffer, output)

                        val scores = output[0]
                        val bestIdx = argMax(scores)
                        val conf = scores[bestIdx]

                        predictionText = "Letter top: ${labels[bestIdx]} (%.4f)".format(conf)
                    } else {
                        displayStatus = "No hand"
                        predictionText = "-"
                    }
                } else {
                    if (captureRequested && !captureActive) {
                        captureRequested = false
                        captureActive = true
                        captureCompleted = false
                        missingHandFrames = 0
                        wordCaptureBuffer.clear()
                        displayStatus = "Recording..."
                        predictionText = "Recording word clip..."
                    }

                    if (captureActive) {
                        val poseResult = poseLandmarker.detect(mpImage)
                        val hasHand = handResult.landmarks().isNotEmpty()

                        if (hasHand) {
                            missingHandFrames = 0
                            displayStatus = "Recording..."
                            predictionText = "Recording word clip... ${wordCaptureBuffer.size + 1}/$WORD_CAPTURE_FRAMES"

                            val frame258 = buildWordRawFrameFromPoseAndHands(
                                poseResult = poseResult,
                                handResult = handResult
                            )

                            wordCaptureBuffer.add(frame258)

                            if (wordCaptureBuffer.size >= WORD_CAPTURE_FRAMES) {
                                captureActive = false
                                captureCompleted = true

                                predictionText = runWordInferenceHolisticStyle(
                                    captureBuffer = wordCaptureBuffer,
                                    wordInterpreter = wordInterpreter,
                                    wordLabels = wordLabels
                                )
                                displayStatus = "Predicted"
                                freezePredictionUntil = now + PREDICTION_HOLD_MS
                            }
                        } else {
                            missingHandFrames += 1

                            if (missingHandFrames >= END_ON_MISSING_HAND_FRAMES &&
                                wordCaptureBuffer.size >= MIN_CAPTURE_FRAMES
                            ) {
                                captureActive = false
                                captureCompleted = true
                                missingHandFrames = 0

                                predictionText = runWordInferenceHolisticStyle(
                                    captureBuffer = wordCaptureBuffer,
                                    wordInterpreter = wordInterpreter,
                                    wordLabels = wordLabels
                                )
                                displayStatus = "Predicted"
                                freezePredictionUntil = now + PREDICTION_HOLD_MS
                            } else {
                                displayStatus = "Recording..."
                                predictionText = "Show hand to continue... ${wordCaptureBuffer.size}/$WORD_CAPTURE_FRAMES"
                            }
                        }
                    } else {
                        if (captureCompleted) {
                            if (now > freezePredictionUntil) {
                                displayStatus = "Ready for next word"
                                predictionText = "Tap Start word capture"
                            }
                        } else {
                            if (now > freezePredictionUntil) {
                                displayStatus = "Ready for word"
                                predictionText = "Tap Start word capture"
                            }
                        }
                    }
                }
            } catch (e: Exception) {
                e.printStackTrace()
                displayStatus = "Error"
                predictionText = e.javaClass.simpleName
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
    }

    Box(Modifier.fillMaxSize()) {
        AndroidView({ previewView }, Modifier.fillMaxSize())

        Column(
            modifier = Modifier
                .align(Alignment.TopCenter)
                .padding(top = 40.dp)
                .zIndex(1f),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Surface {
                Column(modifier = Modifier.padding(12.dp)) {
                    Text(displayStatus)
                    Text(predictionText)
                    Text(if (isWordMode) "Mode: WORD" else "Mode: LETTER")
                }
            }

            Spacer(modifier = Modifier.height(12.dp))

            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                Button(
                    onClick = {
                        isWordMode = !isWordMode
                        captureRequested = false
                        captureActive = false
                        captureCompleted = false
                        missingHandFrames = 0
                        wordCaptureBuffer.clear()
                        freezePredictionUntil = 0L

                        predictionText = if (isWordMode) {
                            "Switched to WORD mode"
                        } else {
                            "Switched to LETTER mode"
                        }

                        displayStatus = if (isWordMode) "Ready for word" else "Ready for letter"
                    }
                ) {
                    Text(if (isWordMode) "Switch to LETTER" else "Switch to WORD")
                }

                if (isWordMode) {
                    Button(
                        onClick = {
                            captureRequested = true
                            captureCompleted = false
                            captureActive = false
                            missingHandFrames = 0
                            freezePredictionUntil = 0L
                            displayStatus = "Ready..."
                            predictionText = "Starting capture..."
                        },
                        enabled = !captureActive
                    ) {
                        Text("Start word capture")
                    }

                    Button(
                        onClick = {
                            captureRequested = false
                            captureActive = false
                            captureCompleted = false
                            missingHandFrames = 0
                            wordCaptureBuffer.clear()
                            freezePredictionUntil = 0L
                            displayStatus = "Ready for word"
                            predictionText = "Cleared"
                        }
                    ) {
                        Text("Clear")
                    }
                }
            }
        }
    }
}

/* ---------------- HELPERS ---------------- */

private fun normalizeLandmarksLetter(
    lm: List<com.google.mediapipe.tasks.components.containers.NormalizedLandmark>
): FloatArray {
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
    scale = kotlin.math.sqrt(scale / 21).coerceAtLeast(1e-6f)

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
    for (i in 0 until 21) out[i * 3] *= -1f
    return out
}

private fun argMax(a: FloatArray): Int {
    var idx = 0
    for (i in a.indices) if (a[i] > a[idx]) idx = i
    return idx
}

private fun loadLabels(c: android.content.Context, name: String) =
    c.assets.open(name).bufferedReader().readLines()

private fun loadModelFile(c: android.content.Context, name: String): ByteBuffer {
    val fd = c.assets.openFd(name)
    val input = FileInputStream(fd.fileDescriptor)
    val ch = input.channel
    return ch.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
}

private fun buildWordRawFrameFromPoseAndHands(
    poseResult: com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult,
    handResult: com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
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
    scale = kotlin.math.sqrt(scale / 21f).coerceAtLeast(1e-6f)

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
        scale = kotlin.math.sqrt(
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
            val norm = kotlin.math.sqrt(x * x + y * y + z * z)
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
        val idx = ((i.toFloat() * (seq.size - 1)) / (targetLen - 1))
            .let { kotlin.math.round(it).toInt() }
        out.add(seq[idx].copyOf())
    }
    return out
}

private fun preprocessWordSequence(rawSeq: List<FloatArray>): Array<FloatArray> {
    val WORD_SEQ_LEN = 30
    val WORD_TRIM_START = 4

    var seq = rawSeq
    if (seq.size >= WORD_TRIM_START + WORD_SEQ_LEN) {
        seq = seq.drop(WORD_TRIM_START)
    }
    seq = resampleSequenceRaw(seq, WORD_SEQ_LEN)

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

    val batches = buildWindowBatches(captureBuffer)
    val avg = FloatArray(wordLabels.size)

    for (batch in batches) {
        val input = Array(1) { Array(WORD_SEQUENCE_LENGTH_PLACEHOLDER) { FloatArray(516) } }
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

private const val WORD_SEQUENCE_LENGTH_PLACEHOLDER = 30

@OptIn(ExperimentalGetImage::class)
private fun imageProxyToBitmap(image: ImageProxy): Bitmap? {
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

    return try {
        val yuvImage = android.graphics.YuvImage(
            nv21,
            android.graphics.ImageFormat.NV21,
            image.width,
            image.height,
            null
        )

        val out = java.io.ByteArrayOutputStream()
        yuvImage.compressToJpeg(
            android.graphics.Rect(0, 0, image.width, image.height),
            100,
            out
        )
        val imageBytes = out.toByteArray()
        android.graphics.BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
    } catch (e: Exception) {
        e.printStackTrace()
        null
    }
}