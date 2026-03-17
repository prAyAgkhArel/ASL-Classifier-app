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
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.util.concurrent.Executors

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
    if (hasCameraPermission) CameraPreviewScreen()
    else Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Text("Camera permission denied")
    }
}

@OptIn(ExperimentalGetImage::class)
@Composable
fun CameraPreviewScreen() {

    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current

    var handStatus by remember { mutableStateOf("No hand detected") }
    var predictionText by remember { mutableStateOf("Prediction: -") }
    var isWordMode by remember { mutableStateOf(false) }

    val SEQUENCE_LENGTH = 30
    val sequenceBuffer = remember { ArrayList<FloatArray>() }

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
            .setNumHands(1)
            .build()

        HandLandmarker.createFromOptions(context, options)
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
            interpreter.close()
            wordInterpreter.close()
            cameraExecutor.shutdown()
        }
    }

    LaunchedEffect(Unit) {

        preview.setSurfaceProvider(previewView.surfaceProvider)

        imageAnalyzer.setAnalyzer(cameraExecutor) { imageProxy ->
            try {
                val bitmap = imageProxyToBitmap(imageProxy)

                if (bitmap != null) {
                    val matrix = Matrix().apply {
                        postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())
                    }

                    val rotatedBitmap = Bitmap.createBitmap(
                        bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true
                    )

                    val mpImage = BitmapImageBuilder(rotatedBitmap).build()
                    val result = handLandmarker.detect(mpImage)

                    if (result.landmarks().isNotEmpty()) {

                        val landmarks = result.landmarks()[0]
                        val handedness = result.handednesses()[0][0].categoryName()
                        handStatus = "Hand: $handedness"

                        var normalized = normalizeLandmarksLetter(landmarks)

                        if (handedness.equals("Left", true)) {
                            normalized = mirrorNormalizedX(normalized)
                        }

                        // ================= WORD MODE =================
                        if (isWordMode) {

                            // 1. Map to 258 base features (Pose + Left Hand + Right Hand)
                            val frame258 = FloatArray(258)
                            val isLeft = handedness.equals("Left", true)

                            // Pose takes indices 0..131. Left hand starts at 132, Right hand starts at 195.
                            val offset = if (isLeft) 132 else 195

                            // Copy the 63 normalized hand features into the correct slot; the rest remain 0.0f
                            System.arraycopy(normalized, 0, frame258, offset, 63)

                            sequenceBuffer.add(frame258)

                            if (sequenceBuffer.size > SEQUENCE_LENGTH) {
                                sequenceBuffer.removeAt(0)
                            }

                            if (sequenceBuffer.size == SEQUENCE_LENGTH) {

                                // 2. Build the 516-feature vector (Base + Velocity)
                                val input516 = FloatArray(SEQUENCE_LENGTH * 516)
                                var idx = 0

                                for (i in 0 until SEQUENCE_LENGTH) {
                                    val currentFrame = sequenceBuffer[i] // FloatArray(258)
                                    // If it's the first frame, velocity is 0, so subtract it from itself
                                    val prevFrame = if (i == 0) currentFrame else sequenceBuffer[i - 1]

                                    // Add the 258 normalized positions
                                    for (j in 0 until 258) {
                                        input516[idx++] = currentFrame[j]
                                    }
                                    // Add the 258 velocity features (current - previous)
                                    for (j in 0 until 258) {
                                        input516[idx++] = currentFrame[j] - prevFrame[j]
                                    }
                                }

                                val buffer = ByteBuffer.allocateDirect(input516.size * 4).order(ByteOrder.nativeOrder())

                                for (v in input516) {
                                    buffer.putFloat(v)
                                }
                                buffer.rewind()

                                val output = Array(1) { FloatArray(wordLabels.size) }
                                wordInterpreter.run(buffer, output)

                                val scores = output[0]
                                val bestIdx = argMax(scores)
                                val conf = scores[bestIdx]

                                // Using 0.42f to match the MAYBE_ACCEPT_CONF from your Python script
                                predictionText = if (conf > 0.42f) {
                                    "Word: ${wordLabels[bestIdx]} (%.2f)".format(conf)
                                } else {
                                    "Word: low confidence"
                                }
                            } else {
                                predictionText = "Collecting: ${sequenceBuffer.size}/30"
                            }

                        }
                        // ================= LETTER MODE =================
                        else {

                            val buffer = normalized21x3ToInputBuffer(normalized)

                            val output = Array(1) { FloatArray(labels.size) }
                            interpreter.run(buffer, output)

                            val scores = output[0]
                            val bestIdx = argMax(scores)
                            val conf = scores[bestIdx]

                            predictionText =
                                if (conf > 0.80f)
                                    "Letter: ${labels[bestIdx]} (%.2f)".format(conf)
                                else
                                    "Letter: low confidence"
                        }

                    } else {
                        handStatus = "No hand"
                        predictionText = "-"
                        sequenceBuffer.clear()
                    }
                }

            } catch (e: Exception) {
                e.printStackTrace()
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
                .clickable {
                    isWordMode = !isWordMode
                    sequenceBuffer.clear()
                }
        ) {
            Text(handStatus)
            Text(predictionText)
            Text(if (isWordMode) "Mode: WORD" else "Mode: LETTER")
            Text("Tap to switch")
        }
    }
}

/* ---------------- HELPERS ---------------- */

private fun normalizeLandmarksLetter(lm: List<com.google.mediapipe.tasks.components.containers.NormalizedLandmark>): FloatArray {
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

@OptIn(ExperimentalGetImage::class)
private fun imageProxyToBitmap(image: ImageProxy): Bitmap? {
    val img = image.image ?: return null
    val y = img.planes[0].buffer
    val u = img.planes[1].buffer
    val v = img.planes[2].buffer

    val nv21 = ByteArray(y.remaining() + u.remaining() + v.remaining())
    y.get(nv21, 0, y.remaining())
    v.get(nv21, y.remaining(), v.remaining())
    u.get(nv21, y.remaining() + v.remaining(), u.remaining())

    val yuv = android.graphics.YuvImage(
        nv21, android.graphics.ImageFormat.NV21,
        image.width, image.height, null
    )

    val out = java.io.ByteArrayOutputStream()
    yuv.compressToJpeg(android.graphics.Rect(0, 0, image.width, image.height), 90, out)
    val bytes = out.toByteArray()

    return android.graphics.BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
}