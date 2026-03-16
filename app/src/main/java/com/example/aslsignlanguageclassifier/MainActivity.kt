package com.example.aslsignlanguageclassifier

import androidx.camera.core.ExperimentalGetImage
import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Size
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner

import android.graphics.Bitmap
import android.graphics.Matrix
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Text
import androidx.compose.runtime.getValue
import androidx.compose.runtime.setValue
import androidx.compose.ui.unit.dp

import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder

class MainActivity : ComponentActivity() {

    private val cameraPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            setContent {
                MaterialTheme {
                    AppScreen(hasCameraPermission = granted)
                }
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        val granted = ContextCompat.checkSelfPermission(
            this,
            Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED

        if (granted) {
            setContent {
                MaterialTheme {
                    AppScreen(hasCameraPermission = true)
                }
            }
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
        Box(
            modifier = Modifier.fillMaxSize(),
            contentAlignment = Alignment.Center
        ) {
            Text("Camera permission denied")
        }
    }
}

@OptIn(androidx.camera.core.ExperimentalGetImage::class)
@Composable
fun CameraPreviewScreen() {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current

    var handStatus by remember { androidx.compose.runtime.mutableStateOf("No hand detected") }
    var predictionText by remember { androidx.compose.runtime.mutableStateOf("Prediction: -") }

    val previewView = remember {
        PreviewView(context).apply {
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
            scaleType = PreviewView.ScaleType.FILL_CENTER
        }
    }

    val cameraExecutor = remember {
        Executors.newSingleThreadExecutor()
    }

    val labels = remember {
        loadLabels(context, "letter_labels.txt")
    }

    val interpreter = remember {
        Interpreter(loadModelFile(context, "asl_letters.tflite"))
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

    DisposableEffect(Unit) {
        onDispose {
            handLandmarker.close()
            interpreter.close()
            cameraExecutor.shutdown()

            val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
            val cameraProvider = cameraProviderFuture.get()
            cameraProvider.unbindAll()
        }
    }

    LaunchedEffect(Unit) {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        val cameraProvider = cameraProviderFuture.get()

        val preview = Preview.Builder()
            .build()
            .also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

        val imageAnalyzer = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()

        imageAnalyzer.setAnalyzer(cameraExecutor) { imageProxy ->
            try {
                val bitmap = imageProxyToBitmap(imageProxy)

                if (bitmap != null) {
                    val rotationDegrees = imageProxy.imageInfo.rotationDegrees

                    val matrix = Matrix().apply {
                        postRotate(rotationDegrees.toFloat())
                    }

                    val rotatedBitmap = Bitmap.createBitmap(
                        bitmap,
                        0,
                        0,
                        bitmap.width,
                        bitmap.height,
                        matrix,
                        true
                    )

                    val mpImage = BitmapImageBuilder(rotatedBitmap).build()
                    val result = handLandmarker.detect(mpImage)

                    if (result.landmarks().isNotEmpty()) {
                        handStatus = "Hand detected: "

                        var selectedHandIndex = 0
                        var selectedHandLabel = "Unknown"

                        val handednessList = result.handednesses()

                        for (i in handednessList.indices) {
                            val categoryList = handednessList[i]
                            if (categoryList.isNotEmpty()) {
                                val label = categoryList[0].categoryName()
                                if (label.equals("Right", ignoreCase = true)) {
                                    selectedHandIndex = i
                                    selectedHandLabel = label
                                    break
                                }
                            }
                        }

                        if (selectedHandLabel == "Unknown" && handednessList.isNotEmpty()) {
                            val firstCategoryList = handednessList[selectedHandIndex]
                            if (firstCategoryList.isNotEmpty()) {
                                selectedHandLabel = firstCategoryList[0].categoryName()
                            }
                        }

                        val firstHand = result.landmarks()[selectedHandIndex]
                        handStatus = "Hand detected: $selectedHandLabel"

                        if (!isLandmarksValidLetter(firstHand)) {
                            predictionText = "Prediction: invalid hand shape"
                        } else {
                            var normalized = normalizeLandmarksLetter(firstHand)

                            if (selectedHandLabel.equals("Left", ignoreCase = true)) {
                                normalized = mirrorNormalizedX(normalized)
                            }

                            val inputBuffer = floatArrayToInputBuffer(normalized)

                            val output = Array(1) { FloatArray(labels.size) }
                            interpreter.run(inputBuffer, output)

                            val scores = output[0]
                            val bestIdx = argMax(scores)
                            val bestLabel = labels.getOrElse(bestIdx) { "Unknown" }
                            val confidence = scores[bestIdx]

                            predictionText =
                                if (confidence >= 0.80f) {
                                    "Prediction: $bestLabel (%.2f)".format(confidence)
                                } else {
                                    "Prediction: low confidence (%.2f)".format(confidence)
                                }
                        }
                    } else {
                        handStatus = "No hand detected"
                        predictionText = "Prediction: -"
                    }
                }
            } catch (e: Exception) {
                e.printStackTrace()
            } finally {
                imageProxy.close()
            }
        }

        val cameraSelector = CameraSelector.Builder()
            .requireLensFacing(CameraSelector.LENS_FACING_FRONT)
            .build()

        try {
            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                lifecycleOwner,
                cameraSelector,
                preview,
                imageAnalyzer
            )
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    Box(modifier = Modifier.fillMaxSize()) {
        AndroidView(
            factory = { previewView },
            modifier = Modifier.fillMaxSize()
        )

        androidx.compose.foundation.layout.Column(
            modifier = Modifier
                .align(Alignment.TopCenter)
                .padding(top = 40.dp)
        ) {
            Text(
                text = handStatus,
                style = MaterialTheme.typography.headlineSmall
            )
            Text(
                text = predictionText,
                style = MaterialTheme.typography.titleMedium
            )
        }
    }
}

@androidx.annotation.OptIn(ExperimentalGetImage::class)
private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap? {
    val image = imageProxy.image ?: return null

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

    val yuvImage = android.graphics.YuvImage(
        nv21,
        android.graphics.ImageFormat.NV21,
        imageProxy.width,
        imageProxy.height,
        null
    )

    val out = java.io.ByteArrayOutputStream()
    yuvImage.compressToJpeg(
        android.graphics.Rect(0, 0, imageProxy.width, imageProxy.height),
        90,
        out
    )
    val jpegBytes = out.toByteArray()
    return android.graphics.BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.size)
}

private fun normalizeLandmarksLetter(
    landmarks: List<com.google.mediapipe.tasks.components.containers.NormalizedLandmark>
): FloatArray {
    val pts = Array(21) { FloatArray(3) }

    for (i in 0 until 21) {
        pts[i][0] = landmarks[i].x()
        pts[i][1] = landmarks[i].y()
        pts[i][2] = landmarks[i].z()
    }

    val wristX = pts[0][0]
    val wristY = pts[0][1]
    val wristZ = pts[0][2]

    for (i in 0 until 21) {
        pts[i][0] -= wristX
        pts[i][1] -= wristY
        pts[i][2] -= wristZ
    }

    var scale = kotlin.math.sqrt(
        pts[9][0] * pts[9][0] +
                pts[9][1] * pts[9][1] +
                pts[9][2] * pts[9][2]
    )

    if (scale < 1e-6f) {
        var sum = 0f
        var count = 0
        for (i in 0 until 21) {
            val norm = kotlin.math.sqrt(
                pts[i][0] * pts[i][0] +
                        pts[i][1] * pts[i][1] +
                        pts[i][2] * pts[i][2]
            )
            sum += norm
            count++
        }
        scale = (sum / count).coerceAtLeast(1e-6f)
    }

    val out = FloatArray(63)
    var idx = 0
    for (i in 0 until 21) {
        out[idx++] = pts[i][0] / scale
        out[idx++] = pts[i][1] / scale
        out[idx++] = pts[i][2] / scale
    }
    return out
}

private fun isLandmarksValidLetter(
    landmarks: List<com.google.mediapipe.tasks.components.containers.NormalizedLandmark>
): Boolean {
    var minX = Float.MAX_VALUE
    var maxX = -Float.MAX_VALUE
    var minY = Float.MAX_VALUE
    var maxY = -Float.MAX_VALUE

    for (lm in landmarks) {
        val x = lm.x()
        val y = lm.y()
        if (x < minX) minX = x
        if (x > maxX) maxX = x
        if (y < minY) minY = y
        if (y > maxY) maxY = y
    }

    val boxArea = (maxX - minX) * (maxY - minY)
    return boxArea in 0.01f..0.8f
}

private fun floatArrayToInputBuffer(values: FloatArray): ByteBuffer {
    val buffer = ByteBuffer.allocateDirect(values.size * 4).order(ByteOrder.nativeOrder())
    for (v in values) buffer.putFloat(v)
    buffer.rewind()
    return buffer
}

private fun loadLabels(context: android.content.Context, fileName: String): List<String> {
    return context.assets.open(fileName).bufferedReader().useLines { lines ->
        lines.map { it.trim() }.filter { it.isNotEmpty() }.toList()
    }
}

private fun loadModelFile(context: android.content.Context, fileName: String): ByteBuffer {
    val fileDescriptor = context.assets.openFd(fileName)
    val inputStream = java.io.FileInputStream(fileDescriptor.fileDescriptor)
    val fileChannel = inputStream.channel
    val startOffset = fileDescriptor.startOffset
    val declaredLength = fileDescriptor.declaredLength
    return fileChannel.map(java.nio.channels.FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
}

private fun mirrorNormalizedX(features: FloatArray): FloatArray {
    val out = features.copyOf()

    // 21 landmarks * (x, y, z)
    for (i in 0 until 21) {
        val xIndex = i * 3
        out[xIndex] = -out[xIndex]
    }

    return out
}

//private fun handLandmarksToInputBuffer(
//    landmarks: List<com.google.mediapipe.tasks.components.containers.NormalizedLandmark>
//): ByteBuffer {
//    val inputBuffer = ByteBuffer.allocateDirect(63 * 4).order(ByteOrder.nativeOrder())
//
//    for (lm in landmarks) {
//        inputBuffer.putFloat(lm.x())
//        inputBuffer.putFloat(lm.y())
//        inputBuffer.putFloat(lm.z())
//    }
//
//    inputBuffer.rewind()
//    return inputBuffer
//}

private fun argMax(array: FloatArray): Int {
    var bestIdx = 0
    var bestVal = array[0]
    for (i in 1 until array.size) {
        if (array[i] > bestVal) {
            bestVal = array[i]
            bestIdx = i
        }
    }
    return bestIdx
}