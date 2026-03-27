@file:OptIn(
    androidx.compose.material3.ExperimentalMaterial3Api::class
)

package com.example.aslsignlanguageclassifier

import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Tab
import androidx.compose.material3.TabRow
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.compose.LocalLifecycleOwner
import android.os.SystemClock

import com.example.aslsignlanguageclassifier.WebSocketSender

@Composable
fun AppEntry(hasCameraPermission: Boolean) {
    var selectedTab by rememberSaveable { mutableStateOf(0) }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Column {
                        Text("Sign Language Classifier", fontWeight = FontWeight.Bold)
                        Text(
                            "Recognition + Reverse Fingerspelling",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                }
            )
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
        ) {
            TabRow(selectedTabIndex = selectedTab) {
                Tab(
                    selected = selectedTab == 0,
                    onClick = { selectedTab = 0 },
                    text = { Text("Sign to Text") }
                )
                Tab(
                    selected = selectedTab == 1,
                    onClick = { selectedTab = 1 },
                    text = { Text("Text to Sign") }
                )
            }

            when (selectedTab) {
                0 -> {
                    if (!hasCameraPermission) {
                        PermissionDeniedScreen()
                    } else {
                        RecognitionRoute()
                    }
                }
                1 -> ReverseFingerspellingScreen()
            }
        }
    }
}


@Composable
fun RecognitionRoute() {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current

    val previewView = rememberPreviewView(context)
    val engine = remember { RecognitionEngine(context) }
    val ttsManager = remember { TextToSpeechManager(context) }
    val wsSender = remember { WebSocketSender("192.168.1.79") }
    val uiState by engine.uiState.collectAsState()

    var lastSpokenWord by remember { mutableStateOf("") }
    var lastSpokenLetter by remember { mutableStateOf("") }
    var suppressNextWordSpeak by remember { mutableStateOf(false) }

    LaunchedEffect(engine, previewView, lifecycleOwner) {
        engine.start(previewView, lifecycleOwner)
        wsSender.connect()
        kotlinx.coroutines.delay(2000)
        wsSender.send("TEXT:HELLO")
    }

    LaunchedEffect(
        uiState.lastPredictedWord,
        uiState.displayStatus,
        uiState.isWordMode
    ) {
        if (!uiState.isWordMode) return@LaunchedEffect

        val word = uiState.lastPredictedWord.trim()
        val shouldSpeak =
            uiState.displayStatus.equals("Predicted", ignoreCase = true) &&
                    word.isNotBlank() &&
                    !word.equals("None", ignoreCase = true) &&
                    !word.equals("UNKNOWN", ignoreCase = true) &&
                    word != lastSpokenWord &&
                    !suppressNextWordSpeak

        if (shouldSpeak) {
            ttsManager.speak(word)
            wsSender.send("TEXT:${word.uppercase()}")
            lastSpokenWord = word
        }

        if (suppressNextWordSpeak && uiState.displayStatus.equals("Ready for word", ignoreCase = true)) {
            suppressNextWordSpeak = false
        }
    }

    LaunchedEffect(
        uiState.predictionText,
        uiState.isWordMode
    ) {
        if (uiState.isWordMode) return@LaunchedEffect

        val letter = extractLetterForSpeech(uiState.predictionText)
        if (letter != null && letter != lastSpokenLetter) {
            ttsManager.speak(letter)
            wsSender.send("TEXT:${letter.uppercase()}")
            lastSpokenLetter = letter
        }
    }

    DisposableEffect(engine, ttsManager) {
        onDispose {
            engine.release()
            ttsManager.shutdown()
            wsSender.close()
        }
    }

    RecognitionScreen(
        previewView = previewView,
        uiState = uiState,
        onSwitchMode = {
            suppressNextWordSpeak = true
            engine.toggleMode()
        },
        onClearWord = {
            suppressNextWordSpeak = true
            engine.clearWordState()
        },
        onSpeak = {
            if (uiState.isWordMode) {
                val word = uiState.lastPredictedWord.trim()
                if (
                    word.isNotBlank() &&
                    !word.equals("None", ignoreCase = true) &&
                    !word.equals("UNKNOWN", ignoreCase = true)
                ) {
                    ttsManager.speak(word)
                    wsSender.send("TEXT:${word.uppercase()}")
                    lastSpokenWord = word
                }
            } else {
                val letter = extractLetterForSpeech(uiState.predictionText)
                if (letter != null) {
                    ttsManager.speak(letter)
                    wsSender.send("TEXT:${letter.uppercase()}")
                    lastSpokenLetter = letter
                }
            }
        }
    )
}

@Composable
private fun rememberPreviewView(context: android.content.Context): PreviewView {
    return remember {
        PreviewView(context).apply {
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
            scaleType = PreviewView.ScaleType.FILL_CENTER
        }
    }
}

@Composable
fun RecognitionScreen(
    previewView: PreviewView,
    uiState: RecognitionUiState,
    onSwitchMode: () -> Unit,
    onClearWord: () -> Unit,
    onSpeak: () -> Unit
) {
    Box(modifier = Modifier.fillMaxSize()) {
        AndroidView(
            factory = { previewView },
            modifier = Modifier.fillMaxSize()
        )

        CameraOverlay()

        Column(
            modifier = Modifier
                .fillMaxSize()
                .windowInsetsPadding(WindowInsets.safeDrawing)
                .statusBarsPadding()
                .navigationBarsPadding()
                .padding(horizontal = 16.dp, vertical = 12.dp),
            verticalArrangement = Arrangement.SpaceBetween
        ) {
            TopInfoCard(uiState = uiState)

            BottomControlsCard(
                uiState = uiState,
                onSwitchMode = onSwitchMode,
                onClearWord = onClearWord,
                onSpeak = onSpeak
            )
        }

        if (!uiState.isEngineReady) {
            EngineLoadingOverlay()
        }
    }
}

@Composable
private fun CameraOverlay() {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(
                brush = Brush.verticalGradient(
                    colors = listOf(
                        Color.Black.copy(alpha = 0.35f),
                        Color.Transparent,
                        Color.Transparent,
                        Color.Black.copy(alpha = 0.45f)
                    )
                )
            )
    )
}

@Composable
private fun TopInfoCard(uiState: RecognitionUiState) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(24.dp),
        colors = CardDefaults.cardColors(
            containerColor = Color.Black.copy(alpha = 0.62f)
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 10.dp)
    ) {
        Column(
            modifier = Modifier.padding(18.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            Text(
                text = if (uiState.isWordMode) "ASL Word Recognition" else "ASL Letter Recognition",
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.Bold,
                color = Color.White
            )

            StatusChip(
                text = uiState.displayStatus,
                isRecording = uiState.isRecording,
                hasError = uiState.errorMessage != null
            )

            PredictionPanel(predictionText = uiState.predictionText)

            Text(
                text = "Last word: ${uiState.lastPredictedWord}",
                style = MaterialTheme.typography.bodyMedium,
                color = Color(0xFFD6D6D6),
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )

            uiState.errorMessage?.let { message ->
                ErrorPanel(message = message)
            }
        }
    }
}

@Composable
private fun StatusChip(
    text: String,
    isRecording: Boolean,
    hasError: Boolean
) {
    val bgColor = when {
        hasError -> Color(0xFF7F1D1D)
        isRecording -> Color(0xFF7C2D12)
        else -> Color(0xFF0F172A)
    }

    val textColor = when {
        hasError -> Color(0xFFFECACA)
        isRecording -> Color(0xFFFED7AA)
        else -> Color(0xFFBAE6FD)
    }

    Surface(
        shape = RoundedCornerShape(16.dp),
        color = bgColor
    ) {
        Text(
            text = text,
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp),
            style = MaterialTheme.typography.bodyMedium,
            color = textColor,
            fontWeight = FontWeight.Medium
        )
    }
}

@Composable
private fun PredictionPanel(predictionText: String) {
    Surface(
        shape = RoundedCornerShape(18.dp),
        color = Color(0xFF111827).copy(alpha = 0.95f)
    ) {
        Text(
            text = predictionText,
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 14.dp, vertical = 14.dp),
            style = MaterialTheme.typography.titleMedium,
            color = Color(0xFFFFF59D),
            fontWeight = FontWeight.SemiBold
        )
    }
}

@Composable
private fun ErrorPanel(message: String) {
    Surface(
        shape = RoundedCornerShape(16.dp),
        color = Color(0xFF3F0D12).copy(alpha = 0.95f)
    ) {
        Text(
            text = message,
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 10.dp),
            style = MaterialTheme.typography.bodySmall,
            color = Color(0xFFFECACA)
        )
    }
}

@Composable
private fun BottomControlsCard(
    uiState: RecognitionUiState,
    onSwitchMode: () -> Unit,
    onClearWord: () -> Unit,
    onSpeak: () -> Unit
){
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(26.dp),
        colors = CardDefaults.cardColors(
            containerColor = Color.Black.copy(alpha = 0.66f)
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 12.dp)
    ) {
        Column(
            modifier = Modifier.padding(18.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp)
        ) {
            Text(
                text = "Controls",
                style = MaterialTheme.typography.titleMedium,
                color = Color.White,
                fontWeight = FontWeight.SemiBold
            )

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                Button(
                    onClick = onSwitchMode,
                    modifier = Modifier.weight(1f),
                    shape = RoundedCornerShape(16.dp),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = if (uiState.isWordMode) {
                            Color(0xFF1565C0)
                        } else {
                            Color(0xFF2E7D32)
                        }
                    )
                ) {
                    Text(
                        text = if (uiState.isWordMode) "Switch to LETTER" else "Switch to WORD",
                        fontWeight = FontWeight.SemiBold
                    )
                }

                Button(
                    onClick = onSpeak,
                    modifier = Modifier.weight(1f),
                    shape = RoundedCornerShape(16.dp),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = Color(0xFF6A1B9A)
                    )
                ) {
                    Text(
                        text = "Speak",
                        fontWeight = FontWeight.SemiBold
                    )
                }

                if (uiState.showClearButton || uiState.isWordMode) {
                    Button(
                        onClick = onClearWord,
                        modifier = Modifier.weight(1f),
                        shape = RoundedCornerShape(16.dp),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = Color(0xFFC62828)
                        )
                    ) {
                        Text(
                            text = "Clear",
                            fontWeight = FontWeight.SemiBold
                        )
                    }
                }
            }

            Text(
                text = if (uiState.isWordMode) {
                    "Word mode records automatically and predicts when the hand disappears or the clip completes."
                } else {
                    "Letter mode predicts continuously while a hand is visible."
                },
                style = MaterialTheme.typography.bodySmall,
                color = Color(0xFFD1D5DB)
            )
        }
    }
}

@Composable
private fun EngineLoadingOverlay() {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.Black.copy(alpha = 0.35f)),
        contentAlignment = Alignment.Center
    ) {
        Card(
            shape = RoundedCornerShape(24.dp),
            colors = CardDefaults.cardColors(
                containerColor = Color(0xFF0F172A).copy(alpha = 0.95f)
            )
        ) {
            Column(
                modifier = Modifier.padding(horizontal = 24.dp, vertical = 20.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(14.dp)
            ) {
                CircularProgressIndicator(color = Color(0xFF93C5FD))
                Text(
                    text = "Starting recognition engine...",
                    style = MaterialTheme.typography.bodyLarge,
                    color = Color.White
                )
            }
        }
    }
}

@Composable
private fun PermissionDeniedScreen() {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFF0B1220)),
        contentAlignment = Alignment.Center
    ) {
        Card(
            shape = RoundedCornerShape(24.dp),
            colors = CardDefaults.cardColors(
                containerColor = Color(0xFF111827)
            ),
            elevation = CardDefaults.cardElevation(defaultElevation = 10.dp)
        ) {
            Column(
                modifier = Modifier.padding(horizontal = 24.dp, vertical = 20.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text(
                    text = "Camera Permission Needed",
                    style = MaterialTheme.typography.titleLarge,
                    color = Color.White,
                    fontWeight = FontWeight.Bold
                )
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = "Please allow camera access to use sign recognition.",
                    style = MaterialTheme.typography.bodyMedium,
                    color = Color(0xFFD1D5DB)
                )
            }
        }
    }
}

private fun extractLetterForSpeech(predictionText: String): String? {
    val clean = predictionText.trim()

    if (
        clean.isBlank() ||
        clean == "-" ||
        clean.equals("Prediction: -", ignoreCase = true) ||
        clean.contains("low confidence", ignoreCase = true) ||
        clean.contains("no hand", ignoreCase = true)
    ) return null

    val afterColon = if (clean.contains(":")) {
        clean.substringAfter(":").trim()
    } else {
        clean
    }

    val token = afterColon
        .substringBefore("(")
        .substringBefore(" ")
        .trim()

    return token.takeIf { it.length == 1 && it[0].isLetter() }
}