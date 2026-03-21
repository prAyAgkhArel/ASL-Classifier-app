package com.example.aslsignlanguageclassifier

data class RecognitionUiState(
    val displayStatus: String = "Initializing...",
    val predictionText: String = "Prediction: -",
    val isWordMode: Boolean = false,
    val lastPredictedWord: String = "None",
    val isEngineReady: Boolean = false,
    val isRecording: Boolean = false,
    val showClearButton: Boolean = false,
    val errorMessage: String? = null
)