package com.example.aslsignlanguageclassifier

import android.content.Context
import android.speech.tts.TextToSpeech
import java.util.Locale

class TextToSpeechManager(
    context: Context
) : TextToSpeech.OnInitListener {

    private var tts: TextToSpeech? = TextToSpeech(context.applicationContext, this)
    private var isReady = false
    private var pendingText: String? = null

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            val result = tts?.setLanguage(Locale.US)
            isReady = result != TextToSpeech.LANG_MISSING_DATA &&
                    result != TextToSpeech.LANG_NOT_SUPPORTED

            tts?.setSpeechRate(0.95f)
            tts?.setPitch(1.0f)

            pendingText?.let { text ->
                speak(text)
                pendingText = null
            }
        }
    }

    fun speak(text: String) {
        val clean = text.trim()
        if (
            clean.isBlank() ||
            clean.equals("None", ignoreCase = true) ||
            clean.equals("UNKNOWN", ignoreCase = true)
        ) return

        if (!isReady) {
            pendingText = clean
            return
        }

        tts?.speak(
            clean,
            TextToSpeech.QUEUE_FLUSH,
            null,
            "sign_language_prediction"
        )
    }

    fun stop() {
        tts?.stop()
    }

    fun shutdown() {
        tts?.stop()
        tts?.shutdown()
        tts = null
    }
}