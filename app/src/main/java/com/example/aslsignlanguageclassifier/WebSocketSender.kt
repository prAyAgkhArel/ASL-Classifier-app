package com.example.aslsignlanguageclassifier

import android.util.Log
import org.java_websocket.client.WebSocketClient
import org.java_websocket.handshake.ServerHandshake
import java.net.URI

class WebSocketSender(
    esp32Ip: String
) {
    @Volatile
    private var connected = false

    private val client = object : WebSocketClient(URI("ws://$esp32Ip/ws")) {
        override fun onOpen(handshakedata: ServerHandshake?) {
            connected = true
            Log.d("WebSocketSender", "WebSocket connected")
        }

        override fun onMessage(message: String?) {
            Log.d("WebSocketSender", "Message from ESP32: $message")
        }

        override fun onClose(code: Int, reason: String?, remote: Boolean) {
            connected = false
            Log.d("WebSocketSender", "WebSocket closed: $reason")
        }

        override fun onError(ex: Exception?) {
            connected = false
            Log.e("WebSocketSender", "WebSocket error", ex)
        }
    }

    fun connect() {
        if (!connected) {
            client.connect()
        }
    }

    fun send(text: String) {
        val clean = text.trim()
        if (clean.isNotBlank() && connected && client.isOpen) {
            Log.d("WebSocketSender", "Sending: $clean")
            client.send(clean)
        } else {
            Log.d(
                "WebSocketSender",
                "Not sent. connected=$connected, open=${client.isOpen}, text='$clean'"
            )
        }
    }

    fun close() {
        connected = false
        client.close()
    }
}