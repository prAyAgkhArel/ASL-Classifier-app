import org.java_websocket.client.WebSocketClient
import org.java_websocket.handshake.ServerHandshake
import java.net.URI

class WebSocketManager(ip: String) {

    private val client = object : WebSocketClient(URI("ws://$ip/ws")) {
        override fun onOpen(handshakedata: ServerHandshake?) {
            println("Connected")
        }

        override fun onMessage(message: String?) {}

        override fun onClose(code: Int, reason: String?, remote: Boolean) {}

        override fun onError(ex: Exception?) {
            ex?.printStackTrace()
        }
    }

    fun connect() = client.connect()

    fun send(text: String) {
        if (client.isOpen) {
            client.send(text)
        }
    }
}