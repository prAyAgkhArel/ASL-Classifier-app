@file:OptIn(
    androidx.compose.foundation.layout.ExperimentalLayoutApi::class,
    androidx.compose.material3.ExperimentalMaterial3Api::class
)

package com.example.aslsignlanguageclassifier

import android.net.Uri
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.SegmentedButton
import androidx.compose.material3.SegmentedButtonDefaults
import androidx.compose.material3.SingleChoiceSegmentedButtonRow
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardCapitalization
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.media3.common.MediaItem
import androidx.media3.exoplayer.ExoPlayer
import androidx.media3.ui.PlayerView
import androidx.compose.material3.ExposedDropdownMenuBox
import androidx.compose.material3.ExposedDropdownMenuDefaults
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MenuAnchorType
import androidx.compose.runtime.key



data class SignSentenceVideo(
    val sentence: String,
    val videoResId: Int
)

@Composable
fun ReverseFingerspellingScreen() {
    var inputText by rememberSaveable { mutableStateOf("") }
    var mode by rememberSaveable { mutableStateOf(1) }

    var expanded by remember { mutableStateOf(false) }
    var selectedVideoResId by rememberSaveable { mutableStateOf<Int?>(null) }
    var selectedSentence by rememberSaveable { mutableStateOf("") }

    val wsSender = remember { WebSocketSender(AppConfig.esp32IP) }

    val sentenceVideos = remember {
        listOf(
            SignSentenceVideo("HELLO", R.raw.hello),
            SignSentenceVideo("HOW ARE YOU", R.raw.how_are_you),
            SignSentenceVideo("WHERE ARE YOU FROM", R.raw.where_are_you_from),
            SignSentenceVideo("NICE TO MEET YOU", R.raw.nice_to_meet_you),
            SignSentenceVideo("THANK YOU", R.raw.thank_you),
            SignSentenceVideo("WHAT ARE YOU DOING", R.raw.what_are_you_doing),
            SignSentenceVideo("WHAT IS YOUR NAME", R.raw.what_is_your_name),
            SignSentenceVideo("WHERE ARE YOU GOING", R.raw.where_are_you_going),
            SignSentenceVideo("ARE YOU MARRIED", R.raw.are_you_married)
        )
    }

    val letters = if (mode == 2) emptyList() else inputText.uppercase().filter { it.isLetter() }.toList()

    LaunchedEffect(Unit) {
        wsSender.connect()
    }

    DisposableEffect(Unit) {
        onDispose {
            wsSender.close()
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        Card(
            shape = RoundedCornerShape(24.dp),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surfaceContainerHigh
            )
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text(
                    "Reverse Fingerspelling",
                    style = MaterialTheme.typography.titleLarge,
                    fontWeight = FontWeight.Bold
                )

                Spacer(modifier = Modifier.height(8.dp))

                Text(
                    "Letter and word modes show fingerspelling. Sentence mode plays preloaded sign videos.",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )

                Spacer(modifier = Modifier.height(16.dp))

                SingleChoiceSegmentedButtonRow(modifier = Modifier.fillMaxWidth()) {
                    SegmentedButton(
                        selected = mode == 0,
                        onClick = {
                            mode = 0
                            selectedVideoResId = null
                            selectedSentence = ""
                        },
                        shape = SegmentedButtonDefaults.itemShape(index = 0, count = 3)
                    ) {
                        Text("Letter")
                    }

                    SegmentedButton(
                        selected = mode == 1,
                        onClick = {
                            mode = 1
                            selectedVideoResId = null
                            selectedSentence = ""
                        },
                        shape = SegmentedButtonDefaults.itemShape(index = 1, count = 3)
                    ) {
                        Text("Word")
                    }

                    SegmentedButton(
                        selected = mode == 2,
                        onClick = {
                            mode = 2
                            inputText = ""
                        },
                        shape = SegmentedButtonDefaults.itemShape(index = 2, count = 3)
                    ) {
                        Text("Sentence")
                    }
                }

                Spacer(modifier = Modifier.height(16.dp))

                if (mode != 2) {
                    OutlinedTextField(
                        value = inputText,
                        onValueChange = { inputText = it },
                        modifier = Modifier.fillMaxWidth(),
                        singleLine = true,
                        label = { Text("Type here") },
                        keyboardOptions = KeyboardOptions(
                            capitalization = KeyboardCapitalization.Characters
                        )
                    )

                    Spacer(modifier = Modifier.height(12.dp))

                    Button(
                        onClick = {
                            val cleaned = inputText.uppercase().trim()
                            if (cleaned.isNotBlank()) {
                                wsSender.send("SIGN:$cleaned")
                            }
                        },
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text("Show on OLED")
                    }
                } else {
                    ExposedDropdownMenuBox(
                        expanded = expanded,
                        onExpandedChange = { expanded = !expanded }
                    ) {
                        OutlinedTextField(
                            value = selectedSentence,
                            onValueChange = {},
                            readOnly = true,
                            label = { Text("Choose sentence video") },
                            trailingIcon = {
                                ExposedDropdownMenuDefaults.TrailingIcon(expanded = expanded)
                            },
                            modifier = Modifier
                                .fillMaxWidth()
                                .menuAnchor(MenuAnchorType.PrimaryNotEditable, true)
                        )

                        ExposedDropdownMenu(
                            expanded = expanded,
                            onDismissRequest = { expanded = false }
                        ) {
                            sentenceVideos.forEach { item ->
                                DropdownMenuItem(
                                    text = { Text(item.sentence) },
                                    onClick = {
                                        selectedSentence = item.sentence
                                        selectedVideoResId = item.videoResId
                                        expanded = false
                                    }
                                )
                            }
                        }
                    }
                }
            }
        }

        if (mode != 2) {
            Card(
                shape = RoundedCornerShape(24.dp),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.surfaceContainer
                )
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text("Sign Preview", style = MaterialTheme.typography.titleMedium)

                    Spacer(modifier = Modifier.height(12.dp))

                    if (letters.isEmpty()) {
                        Text(
                            "Type text to preview fingerspelling.",
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    } else {
                        FlowRow(
                            horizontalArrangement = Arrangement.spacedBy(12.dp),
                            verticalArrangement = Arrangement.spacedBy(12.dp)
                        ) {
                            letters.forEachIndexed { index, letter ->
                                key("${letter}_$index") {
                                    LetterSignCard(letter = letter)
                                }
                            }
                        }
                    }
                }
            }
        }

        if (mode == 2 && selectedVideoResId != null) {
            key(selectedVideoResId) {
                Card(
                    shape = RoundedCornerShape(24.dp),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.surfaceContainer
                    )
                ) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Text(
                            text = selectedSentence,
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.SemiBold
                        )

                        Spacer(modifier = Modifier.height(12.dp))

                        RawVideoPlayer(videoResId = selectedVideoResId!!)
                    }
                }
            }
        }
    }
}

@Composable
private fun RawVideoPlayer(videoResId: Int) {
    val context = LocalContext.current

    val exoPlayer = remember(videoResId) {
        ExoPlayer.Builder(context).build().apply {
            val videoUri = Uri.parse("android.resource://${context.packageName}/$videoResId")
            val mediaItem = MediaItem.fromUri(videoUri)
            setMediaItem(mediaItem)
            prepare()
            playWhenReady = true
        }
    }

    DisposableEffect(exoPlayer) {
        onDispose {
            exoPlayer.release()
        }
    }

    AndroidView(
        factory = { ctx ->
            PlayerView(ctx).apply {
                useController = true
                player = exoPlayer
            }
        },
        update = { playerView ->
            playerView.player = exoPlayer
        },
        modifier = Modifier
            .fillMaxWidth()
            .height(240.dp)
    )
}

@Composable
private fun LetterSignCard(letter: Char) {
    val imageRes = aslLetterImageRes(letter)

    Card(
        shape = RoundedCornerShape(20.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surface
        )
    ) {
        Column(
            modifier = Modifier.padding(12.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            if (imageRes != null) {
                Image(
                    painter = painterResource(id = imageRes),
                    contentDescription = "Sign for $letter",
                    modifier = Modifier.size(84.dp),
                    contentScale = ContentScale.Fit
                )
            } else {
                Text(
                    text = letter.toString(),
                    style = MaterialTheme.typography.headlineMedium
                )
            }

            Spacer(modifier = Modifier.height(8.dp))

            Text(
                text = letter.toString(),
                style = MaterialTheme.typography.titleMedium
            )
        }
    }
}

private fun aslLetterImageRes(letter: Char): Int? {
    return when (letter.uppercaseChar()) {
        'A' -> R.drawable.a
        'B' -> R.drawable.b
        'C' -> R.drawable.c
        'D' -> R.drawable.d
        'E' -> R.drawable.e
        'F' -> R.drawable.f
        'G' -> R.drawable.g
        'H' -> R.drawable.h
        'I' -> R.drawable.i
        'J' -> R.drawable.j
        'K' -> R.drawable.k
        'L' -> R.drawable.l
        'M' -> R.drawable.m
        'N' -> R.drawable.n
        'O' -> R.drawable.o
        'P' -> R.drawable.p
        'Q' -> R.drawable.q
        'R' -> R.drawable.r
        'S' -> R.drawable.s
        'T' -> R.drawable.t
        'U' -> R.drawable.u
        'V' -> R.drawable.v
        'W' -> R.drawable.w
        'X' -> R.drawable.x
        'Y' -> R.drawable.y
        'Z' -> R.drawable.z
        else -> null
    }
}